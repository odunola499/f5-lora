import torch
from torch.nn import functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from .utils import get_epss_timesteps, lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths, MelSpec
from .utils import manual_euler
from random import random
from .lora import LoraManager


class CFM(nn.Module):
    def __init__(self,
                 transformer:nn.Module,
                 sigma = 0.0,
                 odeint_kwargs=None,
                 audio_drop_prob = 0.3,
                 cond_drop_prob = 0.2,
                 mel_spec_kwargs=None,
                 frac_lengths_mask = (0.1, 0.7),
                 vocab_char_map = None):

        super().__init__()
        if mel_spec_kwargs is None:
            mel_spec_kwargs = {}
        if odeint_kwargs is None:
            odeint_kwargs = {
                'method': 'euler'
            }

        self.frac_lengths_mask = frac_lengths_mask
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self.mel_spec.n_mel_channels

        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
            self,
            cond,
            text,
            duration,
            *,
            lens = None,
            steps=32,
            cfg_strength=1.0,
            sway_sampling_coef=None,
            seed: int | None = None,
            max_duration=4096,
            vocoder = None,
            use_epss=True,
            no_ref_audio=False,
            duplicate_test=False,
            t_inter=0.1,
            edit_mask=None,
    ):
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len = cond.shape[:2]
        device = cond.device
        if not lens:
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if self.vocab_char_map:
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if seed:
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        #trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        trajectory = manual_euler(fn, y0, t)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if vocoder:
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
            self,
            inp,  # mel or raw wave
            text,  # noqa: F722
            *,
            lens = None,  # noqa: F821
            noise_scheduler: str | None = None,
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]
        dtype = inp.dtype
        device = inp.device

        if isinstance(text, list):
            if self.vocab_char_map:
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if lens is None:
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if mask is not None:
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=x_t, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
