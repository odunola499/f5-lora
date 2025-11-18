import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
import torchaudio
from librosa.filters import mel as librosa_mel_fn
import re

mel_basis_cache = {}
hann_window_cache = {}

def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)

def mask_from_start_end_indices(seq_len, start, end):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def chunk_text(text, max_chars = 135):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def lens_to_mask(t, length: int | None = None):
    if not length:
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
):
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def list_str_to_tensor(text: list[str], padding_value=-1):
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = torch.nn.utils.rnn.pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text

def mask_from_frac_lengths(seq_len, frac_lengths: Tensor):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


@torch.autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.concat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)



def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel

def manual_euler(func, x0:Tensor, t:Tensor):
    x = x0.clone() # Initial, currently would be just noice
    xs = [x0]

    for i in range(len(t) - 1):
        t_i = t[i]
        t_next = t[i+1]
        dt = t_next - t_i

        dx = func(t_i, x)
        x = x + dt * dx
        xs.append(x)
    return xs




class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p = 2, dim = 1, keepdim = True)
        Nx = Gx / (Gx.mean(dim = -1, keepdim = True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        weight = self.weight
        with torch.autocast(dtype = torch.float32, device_type=x.device.type):
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=weight, eps=self.eps)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @torch.autocast('cuda', enabled = False)
    def forward(self, t, offset = 0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not self.scale:
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = torch.stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale



class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
        device = None
    ):
        super().__init__()
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        # if mel_spec_type == "vocos":
        #     self.extractor = get_vocos_mel_spectrogram
        # elif mel_spec_type == "bigvgan":
        #     self.extractor = get_bigvgan_mel_spectrogram

        self.extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mel_channels,
                power=1,
                center=True,
                normalized=False,
                norm=None,
        )


        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def preprocess(self, waveform):
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)

        assert len(waveform.shape) == 2
        return waveform


    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        wav = self.preprocess(wav)
        init_dtype = wav.dtype
        wav = wav.half()
        with torch.no_grad():
            mel = self.extractor(wav)
        mel = mel.to(dtype = init_dtype)
        mel = mel.clamp(min=1e-5).log()
        return mel


