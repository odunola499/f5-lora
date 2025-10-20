import torch
import math
from torch import nn, Tensor
from .text import TextEmbedding
from .utils import RotaryEmbedding
from .block import DiTBlock, AdaLayerNorm_Final

class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size = 31, groups = 16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.Mish()
        )

    def forward(self, x:Tensor, mask:bool = None):
        if mask is not None:
            mask = mask[...,None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim,dim)
        )

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep).to(dtype = timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x,cond, text_embed, drop_audio_cond = False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        x = torch.cat((x, cond, text_embed), dim = -1)
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x



class DIT(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 depth = 8,
                 heads = 8,
                 dim_head = 64,
                 dropout = 0.1,
                 ff_mult = 4,
                 mel_dim = 100,
                 text_num_embeds = 256,
                 text_dim = 512,
                 text_mask_padding = True,
                 text_embedding_average_upsampling = False,
                 qk_norm = None,
                 conv_layers = 0,
                 pe_attn_head = None,
                 attn_backend = 'torch',
                 attn_mask_enabled = False,
                 long_skip_connection = False,
                 checkpoint_activations = False,):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim

        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding = text_mask_padding,
            average_upsampling = text_embedding_average_upsampling,
            conv_layers = conv_layers
        )

        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

            # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(self,
                        x,
                        cond,
                        text,
                        drop_audio_cond:bool = False,
                        drop_text:bool = False,
                        cache:bool = True,
                        audio_mask:bool = None):
        seq_len = x.shape[1]
        if cache:
            if drop_text:
                text_embed = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                self.text_uncond = text_embed
            else:
                text_embed = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                self.text_cond = text_embed
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: Tensor,  # noised input audio
        cond: Tensor,  # masked cond audio
        text: Tensor,  # text
        time: Tensor,  # time step
        mask = None,  # noqa:
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch, seq_len = x.shape[:2]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.concat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask, rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output

