import torch
from torch import nn, Tensor
from utils import GRN
from torch.nn import functional as F

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

def precompute_freqs(dim, end, theta = 10000, theta_rescale_factor = 1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs =  1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device = freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim = -1)

class ConvNextV2Block(nn.Module):
    def __init__(self, dim:int, intermediate_dim:int, dilation:int = 1):
        super().__init__()
        padding = (dilation * (7-1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding = padding, groups = dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps = 1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x:Tensor):
        residual = x
        x = x.transpose(1,2)
        x = self.dwconv(x)
        x = x.transpose(1,2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x + residual
        return x

class TextEmbedding(nn.Module):
    def __init__(self,
                 text_num_embeds, text_dim, mask_padding = True, average_upsampling = False, conv_layers = 4, conv_mult = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling
        if average_upsampling:
            assert mask_padding, "mask_padding must be True if average_upsampling is True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freq_cis", precompute_freqs(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNextV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch_size, text_len, text_dim = text.shape

        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype = torch.bool)

        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim = 1)
        valid_lens = valid_mask.sum(dim = 1)

        upsampled_text = torch.zeros_like(text) #likely bug

        for i in range(batch_size):
            audio_len = audio_lens[i].item()
            valid_len = valid_lens[i].item()
            if valid_len == 0:
                continue

            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]

            base_repeat = audio_len // valid_len
            remainder = audio_len % valid_len

            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype = torch.long)
            upsampled = valid_data[indices]

            upsampled_text[i, :audio_len, :] = upsampled
        return upsampled_text

    def forward(self,text:Tensor, seq_len, drop_text = False, audio_mask = None):
        text_mask = None
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value = 0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text: # Classifier free Guidance
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), device = text.device, dtype = torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freq_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.upsample_text_by_mask(text, ~text_mask, audio_mask)

        return text

