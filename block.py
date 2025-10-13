import torch
from torch import nn, Tensor
from utils import GRN, apply_rotary_pos_emb
from torch.nn import functional as F


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps = 1e-6)

    def forward(self, x, emb = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4, dropout = 0.0, approximate:str = 'none'):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_ln = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(
            project_ln,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self,x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self,
                 processor,
                 dim:int,
                 heads:int = 8,
                 dim_head:int = 64,
                 dropout:float = 0.0,
                 context_dim:int = None,
                 context_pre_only = False,
                 qk_norm = None,
    ):
        super().__init__()
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if not qk_norm:
            self.q_norm = None
            self.k_norm = None

        elif qk_norm == 'rms_norm':
            from utils import RMSNorm

            self.q_norm = RMSNorm(dim_head, eps = 1e-6)
            self.k_norm = RMSNorm(dim_head, eps = 1e-6)

        else:
            raise ValueError(f'Unknown qk_norm: {qk_norm}')

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ])

    def forward(self,
                x, c = None, mask:bool = None, rope = None, c_rope = None):
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


class AttnProcessor:
    def __init__(self,
                 pe_attn_head=None,
                 attn_backend: str = 'torch',
                 attn_mask_enabled=True):
        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(self,
                 attn: Attention,
                 x: Tensor,  # noised input
                 mask = None,
                 rope = None):
        batch_size = x.shape[0]
        #todo: add flash attention support

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.q_norm is not None:
            query = attn.q_norm(query)
            key = attn.k_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_mask_enabled and mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None # bidirectional attention

        x = F.scaled_dot_product_attention(query, key, value, attn_mask = attn_mask, dropout_p = 0.0, is_causal=False)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)

        x = x.to(query.dtype)
        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


class DiTBlock(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 ff_mult = 4,
                 dropout = 0.1,
                 qk_norm = None,
                 pe_attn_head = None,
                 attn_backend = 'torch',
                 attn_mask_enabled = True,):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps = 1e-6)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask = None, rope = None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x



