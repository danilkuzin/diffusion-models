import math

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM (Perez et al. 2018): ``x * (1 + scale) + shift``.

    x: tokens (B, N, D) or images (B, C, H, W); shift, scale: (B, D) or (B, C), broadcast as
    (B, 1, D) or (B, C, 1, 1). Argument order as in DiT; swapping shift and scale doesn't raise.
    """
    if x.ndim == 4:
        shift = shift[:, :, None, None]
        scale = scale[:, :, None, None]
    else:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero (Peebles & Xie 2023, §3.2, Fig. 3). Forward: x (B, N, D), c (B, D),
    optional attn_mask (N, N) -> (B, N, D). gate_init: both gates' bias at init, 0.0 is adaLN-Zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        # adaLN-Zero: weight and bias start at 0, so every gate is 0 and the block is the identity
        # at init. gate_init > 0 deviates: the gates start there, shift and scale still at 0.
        adaln_linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        nn.init.zeros_(adaln_linear.weight)
        nn.init.zeros_(adaln_linear.bias)
        with torch.no_grad():  # chunk order as in forward: shift, scale, gate (msa), then (mlp)
            adaln_linear.bias[2 * hidden_dim : 3 * hidden_dim].fill_(gate_init)
            adaln_linear.bias[5 * hidden_dim : 6 * hidden_dim].fill_(gate_init)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), adaln_linear)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mods = self.adaln_modulation(c).chunk(6, dim=1)  # 6 x (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

        y = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(y, y, y, need_weights=False, attn_mask=attn_mask)
        attn_out = self.dropout_attn(attn_out)
        x = x + gate_msa.unsqueeze(1) * attn_out

        z = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(z)
        mlp_out = self.dropout_mlp(mlp_out)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class FinalLayer(nn.Module):
    """DiT's last layer (§3.2, "transformer decoder"): adaLN, then a linear to one patch per
    token. Forward: x (B, N, D), c (B, D) -> (B, N, patch_size**2 * out_channels)."""

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        # shift and scale only: there is no residual branch to gate
        adaln_linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(adaln_linear.weight)
        nn.init.zeros_(adaln_linear.bias)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), adaln_linear)

        # zero-init the projection too, as DiT's code does: the untrained net predicts zero noise
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (B, N, patch_size**2 * out_channels)

        return x


def sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding (Vaswani et al. 2017, §3.5) as [sin | cos] halves, not interleaved.
    pos: (N,) -> (N, dim) float32."""
    assert dim % 2 == 0
    freq_dim = dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
    args = pos.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, dim // 2)

    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class GaussianFourierProjection(nn.Module):
    """Random Fourier features of a continuous t (Tancik et al. 2020), as in Song et al. 2021's
    NCSN++. W ~ N(0, scale^2) is fixed. Forward: t with B elements, float32 -> (B, embed_dim)
    as [sin | cos]."""

    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)

        proj = t * self.W[None, :] * 2 * math.pi  # (B, embed_dim // 2)

        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SinusoidalPositionalEmbedding(nn.Module):
    """DDPM's timestep embedding (Ho et al. 2020, Appendix B): the Transformer encoding (Vaswani
    et al. 2017, §3.5). Forward: t (B, 1) long in [0, T) -> (B, embd_dim) float32."""

    def __init__(self, embd_dim: int) -> None:
        super().__init__()
        self.embd_dim = embd_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        i = torch.arange(self.embd_dim // 2, device=t.device).float()
        inv_frequency = t / torch.pow(10000, (2 * i) / self.embd_dim)  # (B, embd_dim // 2)

        pe_even = torch.sin(inv_frequency)
        pe_odd = torch.cos(inv_frequency)

        # [sin | cos] rather than interleaved: a linear layer follows, so the order doesn't matter
        pe = torch.cat([pe_even, pe_odd], dim=-1)

        return pe


def build_time_embedding(embd_dim: int) -> nn.Module:
    """The MLP on top of the raw timestep embedding, Linear -> SiLU -> Linear as in DDPM's and
    DiT's code: (B, embd_dim) -> (B, embd_dim). No trailing SiLU: each consumer applies its own,
    like DDPM's ``dense(nonlinearity(temb))`` and adaLN's SiLU -> Linear."""
    return nn.Sequential(
        nn.Linear(embd_dim, embd_dim),
        nn.SiLU(),
        nn.Linear(embd_dim, embd_dim),
    )
