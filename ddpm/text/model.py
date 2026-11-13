import torch
import torch.nn as nn

from ddpm.layers import (
    DiTBlock,
    FinalLayer,
    GaussianFourierProjection,
    build_time_embedding,
    sincos_1d,
)
from ddpm.text.config import TextDiTModelConfig


def build_token_embedding(V: int, D: int) -> nn.Embedding:
    # V characters, plus mask_id = V
    return nn.Embedding(V + 1, D)


def get_pos_embed(L: int, embed_dim: int) -> torch.Tensor:
    """Fixed sinusoidal positions, (1, L, embed_dim)."""
    emb = sincos_1d(torch.arange(L, dtype=torch.long), embed_dim)

    return emb.unsqueeze(0)


class TextDiT(nn.Module):
    """DiT (Peebles & Xie 2023) over character tokens with patch size 1: bidirectional and
    adaLN-conditioned on t for MDLM, causal and unconditioned for the AR baseline."""

    def __init__(
        self, model_config: TextDiTModelConfig, V: int, L: int, causal: bool, time_cond: bool
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.token_embedding = build_token_embedding(V, model_config.embed_dim)

        self.register_buffer(
            "pos_embed",
            get_pos_embed(L, model_config.embed_dim),
            persistent=False,
        )

        self.causal = causal
        if self.causal:
            # (L, L) bool, True above the diagonal: position i can't attend to j > i
            self.register_buffer(
                "mask",
                torch.triu(torch.ones((L, L), dtype=torch.bool), diagonal=1),
                persistent=False,
            )

        self.time_cond = time_cond
        if self.time_cond:
            self.time_proj = GaussianFourierProjection(model_config.embed_dim, scale=16.0)
            self.time_mlp = build_time_embedding(model_config.embed_dim)

        self.blocks = torch.nn.ModuleList(
            [
                DiTBlock(
                    model_config.embed_dim,
                    model_config.heads,
                    model_config.mlp_ratio,
                    model_config.dropout,
                    gate_init=model_config.gate_init,
                )
                for _ in range(model_config.depth)
            ]
        )
        self.final = FinalLayer(model_config.embed_dim, 1, V)

    def forward(self, z: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
        """z: (B, L) long, with mask_id = V; t: (B,) float, or None without time conditioning
        -> (B, L, V) float logits."""
        B = z.shape[0]

        if self.time_cond:
            assert t is not None
            c = self.time_mlp(self.time_proj(t))
        else:  # adaLN sees a constant 0
            c = torch.zeros((B, self.model_config.embed_dim), device=z.device, dtype=torch.float)

        tokens: torch.Tensor = self.token_embedding(z) + self.pos_embed
        for block in self.blocks:
            if self.causal:
                tokens = block(tokens, c, self.mask)
            else:
                tokens = block(tokens, c)
        tokens = self.final(tokens, c)

        return tokens
