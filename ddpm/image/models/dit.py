import torch
import torch.nn as nn

from ddpm.image.config import Config, DiTModelConfig
from ddpm.image.models.base import BaseModel
from ddpm.image.models.embedding import build_positional_embedding
from ddpm.layers import DiTBlock, FinalLayer, build_time_embedding, sincos_1d


class PatchEmbed(nn.Module):
    """Patchify (DiT §3.2): a stride-p conv. (B, C, H, W) -> (B, N = (H / p)^2, hidden_dim)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N=grid_size**2, hidden_dim)

        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding (DiT §3.2, ViT's sine-cosine version): half the
    channels encode the row, half the column. -> (grid_size**2, embed_dim), row-major over the
    patch grid."""
    assert embed_dim % 4 == 0
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")

    half = embed_dim // 2
    emb_h = sincos_1d(grid_y.reshape(-1), half)
    emb_w = sincos_1d(grid_x.reshape(-1), half)

    return torch.cat([emb_h, emb_w], dim=1)  # (grid_size**2, embed_dim)


def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int, grid_size: int) -> torch.Tensor:
    """(B, N, p * p * C) -> (B, C, H, W). Reads each token's last dim as (patch_h, patch_w, C)."""
    B = x.shape[0]
    x = x.reshape(B, grid_size, grid_size, patch_size, patch_size, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, grid_h, patch_h, grid_w, patch_w)
    img_size = grid_size * patch_size

    return x.reshape(B, out_channels, img_size, img_size)


class DiT(BaseModel):
    """DiT with adaLN-Zero blocks (Peebles & Xie 2023). Deviations: pixel space instead of VAE
    latents, and it predicts eps only (the paper also learns the variance)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert isinstance(self.config.model, DiTModelConfig)
        model_config = self.config.model
        embd_dim = model_config.embd_dim
        assert config.data.width == config.data.height

        # cached so forward() needn't re-narrow the config.model union
        self.patch_size = model_config.patch_size
        self.out_channels = config.data.num_channels

        self.patch_embed = PatchEmbed(
            config.data.width,
            model_config.patch_size,
            config.data.num_channels,
            embd_dim,
        )
        # (1, N, D). A buffer so .to(device) moves it; persistent=False keeps it out of state_dict.
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(embd_dim, self.patch_embed.grid_size).unsqueeze(0),
            persistent=False,
        )

        # adaLN's c (DiT §3.2): the timestep embedding, plus the class embedding if any, (B, D).
        self.positional_embedding = build_positional_embedding(embd_dim, config.noise_schedule)
        self.embd_model = build_time_embedding(embd_dim)

        self.blocks = torch.nn.ModuleList(
            [
                DiTBlock(embd_dim, model_config.heads, model_config.mlp_ratio)
                for _ in range(model_config.depth)
            ]
        )
        self.final = FinalLayer(embd_dim, model_config.patch_size, config.data.num_channels)

        if self.config.guidance is not None:
            self.label_embd = torch.nn.Embedding(config.data.num_classes + 1, embd_dim)

    def forward(
        self, img: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        """img: (B, C, H, W) in [-1, 1]; t: (B,) or (B, 1), long in [0, T);
        lbl: (B,) long, index num_classes = the CFG null class, or None -> eps: (B, C, H, W)."""
        B = img.shape[0]

        c = self.embd_model(self.positional_embedding(t.reshape(B, 1)))
        if lbl is not None:
            c = c + self.label_embd(lbl)

        tokens = self.patch_embed(img) + self.pos_embed  # (B, N, D)
        for block in self.blocks:
            tokens = block(tokens, c)
        tokens = self.final(tokens, c)  # (B, N, p * p * C)

        return unpatchify(tokens, self.patch_size, self.out_channels, self.patch_embed.grid_size)
