import torch

from ddpm.image.config import (
    Config,
    UNetModelConfig,
)
from ddpm.image.models.base import BaseModel
from ddpm.image.models.embedding import build_positional_embedding
from ddpm.layers import build_time_embedding, modulate


class ResBlock(torch.nn.Module):
    """conv -> GroupNorm -> FiLM by the timestep embedding -> SiLU, plus a shortcut. Deviation:
    DDPM's block is pre-activation (GroupNorm -> SiLU -> conv, twice) and adds the embedding as a
    bias between the convs; the scale-shift is AdaGN (Dhariwal & Nichol 2021, §3.1).
    Forward: x (B, in_ch, H, W), t_emb (B, embd_dim) -> (B, out_ch, H, W)."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, num_groups: int, embd_dim: int
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, padding=int(kernel_size // 2))
        self.norm = torch.nn.GroupNorm(num_groups, out_ch)
        self.nonlinearity = torch.nn.SiLU()
        self.emb_linear = torch.nn.Linear(embd_dim, 2 * out_ch)

        self.shortcut: torch.nn.Module
        if in_ch != out_ch:
            self.shortcut = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        emb_matched = self.emb_linear(self.nonlinearity(t_emb))  # DDPM's dense(nonlinearity(temb))
        scale, shift = emb_matched.chunk(2, dim=1)  # 2 x (B, out_ch)
        x1 = self.conv(x)
        x1 = self.norm(x1)
        x1 = modulate(x1, shift, scale)
        x1 = self.nonlinearity(x1)
        x = self.shortcut(x) + x1

        return x


class AttentionBlock(torch.nn.Module):
    """Residual self-attention over the H * W positions. Deviation: no GroupNorm before it (DDPM
    has one). ``t_emb`` is unused; it keeps the signature of ``ResBlock``. (B, C, H, W) -> same."""

    def __init__(self, num_ch: int, num_heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(num_ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape((B, C, H * W)).transpose(-1, -2)  # (B, H * W, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(-1, -2).reshape((B, C, H, W))
        x = x + attn_out

        return x


class DownStage(torch.nn.Module):
    """Blocks, then downsample. Forward -> (downsampled x, skip: x before downsampling)."""

    def __init__(self, blocks: list[torch.nn.Module], downsample: torch.nn.Module) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, t_emb)
        skip = x
        x = self.downsample(x)

        return x, skip


class UpStage(torch.nn.Module):
    """Upsample, then blocks. Its input already has the stage's skip concatenated on channels."""

    def __init__(self, upsample: torch.nn.Module, blocks: list[torch.nn.Module]) -> None:
        super().__init__()
        self.upsample = upsample
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x, t_emb)

        return x


def _res_blocks(
    in_ch: int, out_ch: int, num_res_blocks: int, num_groups: int, embd_dim: int
) -> list[torch.nn.Module]:
    blocks: list[torch.nn.Module] = [ResBlock(in_ch, out_ch, 3, num_groups, embd_dim)]
    blocks += [ResBlock(out_ch, out_ch, 3, num_groups, embd_dim) for _ in range(num_res_blocks - 1)]
    return blocks


def _build_down_stages(
    model_config: UNetModelConfig, stage_channels: list[int]
) -> torch.nn.ModuleList:
    num_stages = len(stage_channels)
    down_stages: list[torch.nn.Module] = []
    in_ch = model_config.num_ch
    for i, ch in enumerate(stage_channels):
        blocks = _res_blocks(
            in_ch, ch, model_config.num_res_blocks, model_config.num_groups, model_config.embd_dim
        )
        if i == model_config.attention_stage_index:
            blocks.append(AttentionBlock(ch, model_config.num_heads))
        is_last = i == num_stages - 1
        # Deviation: a 1x1 stride-2 conv keeps 1 pixel in 4 (DDPM: 3x3 stride 2).
        downsample: torch.nn.Module = (
            torch.nn.Identity() if is_last else torch.nn.Conv2d(ch, ch, kernel_size=1, stride=2)
        )
        down_stages.append(DownStage(blocks, downsample))
        in_ch = ch
    return torch.nn.ModuleList(down_stages)


def _build_bottleneck(model_config: UNetModelConfig, bottleneck_ch: int) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [
            ResBlock(
                bottleneck_ch, bottleneck_ch, 3, model_config.num_groups, model_config.embd_dim
            ),
            AttentionBlock(bottleneck_ch, model_config.num_heads),
            ResBlock(
                bottleneck_ch, bottleneck_ch, 3, model_config.num_groups, model_config.embd_dim
            ),
        ]
    )


def _build_up_stages(
    model_config: UNetModelConfig, stage_channels: list[int]
) -> torch.nn.ModuleList:
    num_stages = len(stage_channels)
    up_stages: list[torch.nn.Module] = []
    in_ch = stage_channels[-1]
    for i in reversed(range(num_stages)):
        concat_ch = in_ch + stage_channels[i]
        is_top = i == 0
        target_ch = stage_channels[i] if is_top else stage_channels[i - 1]
        target_level = i if is_top else i - 1

        upsample: torch.nn.Module
        if is_top:
            upsample = torch.nn.Identity()
            blocks = _res_blocks(
                concat_ch,
                target_ch,
                model_config.num_res_blocks,
                model_config.num_groups,
                model_config.embd_dim,
            )
        else:
            # Deviation: a transposed conv (DDPM: nearest-neighbour x2, then a 3x3 conv).
            upsample = torch.nn.ConvTranspose2d(concat_ch, target_ch, kernel_size=2, stride=2)
            blocks = _res_blocks(
                target_ch,
                target_ch,
                model_config.num_res_blocks,
                model_config.num_groups,
                model_config.embd_dim,
            )

        if target_level == model_config.attention_stage_index:
            blocks.append(AttentionBlock(target_ch, model_config.num_heads))

        up_stages.append(UpStage(upsample, blocks))
        in_ch = target_ch
    return torch.nn.ModuleList(up_stages)


def _build_out_head(
    model_config: UNetModelConfig, final_ch: int, out_ch: int
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    out_norm = torch.nn.GroupNorm(num_groups=model_config.num_groups, num_channels=final_ch)
    out_act = torch.nn.SiLU()
    # Its input is the last stage concatenated with init_conv's output (the stem skip).
    out_conv = torch.nn.Conv2d(final_ch + model_config.num_ch, out_ch, kernel_size=1)
    return out_norm, out_act, out_conv


class UNetWideResNetAttnCIFAR10(BaseModel):
    """DDPM's UNet (Ho et al. 2020, Appendix B): Wide-ResNet blocks, GroupNorm, self-attention at
    one resolution and in the bottleneck, sinusoidal timestep embedding. Deviations: see
    ``ResBlock``, ``AttentionBlock`` and the resampling layers. Also one skip per stage rather than
    per block, 1x1 stem and head convs, and a stem skip into the head."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert isinstance(self.config.model, UNetModelConfig)
        model_config = self.config.model
        embd_dim = model_config.embd_dim

        self.embd_dim = embd_dim
        self.init_conv = torch.nn.Conv2d(config.data.num_channels, model_config.num_ch, 1)
        self.embd_model = build_time_embedding(embd_dim)

        stage_channels = [model_config.num_ch * mult for mult in model_config.channel_mults]
        self.down_stages = _build_down_stages(model_config, stage_channels)
        self.bottleneck = _build_bottleneck(model_config, stage_channels[-1])
        self.up_stages = _build_up_stages(model_config, stage_channels)
        self.out_norm, self.out_act, self.out_conv = _build_out_head(
            model_config, stage_channels[0], config.data.num_channels
        )

        if self.config.guidance is not None:
            self.label_embd = torch.nn.Embedding(config.data.num_classes + 1, embd_dim)

        self.positional_embedding = build_positional_embedding(embd_dim, self.config.noise_schedule)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, C, H, W) in [-1, 1]; t: (B,) or (B, 1), long in [0, T) or float32 in [0, 1];
        lbl: (B,) long, index num_classes = the CFG null class, or None -> eps: (B, C, H, W)."""
        B = x.shape[0]
        t = t.reshape(B, 1)
        pe = self.positional_embedding(t)

        learned_pe = self.embd_model(pe)  # (B, embd_dim)

        if lbl is not None:
            learned_pe += self.label_embd(lbl)

        x = self.init_conv(x)  # (B, num_ch, H, W)
        skip0 = x

        skips: list[torch.Tensor] = []
        for down_stage in self.down_stages:
            x, skip = down_stage(x, learned_pe)  # skip i: (B, num_ch * mult_i, H / 2^i, W / 2^i)
            skips.append(skip)

        for block in self.bottleneck:
            x = block(x, learned_pe)

        for up_stage in self.up_stages:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up_stage(x, learned_pe)

        x = self.out_norm(x)  # (B, num_ch * mult_0, H, W)
        x = self.out_act(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.out_conv(x)

        return x
