import torch
import torch.nn as nn

from ddpm.image.config import AnyScheduleConfig, DiscreteScheduleConfig
from ddpm.layers import SinusoidalPositionalEmbedding


class ScaledSinusoidalEmbedding(SinusoidalPositionalEmbedding):
    """DDPM's sinusoid on 999 t, so continuous t in [0, 1] covers the discrete index range, as the
    score SDE and rectified flow reference code feed their models (``labels = t * 999``).
    Forward: t (B, 1) float32 -> (B, embd_dim)."""

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return super().forward(999.0 * t)


def build_positional_embedding(embd_dim: int, noise_schedule: AnyScheduleConfig) -> nn.Module:
    """The timestep embedding for the schedule family: DDPM's sinusoid on long t in [0, T) for
    discrete schedules, on 999 t for continuous ones (float32 t in [0, 1])."""
    if isinstance(noise_schedule, DiscreteScheduleConfig):
        return SinusoidalPositionalEmbedding(embd_dim)
    return ScaledSinusoidalEmbedding(embd_dim)
