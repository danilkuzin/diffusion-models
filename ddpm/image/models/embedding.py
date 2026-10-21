import torch.nn as nn

from ddpm.image.config import AnyScheduleConfig
from ddpm.layers import SinusoidalPositionalEmbedding


def build_positional_embedding(embd_dim: int, noise_schedule: AnyScheduleConfig) -> nn.Module:
    """The timestep embedding for the schedule family: sinusoidal for discrete schedules (long t in
    [0, T))."""
    return SinusoidalPositionalEmbedding(embd_dim)
