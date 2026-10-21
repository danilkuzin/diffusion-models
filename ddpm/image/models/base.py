import torch

from ddpm.image.config import Config


class BaseModel(torch.nn.Module):
    """Every denoiser: ``forward(x, t, lbl) -> eps``, the same shape as ``x``."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
