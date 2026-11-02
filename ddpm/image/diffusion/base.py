from abc import ABC, abstractmethod
from typing import Any

import torch

from ddpm.image.config import AnySamplerConfig, Config
from ddpm.image.models.base import BaseModel


class BaseSchedule(ABC):
    """The forward process: ``forward(x_0, t, eps) -> x_t``, with ``t`` as
    ``sample_random_timestamps`` returns it, (B, 1)."""

    @abstractmethod
    def sample_random_timestamps(self, batch_size: int, device: str) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        pass

    def training_target(
        self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """What the network regresses onto: eps for every diffusion schedule (DDPM Eq. 14).
        ``RectifiedFlowSchedule`` overrides it with a velocity."""
        return eps


class BaseSampler(ABC):
    def __init__(self, noise_schedule: BaseSchedule, config: AnySamplerConfig) -> None:
        super().__init__()
        self.noise_schedule = noise_schedule
        self.config = config

    @abstractmethod
    def sample(
        self,
        model: BaseModel,
        x: torch.Tensor,
        lbl_batch: torch.Tensor | None,
        config: Config,
        num_steps: int,
    ) -> torch.Tensor:
        """The reverse process: x, N(0, I) noise (B, C, H, W) -> samples (B, C, H, W). Each
        sampler maps x onto its own prior (only VE rescales it), builds its own time grid from
        ``num_steps`` (``generate_timestamps``) and calls the model exactly ``num_steps`` times,
        plus the corrector's calls for PC."""

    @abstractmethod
    def generate_timestamps(self, num_steps: int) -> list[Any]:
        """The time grid ``sample`` walks, from noise towards data. Raises if the sampler can't
        make exactly ``num_steps`` model calls."""


def predict_noise(
    x: torch.Tensor,
    lbl_batch: torch.Tensor | None,
    t_batch: torch.Tensor,
    model: BaseModel,
    config: Config,
) -> torch.Tensor:
    """eps_theta(x, t), guided when the config has guidance and labels are given: Ho & Salimans
    2022, Eq. 6, with omega = 1 + w. x: (B, C, H, W); t_batch: (B,); lbl_batch: (B,) long or None
    -> eps (B, C, H, W), or the velocity for rectified flow. Every sampler calls the model only
    through this."""
    eps: torch.Tensor
    if config.guidance is not None and lbl_batch is not None:
        assert config.omega_classifier_free is not None
        batch_size = x.shape[0]
        null_lbl = torch.full(
            (batch_size,), config.data.num_classes, device=x.device, dtype=torch.long
        )
        eps_uncond: torch.Tensor = model(x, t_batch, null_lbl)
        eps_cond: torch.Tensor = model(x, t_batch, lbl_batch)
        eps = eps_uncond + config.omega_classifier_free * (eps_cond - eps_uncond)
    else:
        eps = model(x, t_batch)

    return eps
