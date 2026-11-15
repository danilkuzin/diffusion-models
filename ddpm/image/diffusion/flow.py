import numpy
import torch

from ddpm.image.config import (
    Config,
    RectifiedFlowSamplerConfig,
    RectifiedFlowScheduleConfig,
)
from ddpm.image.diffusion.base import BaseSampler, BaseSchedule, predict_noise
from ddpm.image.models.base import BaseModel


def flow_clipping(x: torch.Tensor, t: float, v: torch.Tensor) -> torch.Tensor:
    """Static thresholding for rectified flow, as ``sample_clipping`` does it: the predicted
    endpoint pred_x1 = x + (1 - t) v, the data at t = 1, is clamped to [-1, 1] and v backed out
    of it. Skipped for 1 - t < 1e-4, where x already is the sample and 1 / (1 - t) blows up.
    x, v: (B, C, H, W) -> v (B, C, H, W)."""
    remaining = 1.0 - t
    if remaining < 1e-4:
        return v
    pred_x1 = torch.clamp(x + remaining * v, -1.0, 1.0)

    return (pred_x1 - x) / remaining


class RectifiedFlowSchedule(BaseSchedule):
    """Rectified flow's straight path (Liu et al. 2023, Eq. 1), which is Lipman et al. 2023's
    optimal-transport path (Eq. 20) at sigma_min = 0. Time runs the other way to the diffusion
    schedules: t = 0 is noise and t = 1 is data::

        x_t = (1 - t) eps + t x_0,    v = d x_t / dt = x_0 - eps

    x_0 is the clean image, as everywhere in the repo, so here it is where the path ends."""

    def __init__(self, config: RectifiedFlowScheduleConfig) -> None:
        super().__init__()
        self.config = config

    def sample_random_timestamps(self, batch_size: int, device: str) -> torch.Tensor:
        """-> (B, 1) float32, uniform on [0, 1): neither end is singular, so no offset."""
        return torch.rand(size=(batch_size, 1), device=device)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """x_t = (1 - t) eps + t x_0. x_0, eps: (B, C, H, W); t: (B, 1) -> x_t (B, C, H, W)."""
        t = t.view(-1, 1, 1, 1)
        return (1.0 - t) * eps + t * x_0

    def training_target(
        self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """The velocity x_0 - eps, the same at every t (Liu et al. Eq. 1; Lipman et al. Eq. 23
        at sigma_min = 0)."""
        return x_0 - eps


class RectifiedFlowSampler(BaseSampler):
    """Euler on the learned velocity field (Liu et al. 2023), forwards from noise at t = 0 to
    data at t = 1. The model's output, which ``predict_noise`` returns, is a velocity here; CFG
    blends it as it blends eps."""

    def __init__(
        self, noise_schedule: RectifiedFlowSchedule, config: RectifiedFlowSamplerConfig
    ) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: RectifiedFlowSchedule = noise_schedule
        self.config: RectifiedFlowSamplerConfig = config

    def sample(
        self,
        model: BaseModel,
        x: torch.Tensor,
        lbl_batch: torch.Tensor | None,
        config: Config,
        num_steps: int,
    ) -> torch.Tensor:
        timestamps = self.generate_timestamps(num_steps)
        batch_size = x.shape[0]
        device = x.device

        for t, s in zip(timestamps[:-1], timestamps[1:], strict=False):
            t_batch = torch.full(
                size=(batch_size,), fill_value=t, device=device, dtype=torch.float32
            )
            v = predict_noise(x, lbl_batch, t_batch, model, config)
            x = self.reverse(x, t, v, s)

        return x

    def reverse(self, x: torch.Tensor, t: float, v: torch.Tensor, s: float) -> torch.Tensor:
        """x_t -> x_s = x_t + (s - t) v for s > t, with v thresholded. x, v: (B, C, H, W)."""
        return x + flow_clipping(x, t, v) * (s - t)

    def generate_timestamps(self, num_steps: int) -> list[float]:
        """num_steps + 1 points from 0 up to 1, so num_steps model calls."""
        return list(numpy.linspace(0.0, 1.0, num_steps + 1).tolist())
