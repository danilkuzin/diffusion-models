from itertools import pairwise
from pathlib import Path

import pytest
import torch

from ddpm.image.config import Config, RectifiedFlowSamplerConfig, RectifiedFlowScheduleConfig
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.models.base import BaseModel

DATA_STD = 0.25  # x_0 ~ N(0, DATA_STD^2 I): 4 std inside [-1, 1], so clipping ~never fires


def velocity_coefficient(t: float) -> float:
    """k(t) in the exact velocity k(t) x for Gaussian data (`GaussianDataVelocity`)."""
    return (t * DATA_STD**2 - (1 - t)) / marginal_variance(t)


def marginal_variance(t: float) -> float:
    """Variance of x_t = (1 - t) eps + t x_0 for x_0 ~ N(0, s_d^2 I)."""
    return (1 - t) ** 2 + t**2 * DATA_STD**2


class GaussianDataVelocity(BaseModel):
    """The exact velocity for x_0 ~ N(0, s_d^2 I): x_t and x_0 - eps are jointly Gaussian, so
    E[x_0 - eps | x_t] = (t s_d^2 - (1 - t)) x_t / Var(x_t). Records (t, std of x) per call."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[tuple[float, float]] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append((t[0].item(), x.std().item()))
        return velocity_coefficient(t[0].item()) * x  # one t for the whole batch


def euler_trajectory(num_steps: int) -> tuple[list[tuple[float, float]], torch.Tensor]:
    """(t, std of x) at each call of `GaussianDataVelocity`, then (1, std of the samples); and
    the samples."""
    config = Config(
        noise_schedule=RectifiedFlowScheduleConfig(),
        sampler=RectifiedFlowSamplerConfig(),
        device="cpu",
    )
    sampler = create_sampler(config.sampler, create_schedule(config.noise_schedule))
    model = GaussianDataVelocity(config)
    out = sampler.sample(model, torch.randn(8192, 1, 4, 4), None, config, num_steps)
    return [*model.calls, (1.0, out.std().item())], out


def test_flow_euler_with_the_exact_velocity_tracks_the_marginals(viz: Path | None) -> None:
    """Gaussian data has an exact velocity k(t) x (`GaussianDataVelocity`), so any error is
    Euler's. The exact flow keeps x's std at sqrt(Var(x_t)); an Euler step scales it by
    1 + dt k(t) instead. One step lands on the data mean: k(0) = -1, so x_1 = 0, even with the
    exact field. The shortfall shrinks as 1/N."""
    torch.manual_seed(0)
    final = {}
    trajectories = {}
    for n in (100, 10, 2, 1):
        trajectory, out = euler_trajectory(n)
        expected = [trajectory[0][1]]
        for (t, _), (s, _) in pairwise(trajectory):
            expected.append(expected[-1] * abs(1 + (s - t) * velocity_coefficient(t)))
        assert [std for _, std in trajectory] == pytest.approx(expected, rel=1e-3, abs=1e-6)
        final[n] = trajectory[-1][1] / DATA_STD
        trajectories[n] = [(t, std / marginal_variance(t) ** 0.5) for t, std in trajectory]
        if n == 1:
            assert torch.equal(out, torch.zeros_like(out))

    assert final[2] < final[10] < final[100] < 1
    assert final[100] == pytest.approx(1.0, abs=0.03)
    assert (1 - final[10]) * 10 == pytest.approx((1 - final[100]) * 100, rel=0.1)

    if viz:
        from visualisation.image.continuous import plot_std_ratio_trajectories

        plot_std_ratio_trajectories(
            viz,
            {f"Euler, {n} steps": trajectories[n] for n in (100, 10, 2)},
            "Rectified flow, exact velocity: std of x / std of x_t",
            sampling_descends=False,
        )
