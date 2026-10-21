from pathlib import Path

import torch

from ddpm.image.config import CosineScheduleConfig, LinearScheduleConfig
from ddpm.image.diffusion.discrete import (
    CosineDiscreteSchedule,
    DiscreteSchedule,
    LinearDiscreteSchedule,
)

CIFAR10_SAMPLE = Path(__file__).parents[1] / "assets" / "cifar10_sample.pt"

SCHEDULES: dict[str, DiscreteSchedule] = {
    "linear, β 1e-4 → 0.02": LinearDiscreteSchedule(LinearScheduleConfig()),
    "cosine": CosineDiscreteSchedule(CosineScheduleConfig()),
}


def test_noise_schedules_run_from_data_to_noise(viz: Path | None) -> None:
    """DDPM's linear and Nichol & Dhariwal's cosine schedule at T = 1000: every beta_t is a
    variance in (0, 1), so alpha_hat_t falls strictly, from ~1 (x_1 ~ x_0) to ~0, where Eq. 4
    keeps under 1% of x_0 in x_T and q(x_T) ~ N(0, I)."""
    for schedule in SCHEDULES.values():
        assert torch.all((schedule.beta > 0) & (schedule.beta < 1))
        assert torch.all(schedule.alpha_hat[1:] < schedule.alpha_hat[:-1])
        assert schedule.alpha_hat[0] > 0.99
        assert torch.sqrt(schedule.alpha_hat[-1]) < 0.01

    if viz:
        from visualisation.image.diffusion import plot_schedules

        plot_schedules(viz, SCHEDULES)


def test_forward_closed_form_is_the_marginal_of_the_chain(viz: Path | None) -> None:
    """DDPM Eq. 4 is the marginal of Eq. 2's chain: stepping q(x_t | x_{t-1}) from one x_0 gives
    the same per-pixel mean and std as sampling q(x_t | x_0) directly. The betas are large, so an
    off-by-one between chain step i + 1 and alpha_hat[i] fails."""
    torch.manual_seed(0)
    config = LinearScheduleConfig(T=10, beta_start=0.1, beta_end=0.3)
    schedule = LinearDiscreteSchedule(config)
    n = 16384
    x_0 = torch.linspace(-1, 1, 16).reshape(1, 1, 4, 4).repeat(n, 1, 1, 1)

    chain = schedule.forward_per_step(x_0, schedule.beta, config.T)  # [x_0, x_1, ..., x_T]

    for i in range(config.T):
        direct = schedule.forward(x_0, torch.full((n, 1), i), torch.randn_like(x_0))
        torch.testing.assert_close(chain[i + 1].mean(0), direct.mean(0), atol=0.05, rtol=0)
        torch.testing.assert_close(chain[i + 1].std(0), direct.std(0), atol=0.05, rtol=0)

    if viz:
        from visualisation.image.diffusion import plot_forward_strip

        images = torch.load(CIFAR10_SAMPLE)  # (N, 3, 32, 32) uint8
        plot_forward_strip(viz, images[0], SCHEDULES, [0, 50, 100, 250, 500, 750, 1000])
