import math
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from ddpm.image.config import (
    AnySamplerConfig,
    AnyScheduleConfig,
    Config,
    CosineScheduleConfig,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    LinearScheduleConfig,
)
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.diffusion.base import BaseSampler
from ddpm.image.diffusion.discrete import (
    CosineDiscreteSchedule,
    DDIMSampler,
    LinearDiscreteSchedule,
    sample_clipping,
)
from ddpm.image.models.base import BaseModel

SHAPE = (8, 3, 4, 4)

# Every sampler with the schedule `create_sampler` pairs it with.
SAMPLER_CASES: dict[str, tuple[AnyScheduleConfig, AnySamplerConfig]] = {
    "ddpm": (CosineScheduleConfig(T=5), DDPMSamplerConfig()),  # runs only num_steps = T
    "ddim": (CosineScheduleConfig(T=10), DDIMSamplerConfig(eta=0.0)),
}


def make_sampler(name: str) -> tuple[Config, BaseSampler]:
    schedule_config, sampler_config = SAMPLER_CASES[name]
    config = Config(noise_schedule=schedule_config, sampler=sampler_config, device="cpu")
    return config, create_sampler(config.sampler, create_schedule(config.noise_schedule))


class RecordingModel(BaseModel):
    """A denoiser that records each timestep batch and returns 0.1 * x, drawing no random numbers,
    so a sampler's output depends only on its own math and noise."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[torch.Tensor] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append(t.clone())
        return 0.1 * x


# The timestep the model sees at each call for `num_steps=5`, one call per step. DDIM's grid ends
# at -1, the clean image, which the model never sees. Changing any of them changes sampling, so
# it has to be a deliberate edit here.
EXPECTED_CALLS: dict[str, list[int]] = {
    "ddpm": [4, 3, 2, 1, 0],
    "ddim": [8, 6, 4, 2, 0],
}


@pytest.mark.parametrize("name", SAMPLER_CASES)
def test_sample_calls_the_model_at_the_pinned_timesteps(name: str) -> None:
    """Pins what the model sees, call by call: the sampling behaviour itself, independent of how
    each sampler builds its time grid internally."""
    config, sampler = make_sampler(name)
    model = RecordingModel(config)
    x = torch.randn(SHAPE)

    out = sampler.sample(model, x, None, config, num_steps=5)

    assert out.shape == SHAPE
    assert all(t.dtype == torch.long for t in model.calls)
    # One t for the whole batch per call: (B,) long.
    assert [t.tolist() for t in model.calls] == [[s] * SHAPE[0] for s in EXPECTED_CALLS[name]]


DATA_STD = 0.25  # x_0 ~ N(0, DATA_STD^2 I): 4 std inside [-1, 1], so clipping ~never fires
COSINE = CosineDiscreteSchedule(CosineScheduleConfig())  # T = 1000


def alpha_hat(t: int) -> float:
    """alpha_hat at index t; t = -1 is the clean image, with alpha_hat = 1."""
    return 1.0 if t < 0 else float(COSINE.alpha_hat[t])


def marginal_std(t: int) -> float:
    """std of q(x_t) (Eq. 4) for Gaussian data: |a_t|, a_t = (sqrt(ah_t) s, sqrt(1 - ah_t))."""
    return math.sqrt(alpha_hat(t) * DATA_STD**2 + 1 - alpha_hat(t))


def angle(t: int) -> float:
    """a_t's angle from the data axis: 0 at t = -1, -> pi / 2 as t -> T."""
    return math.atan2(math.sqrt(1 - alpha_hat(t)), math.sqrt(alpha_hat(t)) * DATA_STD)


class GaussianDataDenoiser(BaseModel):
    """The exact eps_theta for x_0 ~ N(0, s^2 I): x_t and eps are jointly Gaussian, so
    E[eps | x_t] = sqrt(1 - ah_t) x_t / marginal_std(t)^2. Records (t, std of x) per call."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[tuple[int, float]] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append((int(t[0]), x.std().item()))
        ah = COSINE.alpha_hat[t].view(-1, 1, 1, 1)
        return torch.sqrt(1 - ah) * x / (ah * DATA_STD**2 + 1 - ah)


def std_trajectory(sampler_config: AnySamplerConfig, num_steps: int) -> list[tuple[int, float]]:
    """(t, std of x) at each call of `GaussianDataDenoiser` on the cosine schedule, then
    (-1, std of the samples)."""
    config = Config(noise_schedule=CosineScheduleConfig(), sampler=sampler_config, device="cpu")
    model = GaussianDataDenoiser(config)
    sampler = create_sampler(config.sampler, create_schedule(config.noise_schedule))
    out = sampler.sample(model, torch.randn(8192, 1, 4, 4), None, config, num_steps)
    return [*model.calls, (-1, out.std().item())]


def test_samplers_with_the_exact_denoiser_track_the_forward_marginals(viz: Path | None) -> None:
    """Gaussian data has an exact eps (`GaussianDataDenoiser`), so any error is the sampler's.
    DDPM (Alg. 2) passes through q(x_t)'s std at every step. A DDIM step (eta = 0) from t to s
    is then linear, x_s = (a_s . a_t / |a_t|^2) x_t, so it scales the std by
    cos(angle(t) - angle(s)) (Cauchy-Schwarz): few steps undershoot the data's spread."""
    torch.manual_seed(0)
    ddpm = std_trajectory(DDPMSamplerConfig(), 1000)
    for t, std in ddpm:
        assert std == pytest.approx(marginal_std(t), abs=0.01)

    ddim = {n: std_trajectory(DDIMSamplerConfig(eta=0.0), n) for n in (10, 50)}
    for trajectory in ddim.values():
        t_0, std_0 = trajectory[0]
        shrink = std_0 / marginal_std(t_0)  # N(0, I) enters at the grid's first t, T - T / n
        for (t, std), (s, _) in pairwise(trajectory):
            assert std == pytest.approx(shrink * marginal_std(t), abs=1e-3)
            shrink *= math.cos(angle(t) - angle(s))
        assert trajectory[-1][1] == pytest.approx(shrink * DATA_STD, abs=1e-3)
    assert ddim[10][-1][1] < ddim[50][-1][1] < DATA_STD

    if viz:
        from visualisation.image.diffusion import plot_std_trajectories

        plot_std_trajectories(
            viz,
            {"DDPM, 1000 steps": ddpm, "DDIM, 50 steps": ddim[50], "DDIM, 10 steps": ddim[10]},
            [(t, marginal_std(t)) for t in range(999, -2, -1)],
        )


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_ddim_last_step_returns_the_clipped_pred_x0(eta: float) -> None:
    """The step to s = -1 has ah_s = 1: Eq. 16 gives sigma = 0 for any eta, the direction term
    vanishes, and Eq. 12 leaves exactly the clipped pred_x0, with no noise in it."""
    schedule = LinearDiscreteSchedule(LinearScheduleConfig(T=10, beta_start=0.01, beta_end=0.1))
    sampler = DDIMSampler(schedule, DDIMSamplerConfig(eta=eta))
    x, eps = torch.randn(SHAPE), torch.randn(SHAPE)

    pred_x0, _ = sample_clipping(x, schedule.alpha_hat[0], eps)

    assert torch.equal(sampler.reverse(x, t=0, eps=eps, s=-1), pred_x0)


def test_sample_clipping_clamps_pred_x0_and_keeps_x_t(viz: Path | None) -> None:
    """Static thresholding: an out-of-range pred_x0 clamps to +-1, an in-range one is untouched,
    and the returned (pred_x0, eps) pair still gives back the same x_t."""
    torch.manual_seed(0)
    schedule = LinearDiscreteSchedule(LinearScheduleConfig(T=10, beta_start=0.01, beta_end=0.1))
    t, ah_t = 5, schedule.alpha_hat[5]
    x_0 = torch.rand(SHAPE) * 1.6 - 0.8  # in [-0.8, 0.8], strictly inside
    eps = torch.randn(SHAPE)
    x = schedule.forward(x_0, torch.full((SHAPE[0], 1), t), eps)

    # A wrong eps in the first half of the batch: eps - shift moves pred_x0 by
    # sqrt((1 - ah) / ah) * shift = +2, so it lands in [1.2, 2.8] and must clamp to 1.
    half = SHAPE[0] // 2
    shift = torch.zeros(SHAPE)
    shift[:half] = 2.0 * torch.sqrt(ah_t / (1.0 - ah_t))
    pred_x0, safe_eps = sample_clipping(x, ah_t, eps - shift)

    assert torch.equal(pred_x0[:half], torch.ones_like(pred_x0[:half]))
    # In range, clipping changes nothing.
    assert torch.allclose(pred_x0[half:], x_0[half:], atol=1e-5)
    assert torch.allclose(safe_eps[half:], eps[half:], atol=1e-5)
    # The returned pair still describes the same x_t: x = sqrt(ah) pred_x0 + sqrt(1 - ah) eps.
    assert torch.allclose(
        torch.sqrt(ah_t) * pred_x0 + torch.sqrt(1.0 - ah_t) * safe_eps, x, atol=1e-5
    )

    if viz:
        from visualisation.image.diffusion import plot_clipping

        plot_clipping(viz, x, ah_t, eps - shift)
