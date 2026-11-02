import math
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from ddpm.image.config import (
    AnySamplerConfig,
    AnyScheduleConfig,
    Config,
    EulerMaruyamaVESamplerConfig,
    EulerMaruyamaVPSamplerConfig,
    PFODESamplerConfig,
    PredictorCorrectorSamplerConfig,
    SubVariancePreservingScheduleConfig,
    VarianceExplodingScheduleConfig,
    VariancePreservingScheduleConfig,
)
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.diffusion.sde import (
    LinearContinuousSchedule,
    PredictorCorrectorSampler,
    SubVariancePreservingSchedule,
    VarianceExplodingSchedule,
    VariancePreservingSchedule,
    VPFamilySchedule,
)
from ddpm.image.models.base import BaseModel

# The score SDE paper's CIFAR-10 settings
VP_CONFIG = VariancePreservingScheduleConfig(beta_min=0.1, beta_max=20.0)
VE_CONFIG = VarianceExplodingScheduleConfig(sigma_min=0.01, sigma_max=50.0)
VP = VariancePreservingSchedule(VP_CONFIG)
SUB_VP = SubVariancePreservingSchedule(
    SubVariancePreservingScheduleConfig(beta_min=0.1, beta_max=20.0)
)
VE = VarianceExplodingSchedule(VE_CONFIG)
SCHEDULES: dict[str, LinearContinuousSchedule] = {"VP": VP, "sub-VP": SUB_VP, "VE": VE}

DATA_STD = 0.25  # x_0 ~ N(0, DATA_STD^2 I): 4 std inside [-1, 1], so clipping ~never fires


def coefficients(
    schedule: LinearContinuousSchedule, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean coefficient a, std s) of x_t | x_0 at each t, (N,) -> (N,), (N,)."""
    mean, std = schedule.marginal_prob(torch.ones(len(t), 1, 1, 1, dtype=t.dtype), t)
    return mean.flatten(), std.flatten()


def marginal_std(schedule: LinearContinuousSchedule, t: float) -> float:
    """std of q(x_t) for Gaussian data: sqrt(a^2 s_d^2 + s^2)."""
    a, s = coefficients(schedule, torch.tensor([t], dtype=torch.float64))
    return math.sqrt(a.item() ** 2 * DATA_STD**2 + s.item() ** 2)


def drift_coefficient(schedule: LinearContinuousSchedule, t: float) -> float:
    """f(t) in the drift f(t) x: -beta(t) / 2 for VP and sub-VP (Eqs. 11, 12), 0 for VE (Eq. 9)."""
    if isinstance(schedule, VPFamilySchedule):
        return -0.5 * schedule.beta(t)
    return 0.0


def test_continuous_schedules_run_from_data_to_noise(viz: Path | None) -> None:
    """Score SDE Eq. 29. VP: a^2 + s^2 = 1 at every t, and a(1) < 0.01, so q(x_1) ~ N(0, I).
    Sub-VP: VP's a, and s = s_VP^2, below VP's on (0, 1]. VE: a = 1 and s geometric from
    sigma_min to sigma_max, so it never standardises."""
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    a_vp, s_vp = coefficients(VP, t)
    a_sub, s_sub = coefficients(SUB_VP, t)
    a_ve, s_ve = coefficients(VE, t)

    torch.testing.assert_close(a_vp**2 + s_vp**2, torch.ones_like(t))
    assert s_vp[0] == 0 and a_vp[-1] < 0.01
    torch.testing.assert_close(a_sub, a_vp)
    torch.testing.assert_close(s_sub, s_vp**2)
    assert torch.all(s_sub[1:] < s_vp[1:])
    assert torch.equal(a_ve, torch.ones_like(t))
    torch.testing.assert_close(s_ve[[0, -1]], torch.tensor([0.01, 50.0], dtype=torch.float64))
    log_steps = torch.diff(torch.log(s_ve))  # geometric: equal steps in log s
    torch.testing.assert_close(log_steps, torch.full_like(log_steps, math.log(5000) / 100))

    if viz:
        from visualisation.image.continuous import plot_continuous_schedules

        plot_continuous_schedules(
            viz, t, {"VP": (a_vp, s_vp), "sub-VP": (a_sub, s_sub), "VE": (a_ve, s_ve)}
        )


def variance_and_derivative(schedule: LinearContinuousSchedule, t: float) -> tuple[float, float]:
    """(v, dv/dt) for the variance v = s(t)^2 of x_t | x_0, by autograd in float64."""
    t_tensor = torch.tensor([t], dtype=torch.float64, requires_grad=True)
    _, std = schedule.marginal_prob(torch.ones(1, 1, 1, 1, dtype=torch.float64), t_tensor)
    v = std.pow(2).sum()
    (dv_dt,) = torch.autograd.grad(v, t_tensor)
    return v.item(), dv_dt.item()


def test_diffusion_coefficient_is_the_variance_derivative(viz: Path | None) -> None:
    """For a linear SDE dx = f(t) x dt + g(t) dw, the variance v = s^2 of x_t | x_0 obeys
    dv/dt = 2 f v + g^2 (Särkkä & Solin 2019, Eq. 5.51; score SDE App. B). So autograd of
    `marginal_prob`'s std gives back each schedule's `diffusion_sq`: two formulas for one SDE,
    written separately."""
    ts = [0.01 * k for k in range(1, 101)]
    rows = {}
    for name, schedule in SCHEDULES.items():
        from_marginal = []
        for t in ts:
            v, dv_dt = variance_and_derivative(schedule, t)
            from_marginal.append(dv_dt - 2 * drift_coefficient(schedule, t) * v)
        coded = [schedule.diffusion_sq(t) for t in ts]
        assert coded == pytest.approx(from_marginal, rel=1e-9)
        rows[name] = (coded, from_marginal)

    if viz:
        from visualisation.image.continuous import plot_diffusion_coefficients

        plot_diffusion_coefficients(viz, ts, rows)


class GaussianDataDenoiser(BaseModel):
    """The exact eps for x_0 ~ N(0, s_d^2 I) under a continuous schedule: x_t = a x_0 + s eps is
    Gaussian with variance v = a^2 s_d^2 + s^2, so E[eps | x_t] = s x_t / v. Records (t, std of x)
    per call."""

    def __init__(self, config: Config, schedule: LinearContinuousSchedule) -> None:
        super().__init__(config)
        self.schedule = schedule
        self.calls: list[tuple[float, float]] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, lbl: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls.append((t[0].item(), x.std().item()))
        a, s = coefficients(self.schedule, t)
        a, s = a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)
        return s * x / (a**2 * DATA_STD**2 + s**2)


def std_trajectory(
    schedule_config: AnyScheduleConfig, sampler_config: AnySamplerConfig, num_steps: int
) -> tuple[list[tuple[float, float]], LinearContinuousSchedule]:
    """(t, std of x) at each call of `GaussianDataDenoiser`, then (the grid's end, std of the
    samples)."""
    config = Config(noise_schedule=schedule_config, sampler=sampler_config, device="cpu")
    schedule = create_schedule(config.noise_schedule)
    assert isinstance(schedule, LinearContinuousSchedule)
    sampler = create_sampler(config.sampler, schedule)
    model = GaussianDataDenoiser(config, schedule)
    out = sampler.sample(model, torch.randn(4096, 1, 4, 4), None, config, num_steps)
    return [*model.calls, (sampler.generate_timestamps(num_steps)[-1], out.std().item())], schedule


def expected_stds(
    schedule: LinearContinuousSchedule, trajectory: list[tuple[float, float]], ode: bool
) -> list[float]:
    """std of x along ``trajectory``'s grid under the exact score -x / v, from its first std. An
    Euler-Maruyama step (Eq. 6) scales x by 1 - dt (f + g^2 / v) and adds variance g^2 dt, none
    on the last step; the PF-ODE (Eq. 13) halves the score term and adds nothing."""
    variance = trajectory[0][1] ** 2
    stds = [trajectory[0][1]]
    for k, ((t, _), (s, _)) in enumerate(pairwise(trajectory)):
        dt, g_sq, v = t - s, schedule.diffusion_sq(t), marginal_std(schedule, t) ** 2
        factor = 1 - dt * (drift_coefficient(schedule, t) + (0.5 if ode else 1.0) * g_sq / v)
        last = k == len(trajectory) - 2
        variance = factor**2 * variance + (0.0 if ode or last else g_sq * dt)
        stds.append(math.sqrt(variance))
    return stds


def ratios(
    schedule: LinearContinuousSchedule, trajectory: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """(t, std of x / std of q(x_t)) along ``trajectory``."""
    return [(t, std / marginal_std(schedule, t)) for t, std in trajectory]


def test_sde_samplers_with_the_exact_score_track_the_marginals(viz: Path | None) -> None:
    """Gaussian data has an exact eps (`GaussianDataDenoiser`), so any error is the sampler's.
    Each step is then linear in x, plus noise for Euler-Maruyama, so the std follows
    `expected_stds` exactly, up to sampling noise. Both reach the data's std as the steps grow;
    at 10 steps the PF-ODE is far closer than Euler-Maruyama. VE starts from its prior,
    N(0, sigma_max^2 I)."""
    torch.manual_seed(0)
    euler = {n: std_trajectory(VP_CONFIG, EulerMaruyamaVPSamplerConfig(), n) for n in (1000, 10)}
    pfode = {n: std_trajectory(VP_CONFIG, PFODESamplerConfig(), n) for n in (100, 10)}
    ve, _ = std_trajectory(VE_CONFIG, EulerMaruyamaVESamplerConfig(), 1000)

    for trajectory, schedule in euler.values():
        assert [std for _, std in trajectory] == pytest.approx(
            expected_stds(schedule, trajectory, ode=False), rel=0.02
        )
    for trajectory, schedule in pfode.values():
        assert [std for _, std in trajectory] == pytest.approx(
            expected_stds(schedule, trajectory, ode=True), rel=1e-3
        )
    assert ve[0][1] == pytest.approx(VE_CONFIG.sigma_max, rel=0.01)  # the input's std is ~1
    assert [std for _, std in ve] == pytest.approx(expected_stds(VE, ve, ode=False), rel=0.02)

    final = {
        name: ratios(schedule, trajectory)[-1][1]
        for name, (trajectory, schedule) in [
            ("euler_1000", euler[1000]),
            ("euler_10", euler[10]),
            ("pfode_100", pfode[100]),
            ("pfode_10", pfode[10]),
            ("ve_1000", (ve, VE)),
        ]
    }
    assert final["euler_1000"] == pytest.approx(1.0, abs=0.02)
    assert final["ve_1000"] == pytest.approx(1.0, abs=0.02)
    assert final["pfode_100"] == pytest.approx(1.0, abs=0.03)
    assert final["euler_10"] < final["pfode_10"] < final["pfode_100"]

    if viz:
        from visualisation.image.continuous import plot_std_ratio_trajectories

        plot_std_ratio_trajectories(
            viz,
            {
                "Euler-Maruyama, 1000 steps": ratios(VP, euler[1000][0]),
                "PF-ODE, 100 steps": ratios(VP, pfode[100][0]),
                "PF-ODE, 10 steps": ratios(VP, pfode[10][0]),
            },
            "VP SDE, exact score: std of x / std of q(x_t)",
            sampling_descends=True,
        )


def test_langevin_corrector_is_stationary_up_to_ulas_bias(viz: Path | None) -> None:
    """On p_s = N(0, v I) with the exact score -x / v, App. G's step e = 2 (r ||z|| / ||score||)^2,
    with per-sample norms, is 2 r^2 v^2 / V for a current spread V. Then Var(x) follows
    V <- (1 - e / v)^2 V + 2 e, whose fixed point is V = v (1 + r^2): at r = 0.16 the std ends
    1.3% too wide, the bias of unadjusted Langevin. Starts at half and twice the right std both
    get there. A batch-wide norm, or ||x|| for ||score||, would move the fixed point."""
    torch.manual_seed(0)
    r, s, iterations = 0.16, 0.2, 150
    config = Config(
        noise_schedule=VP_CONFIG,
        sampler=PredictorCorrectorSamplerConfig(num_iter=1, alpha_snr=r),
        device="cpu",
    )
    sampler = create_sampler(config.sampler, create_schedule(config.noise_schedule))
    assert isinstance(sampler, PredictorCorrectorSampler)
    model = GaussianDataDenoiser(config, sampler.noise_schedule)
    v = marginal_std(VP, s) ** 2  # ~0.38, far from 1, so ||x|| and ||score|| differ

    runs = {}
    for start in (0.5, 2.0):
        x = start * math.sqrt(v) * torch.randn(1024, 3, 8, 8)
        stds, expected = [x.std().item()], [x.std().item()]
        variance = expected[0] ** 2
        for _ in range(iterations):
            x = sampler.correct(x, s, None, model, config)
            stds.append(x.std().item())
            step_over_v = 2 * r**2 * v / variance
            variance = (1 - step_over_v) ** 2 * variance + 2 * step_over_v * v
            expected.append(math.sqrt(variance))
        assert stds == pytest.approx(expected, rel=0.01)
        assert stds[-1] == pytest.approx(math.sqrt(v * (1 + r**2)), rel=0.005)
        runs[f"starting at {start} sqrt(v)"] = [std / math.sqrt(v) for std in stds]

    if viz:
        from visualisation.image.continuous import plot_corrector_convergence

        plot_corrector_convergence(viz, runs, math.sqrt(1 + r**2))
