import math
from abc import abstractmethod

import numpy
import torch

from ddpm.image.config import (
    Config,
    EulerMaruyamaVESamplerConfig,
    EulerMaruyamaVPSamplerConfig,
    PFODESamplerConfig,
    PredictorCorrectorSamplerConfig,
    SubVariancePreservingScheduleConfig,
    VarianceExplodingScheduleConfig,
    VariancePreservingScheduleConfig,
)
from ddpm.image.diffusion.base import BaseSampler, BaseSchedule, predict_noise
from ddpm.image.models.base import BaseModel

T_END = 1e-4  # the reverse grid's last time; the step onto it adds no noise


class LinearContinuousSchedule(BaseSchedule):
    """A forward SDE dx = f(x, t) dt + g(t) dw on t in (0, 1], data at t = 0 (Song et al. 2021,
    "Score-Based Generative Modeling through Stochastic Differential Equations", Eq. 5; "score
    SDE" below). f is linear in x, so x_t | x_0 is Gaussian (App. B, Eq. 29)."""

    @abstractmethod
    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of x_t | x_0 (score SDE Eq. 29). x_0: (B, C, H, W); t: (B,) or (B, 1)
        float32 -> mean (B, C, H, W), std (B, 1, 1, 1)."""
        pass

    @abstractmethod
    def diffusion_sq(self, t: float) -> float:
        """g(t)^2 (score SDE Eq. 5) at a float t."""
        pass

    def sample_random_timestamps(self, batch_size: int, device: str) -> torch.Tensor:
        """-> (B, 1) float32, uniform on [1e-5, 1). The offset is the reference code's eps: at
        t = 0 the std is 0 (VP) or sigma_min (VE)."""
        return torch.rand(size=(batch_size, 1), device=device) * (1.0 - 1e-5) + 1e-5

    def forward(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """x_t = mean + std * eps. x_0, eps: (B, C, H, W); t: (B, 1) -> x_t (B, C, H, W)."""
        mean, std = self.marginal_prob(x_0, t)
        return mean + std.view(-1, 1, 1, 1) * eps


class VPFamilySchedule(LinearContinuousSchedule):
    """VP and sub-VP (score SDE Eqs. 11, 12) share the linear beta(t), the drift
    f = -beta(t) x / 2 and the mean coefficient exp(-B(t) / 2), B(t) = int_0^t beta. They differ
    in ``std`` and ``diffusion_sq``, which the VP-family samplers read from the schedule, so
    each sampler runs on either. Siblings, since neither is a special case of the other."""

    def __init__(
        self, config: VariancePreservingScheduleConfig | SubVariancePreservingScheduleConfig
    ) -> None:
        super().__init__()
        self.config = config

    def beta(self, t: float) -> float:
        """beta(t), linear from beta_min at t = 0 to beta_max at t = 1 (score SDE §3.4)."""
        return self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """-B(t) / 2, the log of x_0's coefficient in the mean. Any shape of t."""
        return -0.5 * (
            self.config.beta_min * t + 0.5 * (self.config.beta_max - self.config.beta_min) * (t**2)
        )

    @abstractmethod
    def std(self, log_mean_coeff: torch.Tensor) -> torch.Tensor:
        """The marginal std from log_mean_coeff, (B, 1, 1, 1)."""

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = self.log_mean_coeff(t).view(-1, 1, 1, 1)

        mean = torch.exp(log_mean_coeff) * x_0
        std = self.std(log_mean_coeff)

        return mean, std


class VariancePreservingSchedule(VPFamilySchedule):
    """VP SDE (score SDE Eq. 11): dx = -beta(t) x dt / 2 + sqrt(beta(t)) dw, the continuous
    limit of DDPM's chain. beta(t) / N is DDPM's beta_i: 1e-4 to 0.02 for 0.1 to 20 at N = 1000."""

    def __init__(self, config: VariancePreservingScheduleConfig) -> None:
        super().__init__(config)
        self.config: VariancePreservingScheduleConfig = config

    def diffusion_sq(self, t: float) -> float:
        """g^2 = beta(t) (Eq. 11)."""
        return self.beta(t)

    def std(self, log_mean_coeff: torch.Tensor) -> torch.Tensor:
        # sqrt(1 - e^{-B}): mean coefficient^2 + std^2 = 1, hence "variance preserving"
        return torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))


class VarianceExplodingSchedule(LinearContinuousSchedule):
    """VE SDE (score SDE Eq. 9): dx = sqrt(d sigma(t)^2 / dt) dw, with sigma(t) geometric from
    sigma_min to sigma_max. No drift: the mean stays x_0 and the prior is N(0, sigma_max^2 I)."""

    def __init__(self, config: VarianceExplodingScheduleConfig):
        super().__init__()
        self.config = config

    def diffusion_sq(self, t: float) -> float:
        """g^2 = d sigma(t)^2 / dt = 2 sigma(t)^2 ln(sigma_max / sigma_min) (Eq. 9)."""
        ratio = self.config.sigma_max / self.config.sigma_min
        sigma = self.config.sigma_min * math.pow(ratio, t)
        return 2.0 * sigma**2 * math.log(ratio)

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # std = sigma(t), as in the reference code; Eq. 29's variance is sigma(t)^2 - sigma_min^2
        mean = x_0
        std = self.config.sigma_min * (self.config.sigma_max / self.config.sigma_min) ** t
        std = std.view(-1, 1, 1, 1)

        return mean, std


class SubVariancePreservingSchedule(VPFamilySchedule):
    """Sub-VP SDE (score SDE Eq. 12): VP's drift with less noise, so its variance
    (1 - e^{-B})^2 stays below VP's 1 - e^{-B} at every t > 0."""

    def __init__(self, config: SubVariancePreservingScheduleConfig) -> None:
        super().__init__(config)
        self.config: SubVariancePreservingScheduleConfig = config

    def diffusion_sq(self, t: float) -> float:
        """g^2 = beta(t) (1 - e^{-2B(t)}) (Eq. 12)."""
        log_mean_coeff = self.log_mean_coeff(torch.tensor(t, dtype=torch.float64)).item()  # -B(t)/2
        return self.beta(t) * (1.0 - math.exp(4.0 * log_mean_coeff))  # e^{-2B} = e^{4 * lmc}

    def std(self, log_mean_coeff: torch.Tensor) -> torch.Tensor:
        # The variance is a perfect square, (1 - e^{-B})^2, so the std has no square root.
        return 1.0 - torch.exp(2.0 * log_mean_coeff)


class SDESampler(BaseSampler):
    """Walks a float grid from t = 1 down to T_END, one model call per step."""

    @abstractmethod
    def reverse(self, x: torch.Tensor, t: float, eps: torch.Tensor, s: float) -> torch.Tensor:
        """x_t -> x_s for s < t. x, eps: (B, C, H, W)."""
        pass

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
            eps = predict_noise(x, lbl_batch, t_batch, model, config)
            x = self.reverse(x, t, eps, s)

        return x

    def generate_timestamps(self, num_steps: int) -> list[float]:
        """num_steps + 1 points from 1 down to T_END, uniform in t, so num_steps model calls.
        The reference code stops at 1e-3 for VP and 1e-5 for VE (score SDE App. D.4). linspace
        returns T_END exactly, which the steps' last-step test ``s > T_END`` relies on."""
        return list(numpy.linspace(1.0, T_END, num_steps + 1).tolist())


def vp_score(
    schedule: VPFamilySchedule, x: torch.Tensor, t: float, eps: torch.Tensor
) -> torch.Tensor:
    """The score -eps / std of x_t | x_0's Gaussian, after static thresholding as in
    ``sample_clipping``: pred_x0 = (x - std eps) / mean coefficient, clamped to [-1, 1], and eps
    backed out. The clamp is this repo's; the score SDE samplers don't clip. x, eps:
    (B, C, H, W) -> score (B, C, H, W)."""
    t_tensor = torch.tensor([t], device=x.device).view(-1, 1, 1, 1)
    log_mean_coeff = schedule.log_mean_coeff(t_tensor)
    mean_coeff, std = torch.exp(log_mean_coeff), schedule.std(log_mean_coeff)

    pred_x0 = (x - std * eps) / mean_coeff
    safe_pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    safe_eps = (x - mean_coeff * safe_pred_x0) / std

    return -safe_eps / std


def vp_sde_euler_maruyama_step(
    schedule: VPFamilySchedule, x: torch.Tensor, t: float, eps: torch.Tensor, s: float
) -> torch.Tensor:
    """One Euler-Maruyama step of the reverse SDE (score SDE Eq. 6) from t to s < t, for VP or
    sub-VP. The last step (s = T_END) returns the mean, with no noise, like the reference code's
    ``denoise``. x, eps: (B, C, H, W) -> x_s."""
    score = vp_score(schedule, x, t, eps)

    drift = -0.5 * schedule.beta(t) * x  # f(x, t), the same for VP and sub-VP
    g_sq = schedule.diffusion_sq(t)
    dt = t - s
    f_rev = drift - g_sq * score  # Eq. 6's drift; the step runs backwards in t, so x - f_rev dt
    x_mean = x - f_rev * dt

    if s > T_END:
        z = torch.randn_like(x)
        x_prev = x_mean + math.sqrt(g_sq * dt) * z
    else:
        x_prev = x_mean

    return x_prev


class EulerMaruyamaVPSampler(SDESampler):
    """Euler-Maruyama on the reverse SDE (score SDE Eq. 6) for the VP family, VP or sub-VP."""

    def __init__(
        self, noise_schedule: VPFamilySchedule, config: EulerMaruyamaVPSamplerConfig
    ) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: VPFamilySchedule = noise_schedule
        self.config: EulerMaruyamaVPSamplerConfig = config

    def reverse(self, x: torch.Tensor, t: float, eps: torch.Tensor, s: float) -> torch.Tensor:
        return vp_sde_euler_maruyama_step(self.noise_schedule, x, t, eps, s)


class EulerMaruyamaVESampler(SDESampler):
    """Euler-Maruyama on the reverse VE SDE (score SDE Eq. 6, f = 0), from VE's prior. It uses
    the raw eps, without ``vp_score``'s thresholding, as the reference code does."""

    def __init__(
        self, noise_schedule: VarianceExplodingSchedule, config: EulerMaruyamaVESamplerConfig
    ) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: VarianceExplodingSchedule = noise_schedule
        self.config: EulerMaruyamaVESamplerConfig = config

    def reverse(self, x: torch.Tensor, t: float, eps: torch.Tensor, s: float) -> torch.Tensor:
        device = x.device

        t_tensor = torch.tensor([t], device=device).view(-1, 1, 1, 1)

        _, std = self.noise_schedule.marginal_prob(x, t_tensor)

        score = -eps / std

        g_sq = self.noise_schedule.diffusion_sq(t)

        dt = t - s
        f_rev = -g_sq * score  # f = 0
        x_mean = x - f_rev * dt

        if s > T_END:
            z = torch.randn_like(x)
            x_prev = x_mean + math.sqrt(g_sq * dt) * z
        else:
            x_prev = x_mean

        return x_prev

    def sample(
        self,
        model: BaseModel,
        x: torch.Tensor,
        lbl_batch: torch.Tensor | None,
        config: Config,
        num_steps: int,
    ) -> torch.Tensor:
        """x: N(0, I) noise, scaled to the prior N(0, sigma_max^2 I): x_1 ~ N(x_0, sigma_max^2 I)
        and std(x_0) << sigma_max, as in the reference code's ``prior_sampling``."""
        x = self.noise_schedule.config.sigma_max * x
        return super().sample(model, x, lbl_batch, config, num_steps)


class PredictorCorrectorSampler(SDESampler):
    """Predictor-corrector sampling (score SDE §4.2; App. G, Algorithms 1 and 3), VP family: an
    Euler-Maruyama predictor step from t to s, then ``num_iter`` Langevin corrector steps at s,
    after the last predictor step too. The paper's order; the reference code corrects first."""

    def __init__(
        self, noise_schedule: VPFamilySchedule, config: PredictorCorrectorSamplerConfig
    ) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: VPFamilySchedule = noise_schedule
        self.config: PredictorCorrectorSamplerConfig = config

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
            eps = predict_noise(x, lbl_batch, t_batch, model, config)
            x = self.reverse(x, t, eps, s)
            x = self.correct(x, s, lbl_batch, model, config)

        return x

    def reverse(self, x: torch.Tensor, t: float, eps: torch.Tensor, s: float) -> torch.Tensor:
        return vp_sde_euler_maruyama_step(self.noise_schedule, x, t, eps, s)

    def correct(
        self,
        x: torch.Tensor,
        s: float,
        lbl_batch: torch.Tensor | None,
        model: BaseModel,
        config: Config,
    ) -> torch.Tensor:
        """Langevin steps at fixed s (App. G, Algorithm 5): x <- x + e score + sqrt(2 e) z, with
        e = 2 (r ||z|| / ||score||)^2 for the target signal-to-noise ratio r = alpha_snr.
        Algorithm 5 also multiplies e by alpha_i = 1 - beta_i, ~1 at small steps; omitted here.
        x: (B, C, H, W) -> x."""
        batch_size = x.shape[0]
        s_batch = torch.full((batch_size,), s, device=x.device, dtype=torch.float32)

        for _m in range(self.config.num_iter):
            eps = predict_noise(x, lbl_batch, s_batch, model, config)
            score = vp_score(self.noise_schedule, x, s, eps)

            noise = torch.randn_like(x)
            # Per-sample norms, then their batch mean, as in the reference code
            score_norm = torch.norm(score.reshape(batch_size, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=-1).mean()

            step_size = (self.config.alpha_snr * noise_norm / score_norm) ** 2 * 2.0
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
        return x


class PFODESampler(SDESampler):
    """Euler on the probability flow ODE (score SDE Eq. 13), VP family: Eq. 6's drift with half
    the score term and no noise. Deterministic, with the SDE's marginals."""

    def __init__(self, noise_schedule: VPFamilySchedule, config: PFODESamplerConfig) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: VPFamilySchedule = noise_schedule
        self.config: PFODESamplerConfig = config

    def reverse(self, x: torch.Tensor, t: float, eps: torch.Tensor, s: float) -> torch.Tensor:
        score = vp_score(self.noise_schedule, x, t, eps)
        beta_t = self.noise_schedule.beta(t)

        drift = -0.5 * beta_t * x
        g_sq = self.noise_schedule.diffusion_sq(t)
        dt = t - s
        f_rev = drift - 0.5 * g_sq * score  # Eq. 13: half of Eq. 6's score term
        x_prev = x - f_rev * dt

        return x_prev
