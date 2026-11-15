from abc import abstractmethod

import torch

from ddpm.image.config import (
    Config,
    CosineScheduleConfig,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    LinearScheduleConfig,
)
from ddpm.image.diffusion.base import BaseSampler, BaseSchedule, predict_noise
from ddpm.image.models.base import BaseModel


def sample_clipping(
    x: torch.Tensor, ah_t: torch.Tensor, eps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static thresholding (Saharia et al. 2022, §2.3): pred_x0 from DDIM Eq. 9, clamped to
    [-1, 1], then the eps that is consistent with it and x. DDPM's reference code clips the same
    way: ``p_mean_variance`` takes the posterior mean from the clipped pred_x0, which is Eq. 11
    with this eps. x, eps: (B, C, H, W); ah_t: 0-dim -> (pred_x0, safe_eps), both (B, C, H, W)."""
    pred_x0 = (x - torch.sqrt(1.0 - ah_t) * eps) / torch.sqrt(ah_t)
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    safe_eps = (x - torch.sqrt(ah_t) * pred_x0) / torch.sqrt(1.0 - ah_t)

    return pred_x0, safe_eps


class DiscreteSchedule(BaseSchedule):
    """DDPM's discrete-time forward process (Ho et al. 2020). Timesteps are long in [0, T): index
    i is the paper's t = i + 1, so beta[0] is its beta_1."""

    def __init__(self, config: LinearScheduleConfig | CosineScheduleConfig) -> None:
        super().__init__()
        self.config = config
        self.beta = self.get_beta()  # (T,)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # alpha_hat_t = prod_{s <= t} alpha_s

    @abstractmethod
    def get_beta(self) -> torch.Tensor:
        pass

    def sample_random_timestamps(self, batch_size: int, device: str) -> torch.Tensor:
        """-> (B, 1) long, uniform in [0, T) (DDPM Alg. 1)."""
        return torch.randint(low=0, high=self.config.T, size=(batch_size, 1), device=device)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """DDPM Eq. 4, reparametrised: x_t = sqrt(alpha_hat_t) x_0 + sqrt(1 - alpha_hat_t) eps.
        x_0, eps: (B, C, H, W); t: (B, 1) long -> x_t (B, C, H, W)."""
        a_hat = self.alpha_hat.to(x_0.device)[t]
        a_hat = a_hat.view(-1, 1, 1, 1)
        x_t = torch.sqrt(a_hat) * x_0 + torch.sqrt(1 - a_hat) * eps

        return x_t

    def forward_per_step(self, x: torch.Tensor) -> list[torch.Tensor]:
        """DDPM Eq. 2, one step at a time: x (B, C, H, W) -> [x_0, x_1, ..., x_T]."""

        x_steps = [x]

        for t in range(self.config.T):
            dist = torch.distributions.Normal(
                loc=torch.sqrt(1 - self.beta[t]) * x_steps[-1],
                scale=torch.sqrt(self.beta[t]) * torch.ones_like(x_steps[-1]),
            )
            x_steps.append(dist.sample())  # type: ignore[no-untyped-call]

        return x_steps


class LinearDiscreteSchedule(DiscreteSchedule):
    """DDPM §4: beta_t linear from beta_start to beta_end."""

    def __init__(self, config: LinearScheduleConfig) -> None:
        super().__init__(config)
        self.config: LinearScheduleConfig = config

    def get_beta(self) -> torch.Tensor:
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.T)


class CosineDiscreteSchedule(DiscreteSchedule):
    """Nichol & Dhariwal 2021, Eq. 17: alpha_hat_t = f(t) / f(0) with
    f(t) = cos^2((t / T + s) / (1 + s) * pi / 2), and beta_t = 1 - alpha_hat_t / alpha_hat_{t-1}.
    Deviation: beta is clipped to [1e-4, 1 - 1e-4] (the paper: at most 0.999)."""

    def __init__(self, config: CosineScheduleConfig) -> None:
        super().__init__(config)
        self.config: CosineScheduleConfig = config

    def get_beta(self) -> torch.Tensor:
        t = torch.linspace(0, self.config.T, self.config.T + 1, dtype=torch.float32)  # (T + 1,)

        f_t = (
            torch.cos((t / self.config.T + self.config.s) / (1 + self.config.s) * torch.pi / 2) ** 2
        )
        alpha_hat_full = f_t / f_t[0]
        beta = 1 - alpha_hat_full[1:] / (alpha_hat_full[:-1])
        beta = torch.clip(beta, min=1e-4, max=1.0 - 1e-4)

        return beta


class DDPMSampler(BaseSampler):
    """DDPM ancestral sampling (Ho et al. 2020, Alg. 2) with sigma_t^2 = beta_t, one of §3.2's two
    choices (the other, the posterior variance, gave similar results)."""

    def __init__(self, noise_schedule: DiscreteSchedule, config: DDPMSamplerConfig) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: DiscreteSchedule = noise_schedule
        self.config: DDPMSamplerConfig = config

    def reverse(self, x: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        """One step of Alg. 2: the mean is Eq. 11 with the clipped eps. x, eps: (B, C, H, W)
        -> x_{t-1}, noiseless at t = 0."""
        device = x.device
        ah_t = self.noise_schedule.alpha_hat[t].to(device)
        a_t = self.noise_schedule.alpha[t].to(device)
        b_t = self.noise_schedule.beta[t].to(device)

        _, safe_eps = sample_clipping(x, ah_t, eps)
        mean = 1.0 / torch.sqrt(a_t) * (x - b_t / torch.sqrt(1.0 - ah_t) * safe_eps)
        if t > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(b_t)
            x_prev = mean + sigma_t * z
        else:
            x_prev = mean

        return x_prev

    def sample(
        self,
        model: BaseModel,
        x: torch.Tensor,
        lbl_batch: torch.Tensor | None,
        config: Config,
        num_steps: int,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        for t in self.generate_timestamps(num_steps):
            t_batch = torch.full(size=(batch_size,), fill_value=t, device=device, dtype=torch.long)
            eps = predict_noise(x, lbl_batch, t_batch, model, config)
            x = self.reverse(x, t, eps)

        return x

    def generate_timestamps(self, num_steps: int) -> list[int]:
        """[T - 1, ..., 0]. Alg. 2 has no step size to stretch, so num_steps must be T."""
        T = self.noise_schedule.config.T
        if num_steps != T:
            raise ValueError(f"DDPM runs all T = {T} steps, got num_steps = {num_steps}")
        return list(reversed(range(0, T)))


class DDIMSampler(BaseSampler):
    """DDIM (Song et al. 2021, "Denoising Diffusion Implicit Models"). Its alpha_t is DDPM's
    alpha_hat_t, written ah here."""

    def __init__(self, noise_schedule: DiscreteSchedule, config: DDIMSamplerConfig) -> None:
        super().__init__(noise_schedule, config)
        self.noise_schedule: DiscreteSchedule = noise_schedule
        self.config: DDIMSamplerConfig = config

    def reverse(self, x: torch.Tensor, t: int, eps: torch.Tensor, s: int) -> torch.Tensor:
        """Eq. 12 from timestep t to the earlier s < t; s = -1 is the clean image.
        x, eps: (B, C, H, W) -> x_s."""
        device = x.device
        ah_t = self.noise_schedule.alpha_hat[t].to(device)
        if s >= 0:
            ah_s = self.noise_schedule.alpha_hat[s].to(device)
        else:  # the paper's alpha_0 = 1, so Eq. 16 gives sigma = 0 and x_s is pred_x0
            ah_s = torch.ones((), device=device)

        safe_pred_x0, safe_eps = sample_clipping(x, ah_t, eps)

        # Eq. 16
        sigma_t = (
            self.config.eta * torch.sqrt((1 - ah_s) / (1 - ah_t)) * torch.sqrt(1 - ah_t / ah_s)
        )

        direction = torch.sqrt(1 - ah_s - sigma_t**2) * safe_eps  # "direction pointing to x_t"
        z = torch.randn_like(x) if self.config.eta > 0 else torch.zeros_like(x)
        x_prev = torch.sqrt(ah_s) * safe_pred_x0 + direction + sigma_t * z

        return x_prev

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
            t_batch = torch.full(size=(batch_size,), fill_value=t, device=device, dtype=torch.long)
            eps = predict_noise(x, lbl_batch, t_batch, model, config)
            x = self.reverse(x, t, eps, s)

        return x

    def generate_timestamps(self, num_steps: int) -> list[int]:
        """[T - r, ..., r, 0, -1] with stride r = T // num_steps: DDIM §4.2's sub-sequence tau,
        spaced linearly (App. D.2; the paper uses quadratic spacing for CIFAR-10). ``sample``
        walks consecutive pairs, so it makes num_steps model calls, the last from 0 to the clean
        image at -1: index i is the paper's t = i + 1, and -1 its x_0 (as in the DDIM reference
        code)."""
        T = self.noise_schedule.config.T
        if T % num_steps != 0:
            raise ValueError(f"DDIM needs num_steps to divide T = {T}, got {num_steps}")
        return list(reversed(range(0, T, T // num_steps))) + [-1]
