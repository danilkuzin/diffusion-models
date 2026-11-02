from typing import Literal

import torch
import yaml
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class LinearScheduleConfig(BaseModel):
    """DDPM's schedule (Ho et al. 2020, §4): beta linear from 1e-4 to 0.02 over T = 1000."""

    type: Literal["linear"] = "linear"
    T: PositiveInt = 1000
    beta_start: PositiveFloat = 1e-4
    beta_end: PositiveFloat = 0.02


class CosineScheduleConfig(BaseModel):
    """Nichol & Dhariwal 2021, Eq. 17; ``s`` is its small offset near t = 0."""

    type: Literal["cosine"] = "cosine"
    T: PositiveInt = 1000
    s: PositiveFloat = 0.008


class VariancePreservingScheduleConfig(BaseModel):
    """VP SDE (Song et al. 2021, score SDE Eq. 11): beta(t) linear from beta_min to beta_max on
    t in [0, 1]. The paper uses 0.1 and 20, DDPM's linear beta at N = 1000."""

    type: Literal["vp_sde"] = "vp_sde"
    beta_min: PositiveFloat
    beta_max: PositiveFloat


class VarianceExplodingScheduleConfig(BaseModel):
    """VE SDE (score SDE Eq. 9): sigma(t) geometric from sigma_min to sigma_max. The paper uses
    0.01 and 50 on CIFAR-10."""

    type: Literal["ve_sde"] = "ve_sde"
    sigma_min: PositiveFloat
    sigma_max: PositiveFloat


class SubVariancePreservingScheduleConfig(BaseModel):
    """Sub-VP SDE (score SDE Eq. 12): the VP SDE's beta(t), with less noise injected."""

    type: Literal["sub_vp_sde"] = "sub_vp_sde"
    beta_min: PositiveFloat
    beta_max: PositiveFloat


class RectifiedFlowScheduleConfig(BaseModel):
    """Rectified flow's straight path from noise to data; it has no parameters."""

    type: Literal["rectified_flow"] = "rectified_flow"


class DataConfig(BaseModel):
    batch_size: PositiveInt = 16
    num_channels: PositiveInt = 3
    height: PositiveInt = 32
    width: PositiveInt = 32
    num_classes: PositiveInt = 10
    random_flip: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_channels, self.height, self.width)


class TrainConfig(BaseModel):
    num_epochs: PositiveInt = 100
    lr: PositiveFloat = 2e-4
    grad_clip: PositiveFloat | None = None  # max gradient norm; None = no clipping
    eval_every_n_epochs: PositiveInt = 10


class DDPMSamplerConfig(BaseModel):
    type: Literal["ddpm"] = "ddpm"


class DDIMSamplerConfig(BaseModel):
    type: Literal["ddim"] = "ddim"
    # DDIM Eq. 16: 0 is deterministic, 1 gives the DDPM posterior variance
    eta: NonNegativeFloat = 0.0


class EulerMaruyamaVPSamplerConfig(BaseModel):
    """Euler-Maruyama on the reverse SDE (score SDE Eq. 6). "vp" is the VP family: VP or
    sub-VP."""

    type: Literal["euler_vp"] = "euler_vp"


class EulerMaruyamaVESamplerConfig(BaseModel):
    type: Literal["euler_ve"] = "euler_ve"


class PredictorCorrectorSamplerConfig(BaseModel):
    """Score SDE App. G, VP family: num_iter Langevin corrector steps at signal-to-noise ratio
    alpha_snr after each predictor step. 0.16 is the paper's VE ratio (App. H); it tuned VP's
    per sampler (Table 5)."""

    type: Literal["pc"] = "pc"
    num_iter: PositiveInt = 1
    alpha_snr: PositiveFloat = 0.16


class PFODESamplerConfig(BaseModel):
    """Euler on the probability flow ODE (score SDE Eq. 13), VP family."""

    type: Literal["pfode"] = "pfode"


class RectifiedFlowSamplerConfig(BaseModel):
    """Euler integration of the learned velocity field, t: 0 -> 1 (noise -> data)."""

    type: Literal["flow_euler"] = "flow_euler"


class EMAConfig(BaseModel):
    decay: PositiveFloat  # shadow <- decay * shadow + (1 - decay) * param
    update_freq: PositiveInt  # in batches


class UNetModelConfig(BaseModel):
    type: Literal["unet"] = "unet"
    num_ch: PositiveInt = 16
    num_groups: PositiveInt = 4
    embd_dim: PositiveInt = 64
    num_heads: PositiveInt = 2
    channel_mults: tuple[PositiveInt, ...] = (1, 2, 2, 2)
    num_res_blocks: PositiveInt = 2
    attention_stage_index: NonNegativeInt = 1  # the down/up stage that gets self-attention


class DiTModelConfig(BaseModel):
    type: Literal["dit"] = "dit"
    patch_size: PositiveInt = 2
    heads: PositiveInt = 2
    embd_dim: PositiveInt = 64
    depth: PositiveInt = 6
    mlp_ratio: PositiveFloat = 4.0

    @model_validator(mode="after")
    def check_divisibility(self) -> "DiTModelConfig":
        # nn.MultiheadAttention splits embd_dim across heads
        if self.embd_dim % self.heads != 0:
            raise ValueError(f"embd_dim {self.embd_dim} must be divisible by heads {self.heads}")
        # 2D sincos splits embd_dim across (row, col), each into sin/cos halves
        if self.embd_dim % 4 != 0:
            raise ValueError(f"embd_dim {self.embd_dim} must be divisible by 4")
        return self


DiscreteScheduleConfig = LinearScheduleConfig | CosineScheduleConfig
AnyScheduleConfig = (
    LinearScheduleConfig
    | CosineScheduleConfig
    | VariancePreservingScheduleConfig
    | VarianceExplodingScheduleConfig
    | SubVariancePreservingScheduleConfig
    | RectifiedFlowScheduleConfig
)

AnySamplerConfig = (
    DDPMSamplerConfig
    | DDIMSamplerConfig
    | EulerMaruyamaVPSamplerConfig
    | EulerMaruyamaVESamplerConfig
    | PredictorCorrectorSamplerConfig
    | PFODESamplerConfig
    | RectifiedFlowSamplerConfig
)

AnyModelConfig = UNetModelConfig | DiTModelConfig


class Config(BaseModel):
    # Names the run's checkpoint, log and plot directories (`ddpm.paths.Paths`). Not a path.
    run_name: str = "ddpm"
    noise_schedule: AnyScheduleConfig = Field(
        default_factory=LinearScheduleConfig, discriminator="type"
    )
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    sampler: AnySamplerConfig = Field(default_factory=DDPMSamplerConfig, discriminator="type")
    model: AnyModelConfig = Field(default_factory=UNetModelConfig, discriminator="type")
    ema: EMAConfig | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    guidance: Literal["classifier-free"] | None = None
    # CFG weight: Ho & Salimans 2022, Eq. 6 with omega = 1 + w. 0 gives eps_uncond, 1 eps_cond.
    omega_classifier_free: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def check_guidance(self) -> "Config":
        if (self.guidance is None) != (self.omega_classifier_free is None):
            raise ValueError("omega_classifier_free and guidance should be set together")
        return self


def load_config(filepath: str) -> Config:
    """YAML file -> validated Config."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    config = Config.model_validate(data)

    return config
