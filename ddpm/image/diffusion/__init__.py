from typing import assert_never

from ddpm.image.config import (
    AnySamplerConfig,
    AnyScheduleConfig,
    CosineScheduleConfig,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    EulerMaruyamaVESamplerConfig,
    EulerMaruyamaVPSamplerConfig,
    LinearScheduleConfig,
    PFODESamplerConfig,
    PredictorCorrectorSamplerConfig,
    RectifiedFlowSamplerConfig,
    RectifiedFlowScheduleConfig,
    SubVariancePreservingScheduleConfig,
    VarianceExplodingScheduleConfig,
    VariancePreservingScheduleConfig,
)
from ddpm.image.diffusion.base import BaseSampler, BaseSchedule
from ddpm.image.diffusion.discrete import (
    CosineDiscreteSchedule,
    DDIMSampler,
    DDPMSampler,
    DiscreteSchedule,
    LinearDiscreteSchedule,
)
from ddpm.image.diffusion.flow import RectifiedFlowSampler, RectifiedFlowSchedule
from ddpm.image.diffusion.sde import (
    EulerMaruyamaVESampler,
    EulerMaruyamaVPSampler,
    PFODESampler,
    PredictorCorrectorSampler,
    SubVariancePreservingSchedule,
    VarianceExplodingSchedule,
    VariancePreservingSchedule,
    VPFamilySchedule,
)


def create_sampler(config: AnySamplerConfig, schedule: BaseSchedule) -> BaseSampler:
    """The sampler for ``config``; asserts the schedule suits it (DDPM/DDIM need a discrete one)."""
    match config:
        case DDPMSamplerConfig():
            assert isinstance(schedule, DiscreteSchedule)
            return DDPMSampler(schedule, config)
        case DDIMSamplerConfig():
            assert isinstance(schedule, DiscreteSchedule)
            return DDIMSampler(schedule, config)
        case EulerMaruyamaVPSamplerConfig():
            assert isinstance(schedule, VPFamilySchedule)
            return EulerMaruyamaVPSampler(schedule, config)
        case EulerMaruyamaVESamplerConfig():
            assert isinstance(schedule, VarianceExplodingSchedule)
            return EulerMaruyamaVESampler(schedule, config)
        case PredictorCorrectorSamplerConfig():
            assert isinstance(schedule, VPFamilySchedule)
            return PredictorCorrectorSampler(schedule, config)
        case PFODESamplerConfig():
            assert isinstance(schedule, VPFamilySchedule)
            return PFODESampler(schedule, config)
        case RectifiedFlowSamplerConfig():
            assert isinstance(schedule, RectifiedFlowSchedule)
            return RectifiedFlowSampler(schedule, config)
        case _:
            assert_never(config)


def create_schedule(config: AnyScheduleConfig) -> BaseSchedule:
    match config:
        case LinearScheduleConfig():
            return LinearDiscreteSchedule(config)
        case CosineScheduleConfig():
            return CosineDiscreteSchedule(config)
        case VariancePreservingScheduleConfig():
            return VariancePreservingSchedule(config)
        case VarianceExplodingScheduleConfig():
            return VarianceExplodingSchedule(config)
        case SubVariancePreservingScheduleConfig():
            return SubVariancePreservingSchedule(config)
        case RectifiedFlowScheduleConfig():
            return RectifiedFlowSchedule(config)
        case _:
            assert_never(config)
