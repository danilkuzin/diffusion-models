from typing import assert_never

from ddpm.image.config import (
    AnySamplerConfig,
    AnyScheduleConfig,
    CosineScheduleConfig,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    LinearScheduleConfig,
)
from ddpm.image.diffusion.base import BaseSampler, BaseSchedule
from ddpm.image.diffusion.discrete import (
    CosineDiscreteSchedule,
    DDIMSampler,
    DDPMSampler,
    DiscreteSchedule,
    LinearDiscreteSchedule,
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
        case _:
            assert_never(config)


def create_schedule(config: AnyScheduleConfig) -> BaseSchedule:
    match config:
        case LinearScheduleConfig():
            return LinearDiscreteSchedule(config)
        case CosineScheduleConfig():
            return CosineDiscreteSchedule(config)
        case _:
            assert_never(config)
