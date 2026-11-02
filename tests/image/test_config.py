import typing
from typing import Any

import pytest
import torch

from ddpm.image.config import (
    AnyModelConfig,
    AnySamplerConfig,
    AnyScheduleConfig,
    Config,
    CosineScheduleConfig,
    DataConfig,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    DiTModelConfig,
    EulerMaruyamaVESamplerConfig,
    EulerMaruyamaVPSamplerConfig,
    LinearScheduleConfig,
    PFODESamplerConfig,
    PredictorCorrectorSamplerConfig,
    RectifiedFlowSamplerConfig,
    RectifiedFlowScheduleConfig,
    SubVariancePreservingScheduleConfig,
    UNetModelConfig,
    VarianceExplodingScheduleConfig,
    VariancePreservingScheduleConfig,
)
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.diffusion.discrete import (
    CosineDiscreteSchedule,
    DDIMSampler,
    DDPMSampler,
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
)
from ddpm.image.models import create_model
from ddpm.image.models.dit import DiT
from ddpm.image.models.unet import UNetWideResNetAttnCIFAR10

# Every union member paired with what its factory must return. mypy's `assert_never` catches a
# member with no `case`; these tables also check that each case builds the right class.
EXPECTED_SCHEDULES: dict[Any, Any] = {
    LinearScheduleConfig: LinearDiscreteSchedule,
    CosineScheduleConfig: CosineDiscreteSchedule,
    VariancePreservingScheduleConfig: VariancePreservingSchedule,
    VarianceExplodingScheduleConfig: VarianceExplodingSchedule,
    SubVariancePreservingScheduleConfig: SubVariancePreservingSchedule,
    RectifiedFlowScheduleConfig: RectifiedFlowSchedule,
}

# create_sampler asserts the schedule pairing, so each sampler comes with a legal schedule.
EXPECTED_SAMPLERS: dict[Any, tuple[Any, Any]] = {
    DDPMSamplerConfig: (DDPMSampler, LinearScheduleConfig),
    DDIMSamplerConfig: (DDIMSampler, LinearScheduleConfig),
    EulerMaruyamaVPSamplerConfig: (EulerMaruyamaVPSampler, VariancePreservingScheduleConfig),
    EulerMaruyamaVESamplerConfig: (EulerMaruyamaVESampler, VarianceExplodingScheduleConfig),
    PredictorCorrectorSamplerConfig: (
        PredictorCorrectorSampler,
        VariancePreservingScheduleConfig,
    ),
    PFODESamplerConfig: (PFODESampler, VariancePreservingScheduleConfig),
    RectifiedFlowSamplerConfig: (RectifiedFlowSampler, RectifiedFlowScheduleConfig),
}

EXPECTED_MODELS: dict[Any, Any] = {
    UNetModelConfig: UNetWideResNetAttnCIFAR10,
    DiTModelConfig: DiT,
}


def _build(config_cls: Any) -> Any:
    """A config with every required field set to 0.1, which suits PositiveFloat and
    NonNegativeFloat. A required field of another type fails here."""
    required = {name: 0.1 for name, field in config_cls.model_fields.items() if field.is_required()}
    return config_cls(**required)


def test_schedule_union_is_fully_covered() -> None:
    assert set(typing.get_args(AnyScheduleConfig)) == set(EXPECTED_SCHEDULES)


def test_sampler_union_is_fully_covered() -> None:
    assert set(typing.get_args(AnySamplerConfig)) == set(EXPECTED_SAMPLERS)


def test_model_union_is_fully_covered() -> None:
    assert set(typing.get_args(AnyModelConfig)) == set(EXPECTED_MODELS)


@pytest.mark.parametrize("config_cls", typing.get_args(AnyScheduleConfig))
def test_create_schedule_handles_every_union_member(config_cls: Any) -> None:
    schedule = create_schedule(_build(config_cls))

    assert isinstance(schedule, EXPECTED_SCHEDULES[config_cls])


@pytest.mark.parametrize("config_cls", typing.get_args(AnySamplerConfig))
def test_create_sampler_handles_every_union_member(config_cls: Any) -> None:
    expected_cls, schedule_config_cls = EXPECTED_SAMPLERS[config_cls]
    schedule = create_schedule(_build(schedule_config_cls))

    sampler = create_sampler(_build(config_cls), schedule)

    assert isinstance(sampler, expected_cls)


@pytest.mark.parametrize("config_cls", typing.get_args(AnyModelConfig))
def test_create_model_handles_every_union_member(config_cls: Any) -> None:
    config = Config(
        data=DataConfig(num_channels=3, height=8, width=8, num_classes=2),
        model=config_cls(),
        device="cpu",
    )

    model = create_model(config)
    x = torch.randn(2, 3, 8, 8)

    assert isinstance(model, EXPECTED_MODELS[config_cls])
    # The eps_theta contract: predicted noise has the input's shape.
    assert model(x, torch.randint(0, 1000, (2,))).shape == x.shape
