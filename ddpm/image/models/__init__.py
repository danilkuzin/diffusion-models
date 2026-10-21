from typing import assert_never

from ddpm.image.config import Config, DiTModelConfig, UNetModelConfig
from ddpm.image.models.base import BaseModel
from ddpm.image.models.dit import DiT
from ddpm.image.models.unet import UNetWideResNetAttnCIFAR10


def create_model(config: Config) -> BaseModel:
    match config.model:
        case UNetModelConfig():
            return UNetWideResNetAttnCIFAR10(config)
        case DiTModelConfig():
            return DiT(config)
        case _:
            assert_never(config.model)
