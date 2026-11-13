from typing import Literal

import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class TextDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["shakespeare"] = "shakespeare"
    seq_len: PositiveInt = 256
    batch_size: PositiveInt = 64
    train_frac: NonNegativeFloat = 0.9


class TextDiTModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_dit"] = "text_dit"
    heads: PositiveInt = 2
    embed_dim: PositiveInt = 32
    depth: PositiveInt = 6
    mlp_ratio: PositiveFloat = 4.0
    dropout: NonNegativeFloat = 0.0
    time_cond: bool = True
    # The adaLN gate bias at init. 0.0 is DiT's adaLN-Zero (Peebles & Xie 2023), whose closed
    # gates leave no path between positions at init, so training stalls near the unigram entropy.
    gate_init: NonNegativeFloat = 0.5


class MDLMObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mdlm"] = "mdlm"
    t_min: PositiveFloat = 0.001


class ARObjectiveConfig(BaseModel):
    # The AR baseline, not a diffusion model: a reference for MDLM's bound, with no sampler.
    model_config = ConfigDict(extra="forbid")
    type: Literal["ar"] = "ar"


class TextTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: PositiveInt = 30000
    lr: PositiveFloat = 1e-3
    warmup_steps: PositiveInt = 200
    min_lr_frac: PositiveFloat = 0.1
    weight_decay: NonNegativeFloat = 0.1
    beta2: PositiveFloat = 0.99
    grad_clip: PositiveFloat = 1.0
    seed: NonNegativeInt = 0


ObjectiveConfig = MDLMObjectiveConfig | ARObjectiveConfig


class TextEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every: PositiveInt = 500
    K: PositiveInt = 8


class TextLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ckpt_every: PositiveInt = 2000


class TextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str = "text_mdlm"
    data: TextDataConfig = TextDataConfig()
    model: TextDiTModelConfig = TextDiTModelConfig()
    objective: ObjectiveConfig = Field(default_factory=MDLMObjectiveConfig, discriminator="type")
    train: TextTrainConfig = TextTrainConfig()
    eval: TextEvalConfig = TextEvalConfig()
    log: TextLogConfig = TextLogConfig()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_text_config(filepath: str) -> TextConfig:
    with open(filepath) as f:
        data = yaml.safe_load(f)
    config = TextConfig.model_validate(data)

    return config
