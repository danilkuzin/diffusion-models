from abc import ABC, abstractmethod
from typing import Any, assert_never

import torch

from ddpm.text.config import ARObjectiveConfig, MDLMObjectiveConfig, TextConfig
from ddpm.text.loss import ar_loss, mdlm_loss
from ddpm.text.model import TextDiT
from ddpm.text.schedule import mask, sample_t


class Objective(ABC):
    @abstractmethod
    def loss_stats(
        self, x: torch.Tensor, generator: torch.Generator, model: TextDiT
    ) -> dict[str, Any]:
        """x: (B, L) long -> {"loss": a scalar to backpropagate, "loss_per_seq": (B,) loss per
        character, whose mean is "loss"; plus objective-specific stats}."""
        pass


class MDLMObjective(Objective):
    def __init__(self, mask_id: int, objective: MDLMObjectiveConfig) -> None:
        self.mask_id = mask_id
        self.t_min = objective.t_min

    def loss_stats(
        self, x: torch.Tensor, generator: torch.Generator, model: TextDiT
    ) -> dict[str, Any]:
        t = sample_t(x.shape[0], self.t_min, generator)
        (z, is_masked) = mask(x, t, self.mask_id, generator)

        logits = model(z, t)
        loss, nelbo_s, bert_style_mean = mdlm_loss(logits, x, is_masked, t)

        return {
            "loss": loss,
            "is_masked": is_masked,
            "loss_per_seq": nelbo_s / x.shape[1],
            "t": t,
            # without the 1/t weight, so the small-t noise in "loss" doesn't reach it
            "masked_ce": bert_style_mean,
        }


class ARObjective(Objective):
    """The AR baseline, not a diffusion model: the exact next-character NLL, a reference for
    MDLM's bound. It has no sampler."""

    def loss_stats(
        self, x: torch.Tensor, generator: torch.Generator, model: TextDiT
    ) -> dict[str, Any]:

        logits = model(x, None)
        per_seq = ar_loss(logits, x)

        return {"loss": per_seq.mean(), "loss_per_seq": per_seq}


def create_objective(config: TextConfig, mask_id: int) -> Objective:
    match config.objective:
        case MDLMObjectiveConfig() as objective:
            return MDLMObjective(mask_id, objective)
        case ARObjectiveConfig():
            return ARObjective()
        case _:
            assert_never(config.objective)


def build_model(config: TextConfig, V: int) -> TextDiT:
    # The objective fixes the attention pattern: MDLM is bidirectional and (unless ablated)
    # time-conditioned; the AR baseline is causal and never time-conditioned.
    causal = isinstance(config.objective, ARObjectiveConfig)
    time_cond = config.model.time_cond and not causal
    return TextDiT(config.model, V, config.data.seq_len, causal=causal, time_cond=time_cond)
