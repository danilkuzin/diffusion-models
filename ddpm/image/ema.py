import torch
import torch.nn as nn


class EMA:
    """EMA of the weights, Ho et al. 2020, Appendix B: shadow <- decay*shadow + (1-decay)*param.
    Deviations: configs use decay 0.999 (paper: 0.9999); only trainable params are tracked, frozen
    ones come from the raw weights at eval."""

    def __init__(self, model: nn.Module, decay: float):
        self.shadow_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup_weights: dict[str, torch.Tensor] = {}
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # lerp_(end, w) = self + w * (end - self) = decay * shadow + (1 - decay) * param
                self.shadow_weights[name].lerp_(param.data, 1 - self.decay)

    def load_average_weights(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup_weights[name] = param.data.clone()
                param.data.copy_(self.shadow_weights[name])

    def restore_current_weights(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_weights[name])
        self.backup_weights.clear()

    def load_state_dict(self, shadow_weights: dict[str, torch.Tensor]) -> None:
        self.shadow_weights = shadow_weights
