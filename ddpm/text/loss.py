import torch
import torch.nn.functional as F


def mdlm_loss(
    logits: torch.Tensor, x: torch.Tensor, is_masked: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The continuous-time MDLM NELBO (Sahoo et al. 2024) for alpha_t = 1 - t: the cross-entropy
    on masked positions, weighted by -alpha'_t / (1 - alpha_t) = 1/t.

    logits: (B, L, V), x: (B, L) long, is_masked: (B, L) bool, t: (B,)
    -> (the NELBO per character, a scalar to backpropagate;
        the per-sequence NELBO (B,), summed over positions;
        the unweighted mean cross-entropy over masked positions)
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape((B * L, V)), x.reshape((B * L,)), reduction="none").reshape(
        (B, L)
    )  # (B, L)
    masked_ce = ce * is_masked.to(torch.float32)  # (B, L)
    masked_ce_s = masked_ce.sum(dim=(1))  # (B,)
    nelbo_s = masked_ce_s * (1 / t)  # (B,)
    nelbo = (nelbo_s / L).mean()  # scalar

    bert_style_mean = masked_ce.sum() / is_masked.sum().clamp_min(1e-6)

    return nelbo, nelbo_s, bert_style_mean


def ar_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The AR baseline's next-character NLL: the logits at position i score x at i + 1, so
    L - 1 predictions.
    logits: (B, L, V), x: (B, L) long -> (B,) mean NLL per character."""
    B, L, V = logits.shape
    ce = F.cross_entropy(
        logits[:, :-1].reshape((B * (L - 1), V)),
        x[:, 1:].reshape((B * (L - 1),)),
        reduction="none",
    ).reshape(B, (L - 1))  # (B, L-1)

    return ce.mean(dim=1)
