import torch


def alpha(t: torch.Tensor) -> torch.Tensor:
    """alpha_t = 1 - t, the probability that a position is still unmasked at t. (B,) -> (B,)."""

    return 1 - t


def sample_t(B: int, t_min: float, generator: torch.Generator) -> torch.Tensor:
    """(B,) times, one in each bin [i/B, (i+1)/B) before clipping to [t_min, 1] (the
    low-discrepancy sampler of Kingma et al. 2021), shuffled so t doesn't depend on the row."""
    u = torch.rand((1,), generator=generator, device=generator.device)
    i = torch.arange(0, B, dtype=torch.float, device=generator.device) / B
    u = ((u + i) % 1).clip(t_min, 1)
    idx = torch.randperm(u.nelement(), generator=generator, device=generator.device)
    u = u[idx]

    return u


def mask(
    x: torch.Tensor, t: torch.Tensor, mask_id: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks each position independently with probability 1 - alpha_t.
    x: (B, L) long, t: (B,) -> z: (B, L) long, is_masked: (B, L) bool."""
    (B, L) = x.shape
    u = torch.rand((B, L), generator=generator, device=generator.device)
    is_masked = u < 1 - alpha(t.reshape((B, 1)))
    z = torch.where(is_masked, mask_id, x)
    return z, is_masked
