import torch
import torch.nn.functional as F

from ddpm.text.model import TextDiT


def reverse(
    z: torch.Tensor,
    logits: torch.Tensor,
    t: float,
    s: float,
    mask_id: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """One reverse step t -> s < t (Sahoo et al. 2024, §3.2.2): masked positions unmask with
    probability (t - s)/t to a draw from the model; unmasked ones carry over.
    z: (B, L) long, logits: (B, L, V) -> (B, L) long."""
    B, L = z.shape
    # float64: rounding in a float32 softmax lowers the samples' entropy
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    V = probs.shape[-1]
    candidates = torch.multinomial(probs.reshape((B * L, V)), 1, generator=generator).reshape(
        (B, L)
    )
    unmask = torch.rand((B, L), generator=generator, device=generator.device) < (t - s) / t
    z = torch.where(unmask & (z == mask_id), candidates, z)
    return z


def sample(
    model: TextDiT,
    B: int,
    L: int,
    num_steps: int,
    mask_id: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Ancestral sampling from all-masked at t = 1 to t = 0 on the uniform grid, num_steps model
    calls -> (B, L) long."""
    z = torch.full((B, L), mask_id, dtype=torch.long, device=generator.device)
    timestamps = [i / num_steps for i in range(num_steps, -1, -1)]
    for t, s in zip(timestamps[:-1], timestamps[1:], strict=True):
        logits = model(z, torch.full((B,), t, device=generator.device, dtype=torch.float32))
        z = reverse(z, logits, t, s, mask_id, generator)
    # the last step has s = 0, so its coin is 1 and no mask survives
    assert torch.all(z < mask_id)

    return z
