import torch
import torch.nn.functional as F

from ddpm.text.model import TextDiT


class AncestralSampler:
    def reverse(
        self,
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
        unmask = torch.rand((B, L), generator=generator, device=z.device) < (t - s) / t
        z = torch.where(unmask & (z == mask_id), candidates, z)
        return z

    def sample(
        self,
        model: TextDiT,
        z: torch.Tensor,
        num_steps: int,
        mask_id: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Ancestral sampling from z (B, L) with masked tokens. Time goes from t = 1 to t = 0 on the
        uniform grid, num_steps model calls -> (B, L) long. Unmasked positions are kept."""
        B, L = z.shape
        timestamps = [i / num_steps for i in range(num_steps, -1, -1)]
        for t, s in zip(timestamps[:-1], timestamps[1:], strict=True):
            logits = model(z, torch.full((B,), t, device=z.device, dtype=torch.float32))
            z = self.reverse(z, logits, t, s, mask_id, generator)
        # the last step has s = 0, so its coin is 1 and no mask survives
        assert torch.all(z < mask_id)

        return z


class ConfidenceSampler:
    """Low-confidence remasking (Nie et al. 2025, §2.4; Chang et al. 2022): each step commits the
    most confident masked positions. The order depends on the model's output, so this is not a
    sample from p_theta. greedy: the argmax as candidate, else a draw."""

    def __init__(self, greedy: bool = False) -> None:
        self.greedy = greedy

    def reverse(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        k_i: int,
        mask_id: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """One step: a candidate per position, its probability as the confidence, and each row's
        k_i most confident masked positions revealed; the rest carry over.
        z: (B, L) long, logits: (B, L, V) -> (B, L) long."""
        B, L = z.shape
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        V = probs.shape[-1]
        if self.greedy:
            candidates = torch.argmax(probs, dim=-1)  # (B, L)
        else:
            candidates = torch.multinomial(
                probs.reshape((B * L, V)), 1, generator=generator
            ).reshape((B, L))
        confidence = probs.gather(-1, candidates[..., None]).squeeze(-1)
        confidence.masked_fill_(z != mask_id, -torch.inf)
        # a stable sort, so ties go to the lowest index
        order = torch.argsort(confidence, dim=-1, descending=True, stable=True)
        idx = order.narrow(-1, 0, k_i)
        z = torch.scatter(z, 1, idx, candidates.gather(1, idx))
        return z

    def sample(
        self,
        model: TextDiT,
        z: torch.Tensor,
        num_steps: int,
        mask_id: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """From z (B, L) long, the same number M of masks in every row: step i commits
        k_i = round(i M / T) - round((i - 1) M / T), T = num_steps model calls -> (B, L) long."""
        B, L = z.shape
        M_t = (z == mask_id).sum(dim=1)
        assert torch.all(M_t == M_t[0])
        M = int(M_t[0])
        timestamps = [i / num_steps for i in range(num_steps, -1, -1)]
        for i in range(1, num_steps + 1):
            k_i = round(i * M / num_steps) - round((i - 1) * M / num_steps)
            logits = model(
                z, torch.full((B,), timestamps[i - 1], device=z.device, dtype=torch.float32)
            )
            z = self.reverse(z, logits, k_i, mask_id, generator)

        assert torch.all(z < mask_id)

        return z
