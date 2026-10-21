from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from ddpm.layers import DiTBlock, FinalLayer


def _adaln_linear(block: DiTBlock) -> torch.nn.Linear:
    """Indexing an nn.Sequential is typed as Module, so narrow it for mypy --strict."""
    linear = block.adaln_modulation[-1]
    assert isinstance(linear, torch.nn.Linear)
    return linear


def _mlp_first_linear(block: DiTBlock) -> torch.nn.Linear:
    linear = block.mlp[0]
    assert isinstance(linear, torch.nn.Linear)
    return linear


def test_dit_block_is_identity_at_init() -> None:
    """adaLN-Zero: the gates start at exactly 0, so a fresh block is the identity."""
    B, N, D = 2, 4, 16
    block = DiTBlock(D, num_heads=2)
    x = torch.randn(B, N, D)

    assert torch.equal(block(x, torch.randn(B, D)), x)


def test_dit_block_gate_init_closed_form() -> None:
    """gate_init = g, adaLN weight still 0: for any c, shift = scale = 0 and both gates = g, so
    the block is h = x + g * attn(LN(x)), then h + g * mlp(LN(h))."""
    B, N, D, g = 2, 4, 16, 0.5
    block = DiTBlock(D, num_heads=2, gate_init=g)
    x = torch.randn(B, N, D)

    y = block.norm1(x)
    h = x + g * block.attn(y, y, y, need_weights=False)[0]
    expected = h + g * block.mlp(block.norm2(h))

    assert torch.allclose(block(x, torch.randn(B, D)), expected)


def gate_warmup(
    make_optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer], steps: int
) -> list[tuple[float, float]]:
    """Fits a fresh block to a fixed target -> per step, (mean |gate| over both gates, norm of
    the attention and MLP weights' change since init)."""
    torch.manual_seed(0)
    B, N, D = 16, 8, 16
    block = DiTBlock(D, num_heads=2)
    x, c = torch.randn(B, N, D), torch.randn(B, D)
    target = x + 0.5 * torch.tanh(x @ torch.randn(D, D))
    branches = [block.attn.in_proj_weight, _mlp_first_linear(block).weight]
    initial = [w.detach().clone() for w in branches]
    optimizer = make_optimizer(block.parameters())

    trajectory = []
    for _ in range(steps + 1):
        with torch.no_grad():
            mods = block.adaln_modulation(c).chunk(6, dim=1)
            gate = torch.cat([mods[2], mods[5]]).abs().mean().item()
            change = sum(((w - w_0) ** 2).sum() for w, w_0 in zip(branches, initial, strict=True))
        trajectory.append((gate, float(change) ** 0.5))
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(block(x, c), target).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return trajectory


def test_dit_block_gradient_warmup(viz: Path | None) -> None:
    """adaLN-Zero at init: the gate-producing linear gets gradient, the attention and MLP
    weights get exactly zero (d loss / d branch is proportional to gate = 0)."""
    B, N, D = 2, 4, 16
    block = DiTBlock(D, num_heads=2)
    block(torch.randn(B, N, D), torch.randn(B, D)).sum().backward()

    adaln_grad = _adaln_linear(block).weight.grad
    assert adaln_grad is not None and torch.count_nonzero(adaln_grad) > 0

    attn_grad = block.attn.in_proj_weight.grad
    assert attn_grad is not None and torch.count_nonzero(attn_grad) == 0

    mlp_grad = _mlp_first_linear(block).weight.grad
    assert mlp_grad is not None and torch.count_nonzero(mlp_grad) == 0

    if viz:
        from visualisation.image.dit import plot_gate_warmup

        plot_gate_warmup(
            viz,
            {
                "SGD: the branches' gradient grows with the gates": gate_warmup(
                    lambda p: torch.optim.SGD(p, lr=0.5), 60
                ),
                "Adam: one step late, then rescaled to full size": gate_warmup(
                    lambda p: torch.optim.Adam(p, lr=1e-3), 60
                ),
            },
        )


def test_final_layer_shape_and_zero_init() -> None:
    B, N, D, patch_size, out_channels = 2, 4, 16, 2, 3
    final = FinalLayer(D, patch_size, out_channels)

    out = final(torch.randn(B, N, D), torch.randn(B, D))

    assert out.shape == (B, N, patch_size * patch_size * out_channels)
    assert torch.count_nonzero(out) == 0  # zero-init projection -> exactly zero at init
