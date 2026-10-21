from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from ddpm.image.data import reverse_transform
from visualisation.style import INK_SECONDARY, RC, SEQUENTIAL, SERIES, SURFACE


def plot_patch_tokens(
    out_path: Path, image: torch.Tensor, tokens: torch.Tensor, patch_size: int
) -> None:
    """Left: ``image``, (C, H, W) uint8, cut into the patch grid. Right: the first two grid rows
    of ``tokens``, (N, p * p * C) in [-1, 1], each read back as a (p, p, C) patch in sequence
    order, as ``unpatchify`` reads it."""
    C, H, _ = image.shape
    G = H // patch_size
    patches = reverse_transform(tokens).reshape(-1, patch_size, patch_size, C)  # (N, p, p, C)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12, 3.6))
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 2.4], wspace=0.06)
        right = outer[0, 1].subgridspec(2, G, wspace=0.2, hspace=0.6)

        ax = fig.add_subplot(outer[0, 0])
        ax.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
        edges = [k * patch_size - 0.5 for k in range(G + 1)]
        ax.hlines(edges, -0.5, H - 0.5, color=SURFACE, lw=0.6)
        ax.vlines(edges, -0.5, H - 0.5, color=SURFACE, lw=0.6)
        ax.add_patch(
            Rectangle((-0.5, -0.5), H, 2 * patch_size, fill=False, edgecolor=SERIES[1], lw=2)
        )
        ax.set_title(f"{G} x {G} patches of {patch_size} x {patch_size}, row-major")
        ax.set_axis_off()

        for row in range(2):
            for col in range(G):
                index = row * G + col
                token_ax = fig.add_subplot(right[row, col])
                token_ax.imshow(patches[index].numpy(), interpolation="nearest")
                token_ax.set_axis_off()
                if col in (0, G - 1):
                    token_ax.set_title(str(index), color=INK_SECONDARY, loc="center")
        heading = outer[0, 1].get_position(fig)
        fig.text(
            heading.x0,
            heading.y1,
            f"Tokens 0 to {2 * G - 1} (the outlined rows), each read back as a "
            f"{patch_size} x {patch_size} patch",
            fontsize=10,
            va="bottom",
        )
        fig.savefig(out_path)
        plt.close(fig)


def plot_pos_embed_similarity(
    out_path: Path, pos_embed: torch.Tensor, references: list[tuple[int, int]]
) -> None:
    """``pos_embed`` (G * G, D), row-major: each reference patch's similarity to every patch,
    <e(p), e(q)> / (D / 2), on the G x G grid."""
    G = int(len(pos_embed) ** 0.5)
    D = pos_embed.shape[1]
    sim = (pos_embed @ pos_embed.T / (D / 2)).reshape(G, G, G, G)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, len(references), figsize=(3.4 * len(references) + 1, 3.4))
        for ax, (row, col) in zip(axs, references, strict=True):
            image = ax.imshow(sim[row, col].numpy(), cmap=SEQUENTIAL, vmin=float(sim.min()), vmax=1)
            ax.plot(col, row, "o", mfc="none", mec=SURFACE, mew=1.5, markersize=9)
            ax.set_title(f"vs patch (row {row}, col {col})")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        fig.colorbar(image, ax=axs, shrink=0.85, label="<e(p), e(q)> / (D / 2)")
        fig.savefig(out_path)
        plt.close(fig)


def plot_gate_warmup(out_path: Path, runs: dict[str, list[tuple[float, float]]]) -> None:
    """Per optimiser, per step: (mean |gate|, norm of the branch weights' change), each as a
    fraction of its last value, so the two curves share an axis."""
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.4), sharey=True)
        for ax, (name, trajectory) in zip(axs, runs.items(), strict=True):
            gates, changes = zip(*trajectory, strict=True)
            steps = range(len(trajectory))
            ax.plot(steps, [g / gates[-1] for g in gates], color=SERIES[0], label="mean |gate|")
            ax.plot(
                steps,
                [w / changes[-1] for w in changes],
                color=SERIES[1],
                label="attention + MLP weights, ||W - W_0||",
            )
            ax.set_title(name)
            ax.set_xlabel("optimiser step")
        axs[0].set_ylabel("fraction of the last step's value")
        axs[0].legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
