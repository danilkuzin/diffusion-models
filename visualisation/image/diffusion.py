from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ddpm.image.data import reverse_transform, transform
from ddpm.image.diffusion.discrete import DiscreteSchedule, sample_clipping
from visualisation.style import INK_SECONDARY, NEUTRAL, RC, REFERENCE, SERIES, SURFACE


def plot_schedules(out_path: Path, schedules: dict[str, DiscreteSchedule]) -> None:
    """beta_t, alpha_hat_t and log-SNR(t) = log(alpha_hat_t / (1 - alpha_hat_t)) against the
    paper's t = 1..T, one line per schedule, with each alpha_hat = 0.5 crossing marked."""
    with plt.rc_context(RC):
        fig, (ax_beta, ax_ah, ax_snr) = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
        for i, (name, schedule) in enumerate(schedules.items()):
            t = torch.arange(1, len(schedule.beta) + 1).numpy()  # index i is the paper's t = i + 1
            ah = schedule.alpha_hat
            ax_beta.plot(t, schedule.beta.numpy(), color=SERIES[i], label=name)
            ax_ah.plot(t, ah.numpy(), color=SERIES[i])
            ax_snr.plot(t, torch.log(ah / (1 - ah)).numpy(), color=SERIES[i])

            half = int(torch.argmax((ah < 0.5).int()))  # first index below 0.5
            ax_ah.plot(half + 1, 0.5, "o", color=SERIES[i], markersize=6, markeredgecolor=SURFACE)
            ax_ah.annotate(
                f"t = {half + 1}",
                (half + 1, 0.5),
                xytext=(6, 4),
                textcoords="offset points",
                color=INK_SECONDARY,
            )

        ax_beta.set_yscale("log")
        ax_beta.set_title(r"Noise added per step, $\beta_t$ (log scale)")
        ax_ah.set_title(r"Signal kept, $\bar\alpha_t$ (● where it halves)")
        ax_snr.set_title(r"log-SNR, $\log \bar\alpha_t / (1 - \bar\alpha_t)$")
        for ax in (ax_beta, ax_ah, ax_snr):
            ax.set_xlabel("t")
        fig.legend(loc="lower center", ncol=len(schedules), bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_forward_strip(
    out_path: Path, x_0: torch.Tensor, schedules: dict[str, DiscreteSchedule], ts: list[int]
) -> None:
    """One image, (C, H, W) uint8, noised by Eq. 4 at each of the paper's t in ``ts`` (t = 0 is
    x_0 itself): one row per schedule, the same eps in every cell."""
    x = transform(x_0[None], "cpu")  # (1, C, H, W) in [-1, 1]
    eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(
            len(schedules), len(ts), figsize=(1.2 * len(ts), 1.3 * len(schedules)), squeeze=False
        )
        for row, (name, schedule) in zip(axs, schedules.items(), strict=True):
            for ax, t in zip(row, ts, strict=True):
                x_t = x if t == 0 else schedule.forward(x, torch.tensor([[t - 1]]), eps)
                img = reverse_transform(x_t)[0].permute(1, 2, 0).numpy()  # (H, W, C) uint8
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            row[0].set_ylabel(name, rotation=0, ha="right", va="center")
        for ax, t in zip(axs[0], ts, strict=True):
            ax.set_title(f"t = {t}", loc="center", color=INK_SECONDARY)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_std_trajectories(
    out_path: Path,
    trajectories: dict[str, list[tuple[int, float]]],
    reference: list[tuple[int, float]],
) -> None:
    """Per sampler, the std of the x the model sees at each call, (index t, std) with t = -1 the
    output, against ``reference``, q(x_t)'s std; left absolute, right as a ratio to it."""
    marginal = dict(reference)
    with plt.rc_context(RC):
        fig, (ax_std, ax_ratio) = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
        ref_t, ref_std = zip(*reference, strict=True)
        ax_std.plot([t + 1 for t in ref_t], ref_std, color=REFERENCE, lw=6, label="std of q(x_t)")
        ax_ratio.axhline(1.0, color=REFERENCE, lw=6)
        for i, (name, trajectory) in enumerate(trajectories.items()):
            ts = [t + 1 for t, _ in trajectory]  # the paper's t; the output lands at 0
            stds = [std for _, std in trajectory]
            marker = "o" if len(trajectory) <= 60 else None
            ax_std.plot(ts, stds, color=SERIES[i], marker=marker, markersize=4, label=name)
            ratios = [std / marginal[t] for t, std in trajectory]
            ax_ratio.plot(ts, ratios, color=SERIES[i], marker=marker, markersize=4)
            ax_ratio.annotate(
                f"{ratios[-1]:.2f}",
                (0, ratios[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
            )
        ax_std.invert_xaxis()  # sampling runs from t = T to 0
        ax_std.set_title("std of the x each model call sees")
        ax_ratio.set_title("... as a fraction of q(x_t)'s std")
        for ax in (ax_std, ax_ratio):
            ax.set_xlabel("t (sampling runs left to right)")
        ax_std.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_clipping(out_path: Path, x: torch.Tensor, ah_t: torch.Tensor, eps: torch.Tensor) -> None:
    """``sample_clipping`` in the (pred_x0, eps) plane: x_t fixes each pixel to the line
    sqrt(ah) pred_x0 + sqrt(1 - ah) eps = x_t, and an out-of-range pred_x0 slides along it to
    +-1. x, eps: (B, C, H, W), the eps the model gave; ah_t: 0-dim."""
    pred_x0 = (x - torch.sqrt(1 - ah_t) * eps) / torch.sqrt(ah_t)  # DDIM Eq. 9, unclipped
    safe_pred_x0, safe_eps = sample_clipping(x, ah_t, eps)
    idx = torch.linspace(0, x.numel() - 1, 48).long()  # an even spread of pixels
    p, e, sp, se = (v.flatten()[idx].numpy() for v in (pred_x0, eps, safe_pred_x0, safe_eps))
    moved = abs(p) > 1
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.axvspan(-1, 1, color=NEUTRAL, label="valid pred_x0, [-1, 1]")
        for p_i, e_i, sp_i, se_i in zip(p[moved], e[moved], sp[moved], se[moved], strict=True):
            ax.annotate(
                "",
                xy=(float(sp_i), float(se_i)),
                xytext=(float(p_i), float(e_i)),
                arrowprops={"arrowstyle": "-|>", "color": SERIES[1], "lw": 1, "shrinkB": 3},
            )
        ax.plot(p[~moved], e[~moved], "o", color=SERIES[0], label="in range: unchanged")
        ax.plot(
            p[moved], e[moved], "o", color=SERIES[1], mfc=SURFACE, label="out of range, as given"
        )
        ax.plot(sp[moved], se[moved], "o", color=SERIES[1], label="clipped, on the same x_t line")
        ax.set_xlabel("pred_x0 (DDIM Eq. 9)")
        ax.set_ylabel("eps")
        ax.set_title("Static thresholding keeps x_t: each arrow runs along its x_t line")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
