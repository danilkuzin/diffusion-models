from pathlib import Path

import matplotlib.pyplot as plt
import torch

from visualisation.style import INK_SECONDARY, RC, REFERENCE, SERIES


def plot_continuous_schedules(
    out_path: Path, t: torch.Tensor, coefficients: dict[str, tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """Mean coefficient a(t) and std s(t) of x_t | x_0 against t, one line per schedule. t: (N,)
    with t[0] = 0; each (a, s): (N,), (N,)."""
    with plt.rc_context(RC):
        fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
        ax_s.axhline(1.0, color=REFERENCE, lw=6, label="std of N(0, I)")
        for i, (name, (a, s)) in enumerate(coefficients.items()):
            ax_a.plot(t, a, color=SERIES[i], label=name)
            ax_s.plot(t[1:], s[1:], color=SERIES[i])  # s(0) = 0 for VP and sub-VP: off a log axis
        ax_s.set_yscale("log")
        ax_a.set_title("Mean coefficient a(t) (VP and sub-VP coincide)")
        ax_s.set_title("Std s(t) (log scale)")
        for ax in (ax_a, ax_s):
            ax.set_xlabel("t")
        ax_a.legend(loc="center right")
        ax_s.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_diffusion_coefficients(
    out_path: Path, ts: list[float], rows: dict[str, tuple[list[float], list[float]]]
) -> None:
    """Per schedule, g(t)^2 as ``diffusion_sq`` codes it (a line) over dv/dt - 2 f v from the
    marginal variance (a wide pale line): rows map name -> (coded, from the marginal)."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, (name, (coded, from_marginal)) in enumerate(rows.items()):
            label = "dv/dt - 2 f v, by autograd" if i == 0 else None
            ax.plot(ts, from_marginal, color=REFERENCE, lw=6, label=label)
            ax.plot(ts, coded, color=SERIES[i], label=f"{name}: diffusion_sq")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title("g(t)^2: as coded, and from the marginal variance")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_std_ratio_trajectories(
    out_path: Path,
    trajectories: dict[str, list[tuple[float, float]]],
    title: str,
    sampling_descends: bool,
) -> None:
    """Per sampler, (t, std of x / std of the exact marginal) at each model call and at the output,
    against 1."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.axhline(1.0, color=REFERENCE, lw=6, label="the exact marginal")
        for i, (name, trajectory) in enumerate(trajectories.items()):
            ts = [t for t, _ in trajectory]
            ratios = [ratio for _, ratio in trajectory]
            marker = "o" if len(trajectory) <= 60 else None
            ax.plot(ts, ratios, color=SERIES[i], marker=marker, markersize=4, label=name)
            ax.annotate(
                f"{ratios[-1]:.2f}",
                (ts[-1], ratios[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
            )
        if sampling_descends:
            ax.invert_xaxis()
        ax.set_xlabel("t (sampling runs left to right)")
        ax.set_title(title)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_corrector_convergence(
    out_path: Path, runs: dict[str, list[float]], fixed_point: float
) -> None:
    """Per start, std of x / the true std sqrt(v) after each corrector iteration, against 1 and
    ULA's fixed point; right, the last iterations on a linear scale."""
    with plt.rc_context(RC):
        fig, (ax_all, ax_end) = plt.subplots(1, 2, figsize=(11, 3.6))
        n = len(next(iter(runs.values())))
        tail = range(n - n // 3, n)
        for ax, iterations in ((ax_all, range(n)), (ax_end, tail)):
            ax.axhline(1.0, color=REFERENCE, lw=6, label="sqrt(v), the true std")
            ax.axhline(fixed_point, color=INK_SECONDARY, ls="--", lw=1, label="ULA's fixed point")
            for i, (name, ratios) in enumerate(runs.items()):
                ax.plot(iterations, [ratios[k] for k in iterations], color=SERIES[i], label=name)
            ax.set_xlabel("corrector iteration")
        ax_all.set_yscale("log")
        ax_all.set_title("Langevin at fixed s: std of x / sqrt(v) (log scale)")
        ax_end.set_title(f"The last {n // 3} iterations: {fixed_point:.4f}, not 1")
        ax_all.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
