"""One look for every test-suite figure, applied with ``plt.rc_context(RC)``."""

from typing import Any

from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.typing import RcKeyType

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: in this order, never cycled
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
NEUTRAL = "#f0efec"  # a recessive fill, e.g. a valid range
REFERENCE = "#e1e0d9"  # a wide pale line under the data, e.g. a closed form
# One hue, light -> dark, for magnitudes (e.g. a heatmap).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

RC: dict[RcKeyType, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.prop_cycle": cycler(color=SERIES),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": REFERENCE,
    "grid.linewidth": 0.6,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
}
