"""Centralized visual styling for all asymptotic safety plots.

Provides a unified color palette, font settings, and helper functions
so that every visualization in the package has a consistent look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    from asymsafety.analysis.fixed_points import FixedPoint

# ---------------------------------------------------------------------------
# Semantic colour palette
# ---------------------------------------------------------------------------

COLOR_RELEVANT = "#2176AE"       # blue — UV-attractive / relevant directions
COLOR_IRRELEVANT = "#F77F00"     # orange — UV-repulsive / irrelevant
COLOR_NGFP = "#06D6A0"          # green — non-Gaussian fixed point marker
COLOR_GFP = "#FFFFFF"           # white — Gaussian fixed point marker
COLOR_SEPARATRIX = "#EF476F"    # red-pink — separatrix curves
COLOR_TRAJECTORY = "#073B4C"    # dark teal — default trajectory colour
COLOR_SINGULARITY = "#EF476F"   # red-pink — singularity boundaries

COLOR_LITIM = "#2176AE"         # blue — Litim regulator
COLOR_EXPONENTIAL = "#F77F00"   # orange — Exponential regulator

CMAP_FLOW = "coolwarm"
CMAP_SPEED = "inferno"
CMAP_PRESSURE = "coolwarm"

# Qualitative palette for multiple trajectories / exponents
TRAJECTORY_COLORS = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261",
    "#e76f51", "#606c38", "#283618", "#dda15e",
]

# ---------------------------------------------------------------------------
# Matplotlib rcParams style
# ---------------------------------------------------------------------------

ASYM_STYLE: dict[str, object] = {
    # Fonts
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "mathtext.fontset": "cm",
    # Grid
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    # Lines
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    # Legend
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}


def apply_style() -> None:
    """Apply the package-wide matplotlib style."""
    plt.rcParams.update(ASYM_STYLE)


# ---------------------------------------------------------------------------
# Coupling name → LaTeX label
# ---------------------------------------------------------------------------

_COUPLING_LABELS: dict[str, str] = {
    "g": r"$g$",
    "lambda": r"$\lambda$",
    "lambda_": r"$\lambda$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "xi": r"$\xi$",
    "eta": r"$\eta$",
    "lambda_adm": r"$\lambda_{\mathrm{ADM}}$",
}


def coupling_label(name: str) -> str:
    """Return a LaTeX-formatted label for a coupling name."""
    return _COUPLING_LABELS.get(name, f"${name}$")


# ---------------------------------------------------------------------------
# Fixed-point annotation helper
# ---------------------------------------------------------------------------


def annotate_fixed_point(
    ax: Axes,
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    *,
    fontsize: int = 10,
    offset: tuple[float, float] = (12, 12),
) -> None:
    """Draw a marker and coordinate label at a fixed point.

    Args:
        ax: Matplotlib axes.
        fp: The fixed point to annotate.
        x_coupling: Coupling on the x-axis.
        y_coupling: Coupling on the y-axis.
        fontsize: Annotation text size.
        offset: (dx, dy) pixel offset for the annotation text.
    """
    x = fp.location.get(x_coupling, 0.0)
    y = fp.location.get(y_coupling, 0.0)

    if fp.is_gaussian:
        marker, color, edge = "o", COLOR_GFP, "black"
        label = "GFP"
    else:
        marker, color, edge = "*", COLOR_NGFP, "black"
        label = f"NGFP ({fp.relevant_directions} rel.)"

    ax.plot(
        x, y,
        marker=marker, color=color, markersize=13,
        markeredgecolor=edge, markeredgewidth=1.5,
        label=label, zorder=5,
    )

    coord_text = ", ".join(
        f"{coupling_label(c).strip('$')}^*\\!={fp.location[c]:.3f}"
        for c in (x_coupling, y_coupling)
        if c in fp.location
    )
    ax.annotate(
        f"${coord_text}$",
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8),
        zorder=6,
    )


# ---------------------------------------------------------------------------
# Colour-bar helper
# ---------------------------------------------------------------------------


def add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable | None = None,
    *,
    label: str = "",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = CMAP_FLOW,
) -> Colorbar:
    """Add a consistently-styled colourbar.

    If *mappable* is ``None``, a new :class:`ScalarMappable` is created
    from *vmin*, *vmax*, and *cmap*.
    """
    if mappable is None:
        mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
        mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return cbar
