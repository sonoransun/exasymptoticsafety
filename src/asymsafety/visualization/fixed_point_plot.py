"""Fixed point and critical exponent visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from asymsafety.analysis.continuation import ContinuationResult
from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.visualization.style import (
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    coupling_label,
)


def plot_critical_exponents(
    continuation: ContinuationResult,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot critical exponents as a function of an external parameter.

    Args:
        continuation: Result of parameter continuation.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with two panels: Re(θ) and Im(θ).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    thetas = continuation.critical_exponents_array
    param = continuation.parameter_values

    n_exponents = thetas.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_exponents))

    for i in range(n_exponents):
        ax1.plot(param, thetas[:, i].real, '-o', color=colors[i],
                markersize=3, label=f"$\\theta_{i+1}$")
        ax2.plot(param, thetas[:, i].imag, '-o', color=colors[i],
                markersize=3)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_ylabel(r"Re($\theta_i$)", fontsize=12)
    ax2.set_ylabel(r"Im($\theta_i$)", fontsize=12)
    ax2.set_xlabel(continuation.parameter_name, fontsize=12)
    ax1.set_title("Critical Exponents", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Shaded bands for relevant / irrelevant regions
    y_lo, y_hi = ax1.get_ylim()
    ax1.axhspan(0, y_hi, color=COLOR_RELEVANT, alpha=0.06)
    ax1.axhspan(y_lo, 0, color=COLOR_IRRELEVANT, alpha=0.06)
    ax1.text(
        0.02, 0.95, "relevant", transform=ax1.transAxes,
        fontsize=9, color=COLOR_RELEVANT, va="top", alpha=0.7,
    )
    ax1.text(
        0.02, 0.05, "irrelevant", transform=ax1.transAxes,
        fontsize=9, color=COLOR_IRRELEVANT, va="bottom", alpha=0.7,
    )
    ax1.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    return fig


def plot_fixed_point_locations(
    continuation: ContinuationResult,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Plot fixed point coupling values as parameter varies."""
    fig, axes = plt.subplots(1, len(continuation.locations),
                             figsize=figsize, squeeze=False)

    for i, (name, values) in enumerate(continuation.locations.items()):
        ax = axes[0, i]
        ax.plot(continuation.parameter_values, values, 'b-o', markersize=3)
        ax.set_xlabel(continuation.parameter_name, fontsize=12)
        ax.set_ylabel(coupling_label(name).rstrip("$") + "^*$", fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fixed Point Location", fontsize=14)
    fig.tight_layout()
    return fig
