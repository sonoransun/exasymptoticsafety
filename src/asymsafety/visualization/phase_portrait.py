"""Phase portrait visualization for RG flows.

Produces 2D streamplots showing the RG flow in coupling space,
with fixed points marked and separatrices highlighted.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import FlowIntegrator, RGTrajectory
from asymsafety.beta.system import BetaFunctionSystem


def phase_portrait_2d(
    system: BetaFunctionSystem,
    x_coupling: str,
    y_coupling: str,
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 25,
    fixed_points: list[FixedPoint] | None = None,
    trajectories: list[RGTrajectory] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Create a 2D phase portrait of the RG flow.

    Args:
        system: Beta function system.
        x_coupling: Name of the coupling for the x-axis.
        y_coupling: Name of the coupling for the y-axis.
        x_range: (x_min, x_max) range.
        y_range: (y_min, y_max) range.
        n_grid: Number of grid points per axis for streamplot.
        fixed_points: Fixed points to mark on the plot.
        trajectories: RG trajectories to overlay.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create grid for streamplot
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    # Evaluate beta functions on the grid
    rhs = system.rhs_vector()
    names = system.coupling_names
    x_idx = names.index(x_coupling)
    y_idx = names.index(y_coupling)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(n_grid):
        for j in range(n_grid):
            point = np.zeros(len(names))
            point[x_idx] = X[i, j]
            point[y_idx] = Y[i, j]
            try:
                beta_vals = rhs(0, point)
                U[i, j] = beta_vals[x_idx]
                V[i, j] = beta_vals[y_idx]
            except (ValueError, ZeroDivisionError, OverflowError):
                U[i, j] = 0
                V[i, j] = 0

    # Normalize arrow lengths for visibility
    speed = np.sqrt(U**2 + V**2)
    speed = np.where(speed == 0, 1, speed)

    # Streamplot
    ax.streamplot(X, Y, U, V,
                  color=np.log1p(speed), cmap='coolwarm',
                  density=1.5, linewidth=0.8, arrowsize=1.2)

    # Mark fixed points
    if fixed_points:
        for fp in fixed_points:
            x_val = fp.location.get(x_coupling, 0)
            y_val = fp.location.get(y_coupling, 0)
            if fp.is_gaussian:
                marker = 'o'
                color = 'white'
                label = 'GFP'
            else:
                marker = '*'
                color = 'yellow'
                label = f'NGFP ({fp.relevant_directions} rel.)'
            ax.plot(x_val, y_val, marker=marker, color=color,
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=1.5, label=label, zorder=5)

    # Overlay trajectories
    if trajectories:
        for traj in trajectories:
            x_vals = traj.coupling_values.get(x_coupling)
            y_vals = traj.coupling_values.get(y_coupling)
            if x_vals is not None and y_vals is not None:
                ax.plot(x_vals, y_vals, 'k-', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(f"${x_coupling}$", fontsize=14)
    ax.set_ylabel(f"${y_coupling}$", fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("RG Flow Phase Portrait", fontsize=14)

    if fixed_points:
        ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


def flow_diagram(
    trajectories: list[RGTrajectory],
    coupling_names: list[str] | None = None,
    title: str = "Running Couplings",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot running couplings vs RG scale.

    Args:
        trajectories: List of RG trajectories to plot.
        coupling_names: Which couplings to plot (default: all).
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for traj in trajectories:
        names = coupling_names or traj.coupling_names
        for name in names:
            if name in traj.coupling_values:
                ax.plot(traj.t_values, traj.coupling_values[name],
                        label=f"${name}$")

    ax.set_xlabel(r"$t = \log(k/k_0)$", fontsize=14)
    ax.set_ylabel("Coupling value", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
