"""Vectorized phase portrait grid evaluation."""

from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure

from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.analysis.fixed_points import FixedPoint


def phase_portrait_2d_accelerated(
    system: BetaFunctionSystem,
    x_coupling: str,
    y_coupling: str,
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 25,
    fixed_points: list[FixedPoint] | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """2D phase portrait using batch evaluation (no Python loops).

    Replaces the nested for-loop in phase_portrait_2d with a single
    batch evaluation call.
    """
    import matplotlib.pyplot as plt

    names = system.coupling_names
    x_idx = names.index(x_coupling)
    y_idx = names.index(y_coupling)

    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    # Build (N, n_couplings) array
    N_points = n_grid * n_grid
    points = np.zeros((N_points, system.dimension))
    points[:, x_idx] = X.ravel()
    points[:, y_idx] = Y.ravel()

    # Batch evaluate
    evaluator = system.batch_evaluator("numpy")
    betas = evaluator.evaluate_batch(points)

    U = betas[:, x_idx].reshape(n_grid, n_grid)
    V = betas[:, y_idx].reshape(n_grid, n_grid)

    # Replace NaN with 0 for plotting
    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.streamplot(X, Y, U, V, color=np.log1p(speed), cmap='coolwarm',
                  density=1.5, linewidth=0.8, arrowsize=1.2)

    if fixed_points:
        for fp in fixed_points:
            x_val = fp.location.get(x_coupling, 0)
            y_val = fp.location.get(y_coupling, 0)
            marker = 'o' if fp.is_gaussian else '*'
            color = 'white' if fp.is_gaussian else 'yellow'
            label = 'GFP' if fp.is_gaussian else f'NGFP ({fp.relevant_directions} rel.)'
            ax.plot(x_val, y_val, marker=marker, color=color,
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=1.5, label=label, zorder=5)
        ax.legend(fontsize=10)

    ax.set_xlabel(f"${x_coupling}$", fontsize=14)
    ax.set_ylabel(f"${y_coupling}$", fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title("RG Flow Phase Portrait (accelerated)", fontsize=14)
    fig.tight_layout()
    return fig
