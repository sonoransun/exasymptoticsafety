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
from asymsafety.visualization.style import (
    add_colorbar,
    annotate_fixed_point,
    coupling_label,
    COLOR_NGFP,
    COLOR_GFP,
    COLOR_SINGULARITY,
    TRAJECTORY_COLORS,
)


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
    strm = ax.streamplot(X, Y, U, V,
                         color=np.log1p(speed), cmap='coolwarm',
                         density=1.5, linewidth=0.8, arrowsize=1.2)
    add_colorbar(fig, ax, strm.lines, label=r"$\log(1 + |\beta|)$")

    # Singularity line at lambda = 0.5
    if y_coupling == "lambda" and y_range[0] <= 0.5 <= y_range[1]:
        ax.axvline(
            x=0.5, color=COLOR_SINGULARITY, ls="--", lw=1.2, alpha=0.6,
            label=r"$\lambda = \tfrac{1}{2}$ singularity",
        )

    # Mark fixed points
    if fixed_points:
        for fp in fixed_points:
            annotate_fixed_point(ax, fp, x_coupling, y_coupling)

    # Overlay trajectories
    if trajectories:
        for i, traj in enumerate(trajectories):
            x_vals = traj.coupling_values.get(x_coupling)
            y_vals = traj.coupling_values.get(y_coupling)
            if x_vals is not None and y_vals is not None:
                color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
                ax.plot(x_vals, y_vals, '-', color=color,
                        linewidth=1.5, alpha=0.7)

    ax.set_xlabel(coupling_label(x_coupling), fontsize=14)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=14)
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
                        label=coupling_label(name))

    ax.set_xlabel(r"$t = \log(k/k_0)$", fontsize=14)
    ax.set_ylabel("Coupling value", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # UV / IR direction annotations
    ax.annotate(
        r"IR $\leftarrow$", xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=9, color="0.4", va="top",
    )
    ax.annotate(
        r"$\rightarrow$ UV", xy=(0.98, 0.98), xycoords="axes fraction",
        fontsize=9, color="0.4", va="top", ha="right",
    )

    fig.tight_layout()
    return fig


def annotated_eh_phase_portrait(
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Publication-quality annotated phase portrait of Einstein-Hilbert RG flow.

    Self-contained "hero figure" that builds the EH system, finds fixed points,
    computes stability, integrates representative trajectories, and produces
    a fully annotated streamplot with eigenvector arrows, singularity boundary,
    and UV/IR labels.

    Args:
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.  Always returns a figure even if some
        analysis steps fail.
    """
    from asymsafety.beta.einstein_hilbert import build_eh_beta_system
    from asymsafety.analysis.fixed_points import FixedPointFinder
    from asymsafety.analysis.stability import analyze_stability
    from asymsafety.visualization.style import (
        CMAP_FLOW,
        COLOR_RELEVANT,
        COLOR_IRRELEVANT,
    )

    # ------------------------------------------------------------------
    # 1. Build the Einstein-Hilbert beta function system
    # ------------------------------------------------------------------
    system = build_eh_beta_system(d=4)

    x_coupling = "g"
    y_coupling = "lambda"
    x_range = (0.0, 1.2)
    y_range = (-0.3, 0.45)
    n_grid = 30

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ------------------------------------------------------------------
    # 2. Find fixed points
    # ------------------------------------------------------------------
    finder = FixedPointFinder(system)
    gfp: FixedPoint | None = None
    ngfp: FixedPoint | None = None

    try:
        gfp = finder.find_fixed_point({"g": 0.01, "lambda": 0.01})
    except Exception:
        pass

    try:
        ngfp = finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3. Stability analysis at the NGFP
    # ------------------------------------------------------------------
    sa = None
    if ngfp is not None:
        try:
            sa = analyze_stability(system, ngfp)
        except Exception:
            sa = None

    # ------------------------------------------------------------------
    # 4. Integrate representative trajectories
    # ------------------------------------------------------------------
    initial_conditions_list = [
        {"g": 0.1, "lambda": 0.05},
        {"g": 0.3, "lambda": -0.1},
        {"g": 1.0, "lambda": 0.1},
        {"g": 0.5, "lambda": 0.3},
        {"g": 0.8, "lambda": -0.2},
        {"g": 0.2, "lambda": 0.2},
    ]

    trajectories: list[RGTrajectory] = []
    integrator = FlowIntegrator(system)
    for ic in initial_conditions_list:
        try:
            traj = integrator.integrate(ic, t_span=(-5, 5))
            trajectories.append(traj)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5. Streamplot on a 30x30 grid
    # ------------------------------------------------------------------
    rhs = system.rhs_vector()
    names = system.coupling_names
    x_idx = names.index(x_coupling)
    y_idx = names.index(y_coupling)

    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

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

    speed = np.sqrt(U**2 + V**2)
    speed_safe = np.where(speed == 0, 1, speed)

    strm = ax.streamplot(
        X, Y, U, V,
        color=np.log1p(speed_safe),
        cmap=CMAP_FLOW,
        density=1.5,
        linewidth=0.8,
        arrowsize=1.2,
    )

    # Colorbar
    add_colorbar(fig, ax, strm.lines, label=r"$\log(1 + |\beta|)$")

    # ------------------------------------------------------------------
    # 6. Draw fixed points
    # ------------------------------------------------------------------
    if gfp is not None:
        annotate_fixed_point(ax, gfp, x_coupling, y_coupling)
    if ngfp is not None:
        annotate_fixed_point(ax, ngfp, x_coupling, y_coupling)

    # ------------------------------------------------------------------
    # 7. Eigenvector arrows at the NGFP
    # ------------------------------------------------------------------
    if sa is not None and ngfp is not None:
        fp_g = ngfp.location["g"]
        fp_lam = ngfp.location["lambda"]
        arrow_scale = 0.2

        for col_idx in range(sa.eigenvectors.shape[1]):
            evec = sa.eigenvectors[:, col_idx].real
            theta = sa.critical_exponents[col_idx]

            # Normalize to desired length in coupling space
            norm = np.sqrt(evec[x_idx]**2 + evec[y_idx]**2)
            if norm < 1e-12:
                continue
            dx = arrow_scale * evec[x_idx] / norm
            dy = arrow_scale * evec[y_idx] / norm

            is_relevant = theta.real > 0
            color = COLOR_RELEVANT if is_relevant else COLOR_IRRELEVANT
            label = (r"relevant ($\theta > 0$)" if is_relevant
                     else r"irrelevant ($\theta < 0$)")

            ax.annotate(
                "",
                xy=(fp_g + dx, fp_lam + dy),
                xytext=(fp_g, fp_lam),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
                zorder=4,
            )
            ax.annotate(
                "",
                xy=(fp_g - dx, fp_lam - dy),
                xytext=(fp_g, fp_lam),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
                zorder=4,
            )
            # Label near the positive arrow tip
            ax.text(
                fp_g + dx * 1.15, fp_lam + dy * 1.15,
                label,
                fontsize=9,
                color=color,
                ha="center",
                va="center",
                zorder=7,
            )

    # ------------------------------------------------------------------
    # 8. Overlay trajectories
    # ------------------------------------------------------------------
    for i, traj in enumerate(trajectories):
        x_vals = traj.coupling_values.get(x_coupling)
        y_vals = traj.coupling_values.get(y_coupling)
        if x_vals is not None and y_vals is not None:
            color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
            ax.plot(x_vals, y_vals, "-", color=color,
                    linewidth=1.5, alpha=0.7)

    # ------------------------------------------------------------------
    # 9. Singularity boundary at lambda = 1/2
    # ------------------------------------------------------------------
    if y_range[0] <= 0.5 <= y_range[1]:
        ax.axhline(
            y=0.5,
            color=COLOR_SINGULARITY,
            ls="--",
            lw=1.2,
            alpha=0.6,
            label=r"$\lambda = \tfrac{1}{2}$ singularity",
        )

    # ------------------------------------------------------------------
    # 10. Labels and annotations
    # ------------------------------------------------------------------
    ax.set_xlabel(coupling_label(x_coupling), fontsize=14)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(
        r"Einstein–Hilbert RG Flow ($d=4$, Litim regulator)",
        fontsize=14,
    )

    # UV / IR region labels
    if ngfp is not None:
        ax.text(
            ngfp.location["g"] + 0.15,
            ngfp.location["lambda"] + 0.08,
            "UV",
            fontsize=12,
            fontweight="bold",
            color="0.3",
            ha="center",
            va="center",
            zorder=7,
        )
    ax.text(
        0.08, -0.15,
        "IR",
        fontsize=12,
        fontweight="bold",
        color="0.3",
        ha="center",
        va="center",
        zorder=7,
    )

    ax.legend(loc="upper left", fontsize=10)

    # ------------------------------------------------------------------
    # 11. Finalize
    # ------------------------------------------------------------------
    fig.tight_layout()
    return fig
