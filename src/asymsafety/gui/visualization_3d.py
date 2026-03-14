"""3D visualization of RG flows, phase portraits, and fixed point stability.

Provides mplot3d-based plotting functions that accept domain objects
(BetaFunctionSystem, FixedPoint, RGTrajectory, StabilityAnalysis) and
return matplotlib Figure instances, following the same conventions as
the 2D visualization module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers '3d' projection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import RGTrajectory
from asymsafety.analysis.stability import StabilityAnalysis
from asymsafety.beta.system import BetaFunctionSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coupling_index(names: list[str], coupling: str) -> int:
    """Return the index of *coupling* in *names*, or -1 if absent."""
    try:
        return names.index(coupling)
    except ValueError:
        return -1


def _get_trajectory_values(
    traj: RGTrajectory,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract (x, y, z) arrays from a trajectory, padding with zeros
    for couplings that are absent."""
    if traj.t_values is None or len(traj.t_values) == 0:
        return None
    n = len(traj.t_values)
    x = traj.coupling_values.get(x_coupling, np.zeros(n))
    y = traj.coupling_values.get(y_coupling, np.zeros(n))
    z = traj.coupling_values.get(z_coupling, np.zeros(n))
    return x, y, z


def _fp_coords(
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> tuple[float, float, float]:
    """Return (x, y, z) location of a fixed point, defaulting to 0."""
    return (
        fp.location.get(x_coupling, 0.0),
        fp.location.get(y_coupling, 0.0),
        fp.location.get(z_coupling, 0.0),
    )


def _eigenvector_components(
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> list[tuple[np.ndarray, complex]]:
    """Return list of (3-component eigenvector, critical exponent) pairs.

    Each eigenvector is projected onto the three chosen coupling axes.
    If the fixed point has no eigenvector data the list is empty.
    """
    if fp.eigenvectors is None or fp.eigenvectors.size == 0:
        return []
    if fp.critical_exponents is None or fp.critical_exponents.size == 0:
        return []

    names = list(fp.location.keys())
    x_idx = _coupling_index(names, x_coupling)
    y_idx = _coupling_index(names, y_coupling)
    z_idx = _coupling_index(names, z_coupling)

    n_dim = fp.eigenvectors.shape[0]
    results: list[tuple[np.ndarray, complex]] = []
    for col in range(fp.eigenvectors.shape[1]):
        vec = fp.eigenvectors[:, col]
        vx = vec[x_idx].real if 0 <= x_idx < n_dim else 0.0
        vy = vec[y_idx].real if 0 <= y_idx < n_dim else 0.0
        vz = vec[z_idx].real if 0 <= z_idx < n_dim else 0.0
        results.append((np.array([vx, vy, vz]), fp.critical_exponents[col]))
    return results


def _draw_fixed_points(
    ax,
    fixed_points: Sequence[FixedPoint],
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> None:
    """Mark fixed points on *ax* using the project's marker conventions."""
    for fp in fixed_points:
        xv, yv, zv = _fp_coords(fp, x_coupling, y_coupling, z_coupling)
        if fp.is_gaussian:
            ax.plot(
                [xv], [yv], [zv],
                marker='o', color='white', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label='GFP', zorder=5,
            )
        else:
            ax.plot(
                [xv], [yv], [zv],
                marker='*', color='limegreen', markersize=14,
                markeredgecolor='black', markeredgewidth=0.8,
                label=f'NGFP ({fp.relevant_directions} rel.)', zorder=5,
            )


def _draw_eigenvectors(
    ax,
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    scale: float,
) -> None:
    """Draw eigenvector arrows at *fp* using ax.quiver.

    Blue arrows for relevant directions (Re(theta) > 0),
    orange arrows for irrelevant (Re(theta) < 0).
    """
    ev_pairs = _eigenvector_components(fp, x_coupling, y_coupling, z_coupling)
    if not ev_pairs:
        return

    xv, yv, zv = _fp_coords(fp, x_coupling, y_coupling, z_coupling)

    for vec, theta in ev_pairs:
        norm = np.linalg.norm(vec)
        if norm < 1e-14:
            continue
        direction = vec / norm * scale
        color = 'dodgerblue' if theta.real > 0 else 'darkorange'
        ax.quiver(
            xv, yv, zv,
            direction[0], direction[1], direction[2],
            color=color, arrow_length_ratio=0.15, linewidth=1.8,
        )
        # Also draw the opposite direction for bidirectional indication
        ax.quiver(
            xv, yv, zv,
            -direction[0], -direction[1], -direction[2],
            color=color, arrow_length_ratio=0.15, linewidth=1.8,
            alpha=0.4,
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def flow_trajectories_3d(
    trajectories: list[RGTrajectory],
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    fixed_points: list[FixedPoint] | None = None,
    show_eigenvectors: bool = True,
    eigenvector_scale: float = 0.3,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot RG trajectories in 3D coupling space.

    Each trajectory is drawn as a line whose colour varies from blue
    (IR / early RG time) to red (UV / late RG time).

    Args:
        trajectories: RG trajectories to plot.
        x_coupling: Coupling name for the x-axis.
        y_coupling: Coupling name for the y-axis.
        z_coupling: Coupling name for the z-axis.
        fixed_points: Fixed points to mark.
        show_eigenvectors: Whether to draw eigenvector arrows at FPs.
        eigenvector_scale: Length scale for eigenvector arrows.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories with colour gradient along RG time
    for traj in trajectories:
        vals = _get_trajectory_values(traj, x_coupling, y_coupling, z_coupling)
        if vals is None:
            continue
        xs, ys, zs = vals
        n = len(xs)
        if n < 2:
            continue

        # Segment-wise colouring: blue (t_min) -> red (t_max)
        t = traj.t_values
        t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-30)
        cmap = plt.cm.coolwarm
        for k in range(n - 1):
            ax.plot(
                xs[k:k + 2], ys[k:k + 2], zs[k:k + 2],
                color=cmap(t_norm[k]),
                linewidth=1.4, alpha=0.85,
            )

    # Fixed points
    if fixed_points:
        _draw_fixed_points(ax, fixed_points, x_coupling, y_coupling, z_coupling)

        if show_eigenvectors:
            for fp in fixed_points:
                _draw_eigenvectors(
                    ax, fp,
                    x_coupling, y_coupling, z_coupling,
                    scale=eigenvector_scale,
                )

    ax.set_xlabel(f"${x_coupling}$", fontsize=13)
    ax.set_ylabel(f"${y_coupling}$", fontsize=13)
    ax.set_zlabel(f"${z_coupling}$", fontsize=13)
    ax.set_title("RG Flow Trajectories (3D)", fontsize=14)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, int] = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = 1
            unique_h.append(h)
            unique_l.append(l)
    if unique_l:
        ax.legend(unique_h, unique_l, fontsize=10)

    fig.tight_layout()
    return fig


def phase_portrait_3d(
    system: BetaFunctionSystem,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 0.5),
    z_range: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 8,
    fixed_points: list[FixedPoint] | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Create a 3D quiver phase portrait of the RG flow.

    Beta-function vectors are evaluated on a regular grid and rendered
    as arrows coloured by the logarithmic speed log(1 + |beta|).

    Args:
        system: Beta function system.
        x_coupling: Coupling name for the x-axis.
        y_coupling: Coupling name for the y-axis.
        z_coupling: Coupling name for the z-axis.
        x_range: (min, max) for the x-axis.
        y_range: (min, max) for the y-axis.
        z_range: (min, max) for the z-axis.
        n_grid: Number of grid points per axis.
        fixed_points: Fixed points to mark.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Build evaluation grid
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    zs = np.linspace(z_range[0], z_range[1], n_grid)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    names = system.coupling_names
    rhs = system.rhs_vector()
    x_idx = _coupling_index(names, x_coupling)
    y_idx = _coupling_index(names, y_coupling)
    z_idx = _coupling_index(names, z_coupling)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)

    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                point = np.zeros(len(names))
                if x_idx >= 0:
                    point[x_idx] = X[i, j, k]
                if y_idx >= 0:
                    point[y_idx] = Y[i, j, k]
                if z_idx >= 0:
                    point[z_idx] = Z[i, j, k]
                try:
                    beta_vals = rhs(0, point)
                    U[i, j, k] = beta_vals[x_idx] if x_idx >= 0 else 0.0
                    V[i, j, k] = beta_vals[y_idx] if y_idx >= 0 else 0.0
                    W[i, j, k] = beta_vals[z_idx] if z_idx >= 0 else 0.0
                except (ValueError, ZeroDivisionError, OverflowError):
                    U[i, j, k] = 0.0
                    V[i, j, k] = 0.0
                    W[i, j, k] = 0.0

    # Colour by log-speed
    speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    log_speed = np.log1p(speed)
    # Normalise to [0, 1] for colourmap
    ls_min, ls_max = log_speed.min(), log_speed.max()
    if ls_max - ls_min > 1e-30:
        c_norm = (log_speed - ls_min) / (ls_max - ls_min)
    else:
        c_norm = np.zeros_like(log_speed)

    cmap = plt.cm.coolwarm
    # Flatten arrays for quiver
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    Uf = U.ravel()
    Vf = V.ravel()
    Wf = W.ravel()
    cf = c_norm.ravel()

    # Normalise arrow lengths for visibility
    max_speed = speed.max()
    if max_speed > 1e-30:
        arrow_len = (x_range[1] - x_range[0]) / n_grid * 0.4
        norm_factor = arrow_len / max_speed
        Uf = Uf * norm_factor
        Vf = Vf * norm_factor
        Wf = Wf * norm_factor

    # mplot3d quiver does not accept per-arrow colours directly, so we
    # batch arrows into a small number of colour bins.
    n_bins = 16
    bins = np.linspace(0, 1, n_bins + 1)
    for b in range(n_bins):
        mask = (cf >= bins[b]) & (cf < bins[b + 1])
        if b == n_bins - 1:
            mask = mask | (cf >= bins[b + 1])  # include max
        if not np.any(mask):
            continue
        colour = cmap((bins[b] + bins[b + 1]) / 2)
        ax.quiver(
            Xf[mask], Yf[mask], Zf[mask],
            Uf[mask], Vf[mask], Wf[mask],
            color=colour, arrow_length_ratio=0.2, linewidth=0.8, alpha=0.8,
        )

    # Mark fixed points
    if fixed_points:
        _draw_fixed_points(ax, fixed_points, x_coupling, y_coupling, z_coupling)

    ax.set_xlabel(f"${x_coupling}$", fontsize=13)
    ax.set_ylabel(f"${y_coupling}$", fontsize=13)
    ax.set_zlabel(f"${z_coupling}$", fontsize=13)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_title("RG Flow Phase Portrait (3D)", fontsize=14)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, int] = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = 1
            unique_h.append(h)
            unique_l.append(l)
    if unique_l:
        ax.legend(unique_h, unique_l, fontsize=10)

    fig.tight_layout()
    return fig


def fixed_point_stability_3d(
    fixed_point: FixedPoint,
    stability: StabilityAnalysis,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    scale: float = 0.5,
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    """Zoomed 3D view of a single fixed point with eigenvector arrows.

    Arrow colour encodes relevance:
        * blue  -- relevant  (Re(theta) > 0)
        * orange -- irrelevant (Re(theta) < 0)

    Arrow thickness and transparency are proportional to
    |Re(theta_i)| / max|Re(theta)|, highlighting the dominant directions.

    For complex-conjugate eigenvalue pairs an arc is drawn at the tip of
    the arrow to indicate spiral flow.

    Args:
        fixed_point: The fixed point to visualise.
        stability: Stability analysis containing eigenvectors / exponents.
        x_coupling: Coupling name for the x-axis.
        y_coupling: Coupling name for the y-axis.
        z_coupling: Coupling name for the z-axis.
        scale: Length scale for the eigenvector arrows.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xc, yc, zc = _fp_coords(fixed_point, x_coupling, y_coupling, z_coupling)

    # Draw the fixed point marker
    if fixed_point.is_gaussian:
        ax.plot(
            [xc], [yc], [zc],
            marker='o', color='white', markersize=16,
            markeredgecolor='black', markeredgewidth=2,
            label='GFP', zorder=5,
        )
        fp_label = "Gaussian FP"
    else:
        ax.plot(
            [xc], [yc], [zc],
            marker='*', color='limegreen', markersize=18,
            markeredgecolor='black', markeredgewidth=1,
            label=f'NGFP ({fixed_point.relevant_directions} rel.)', zorder=5,
        )
        fp_label = f"NGFP ({fixed_point.relevant_directions} relevant)"

    # Retrieve eigenvectors and critical exponents from the stability analysis
    evecs = stability.eigenvectors
    thetas = stability.critical_exponents

    if evecs is not None and evecs.size > 0 and thetas is not None and thetas.size > 0:
        names = list(fixed_point.location.keys())
        x_idx = _coupling_index(names, x_coupling)
        y_idx = _coupling_index(names, y_coupling)
        z_idx = _coupling_index(names, z_coupling)
        n_dim = evecs.shape[0]

        # Maximum |Re(theta)| for normalising thickness/alpha
        max_re = max(abs(theta.real) for theta in thetas)
        if max_re < 1e-30:
            max_re = 1.0

        # Track which eigenvalues we have already drawn as part of a
        # complex pair so we don't draw them twice.
        drawn_as_pair: set[int] = set()

        for col in range(evecs.shape[1]):
            vec = evecs[:, col]
            theta = thetas[col]

            vx = vec[x_idx].real if 0 <= x_idx < n_dim else 0.0
            vy = vec[y_idx].real if 0 <= y_idx < n_dim else 0.0
            vz = vec[z_idx].real if 0 <= z_idx < n_dim else 0.0
            direction = np.array([vx, vy, vz])
            norm = np.linalg.norm(direction)
            if norm < 1e-14:
                continue
            direction = direction / norm * scale

            color = 'dodgerblue' if theta.real > 0 else 'darkorange'
            weight = abs(theta.real) / max_re
            lw = 1.0 + 2.5 * weight
            alpha = 0.35 + 0.65 * weight

            ax.quiver(
                xc, yc, zc,
                direction[0], direction[1], direction[2],
                color=color, arrow_length_ratio=0.15,
                linewidth=lw, alpha=alpha,
            )
            ax.quiver(
                xc, yc, zc,
                -direction[0], -direction[1], -direction[2],
                color=color, arrow_length_ratio=0.15,
                linewidth=lw, alpha=alpha * 0.5,
            )

            # Draw spiral-flow arc for complex eigenvalue pairs
            if abs(theta.imag) > 1e-10 and col not in drawn_as_pair:
                # Find the conjugate partner
                for col2 in range(col + 1, evecs.shape[1]):
                    if col2 in drawn_as_pair:
                        continue
                    if (abs(thetas[col2].real - theta.real) < 1e-8
                            and abs(thetas[col2].imag + theta.imag) < 1e-8):
                        drawn_as_pair.add(col)
                        drawn_as_pair.add(col2)
                        break

                # Draw a small circle/arc at the arrow tip to indicate spiral
                tip = np.array([xc, yc, zc]) + direction
                # Create arc points in a plane perpendicular to *direction*
                # by constructing two orthonormal vectors.
                d_hat = direction / np.linalg.norm(direction)
                # Choose a vector not parallel to d_hat
                ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(d_hat, ref)) > 0.9:
                    ref = np.array([0.0, 1.0, 0.0])
                u = np.cross(d_hat, ref)
                u = u / np.linalg.norm(u)
                v = np.cross(d_hat, u)

                arc_r = scale * 0.12
                arc_angles = np.linspace(0, 1.5 * np.pi, 30)
                arc_pts = tip[:, None] + arc_r * (
                    u[:, None] * np.cos(arc_angles)[None, :]
                    + v[:, None] * np.sin(arc_angles)[None, :]
                )
                ax.plot(
                    arc_pts[0], arc_pts[1], arc_pts[2],
                    color=color, linewidth=1.2, alpha=alpha * 0.8,
                    linestyle='--',
                )

    # Set view limits centred on the fixed point
    margin = scale * 1.5
    ax.set_xlim(xc - margin, xc + margin)
    ax.set_ylim(yc - margin, yc + margin)
    ax.set_zlim(zc - margin, zc + margin)

    ax.set_xlabel(f"${x_coupling}$", fontsize=13)
    ax.set_ylabel(f"${y_coupling}$", fontsize=13)
    ax.set_zlabel(f"${z_coupling}$", fontsize=13)

    # Build informative title
    loc_str = ", ".join(
        f"${k}^*$={v:.4f}" for k, v in fixed_point.location.items()
    )
    title_lines = [
        f"{fp_label}",
        f"Location: {loc_str}",
    ]
    if thetas is not None and thetas.size > 0:
        theta_strs = []
        for i, th in enumerate(thetas):
            if abs(th.imag) < 1e-10:
                theta_strs.append(f"{th.real:.3f}")
            else:
                theta_strs.append(f"{th.real:.3f}{th.imag:+.3f}i")
        title_lines.append(
            r"$\theta_i$ = [" + ", ".join(theta_strs) + "]"
        )
    ax.set_title("\n".join(title_lines), fontsize=12)

    fig.tight_layout()
    return fig
