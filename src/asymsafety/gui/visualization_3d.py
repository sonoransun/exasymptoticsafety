"""3D visualization of RG flows, phase portraits, and fixed-point stability.

This module provides ``mplot3d``-based plotting functions that accept the
package's domain objects (:class:`~asymsafety.beta.system.BetaFunctionSystem`,
:class:`~asymsafety.analysis.fixed_points.FixedPoint`,
:class:`~asymsafety.analysis.flow.RGTrajectory`,
:class:`~asymsafety.analysis.stability.StabilityAnalysis`) and return
:class:`matplotlib.figure.Figure` instances. It shares the colour palette,
LaTeX coupling labels, citation helpers, and proxy-artist legend builder
defined in :mod:`asymsafety.visualization.style`, so 3D figures match
the 2D companion plots in :mod:`asymsafety.visualization.phase_portrait`
and :mod:`asymsafety.visualization.conceptual`.

Physics
-------
The ``z`` axis defaults to **RG time** ``t = log(k/k_0)`` for trajectory
plots — turning the 3D figure into a (g, lambda, t) "world-line" view of
the flow. Pass an explicit ``z_coupling`` to project onto a third coupling
direction (e.g. ``"lambda_adm"`` for foliated truncations or ``"alpha"``
for quadratic gravity). Eigenvector arrows colour-code relevance:

- **Blue** = relevant (Re(theta) > 0, UV-attractive)
- **Orange** = irrelevant (Re(theta) < 0, UV-repulsive)
- **Dashed arc** = spiral flow at the tip of a complex-conjugate eigenvalue

Arrow thickness and alpha scale with ``|Re theta_i| / max|Re theta|`` so
the dominant relevant directions are visually emphasised.

References
----------
- Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] — canonical
  Einstein-Hilbert NGFP whose 3D world-line view (g, lambda, t) is
  produced by :func:`flow_trajectories_3d` with ``z_coupling=None``.
- Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274] —
  comprehensive RG flow review with 3D parameter-space figures.
- Manrique et al. (2011), Phys. Rev. Lett. 106, 251302 [1003.5129] —
  foliated truncation (g, lambda, lambda_ADM) seen in
  :func:`foliated_phase_portrait_3d`.
- See ``docs/LITERATURE.md`` for the full bibliography.

See Also
--------
:mod:`asymsafety.visualization.phase_portrait`
    2D streamplots with the same colour palette.
:mod:`asymsafety.visualization.style`
    Shared palette, LaTeX labels, citation and legend helpers.
:mod:`asymsafety.validation.reuter_1998`
    Benchmark fixed-point coordinates for Einstein-Hilbert.
:mod:`asymsafety.validation.manrique_2011`
    Benchmark fixed-point coordinates for foliated EH.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers '3d' projection

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import RGTrajectory
from asymsafety.analysis.stability import StabilityAnalysis
from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.visualization.style import (
    add_colorbar,
    add_legend_panel,
    add_reference_box,
    coupling_label,
    format_arxiv,
    theta_label,
    COLOR_NGFP,
    COLOR_GFP,
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    COLOR_SINGULARITY,
)


# Sentinel used when the third axis represents RG time rather than a coupling.
_TIME_AXIS = "__rg_time__"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coupling_index(names: list[str], coupling: str) -> int:
    """Return the index of *coupling* in *names*, or -1 if absent."""
    try:
        return names.index(coupling)
    except ValueError:
        return -1


def _resolve_z(
    z_coupling: str | None,
    x_coupling: str,
    y_coupling: str,
) -> str:
    """Decide what the third axis represents and warn on a duplicate.

    If *z_coupling* is ``None``, return the :data:`_TIME_AXIS` sentinel
    so callers know to use RG time. If it duplicates *x_coupling* or
    *y_coupling*, emit a one-time warning explaining that the silent
    zero-padded path has been removed and the axis will fall back to
    RG time instead. This guards against the legacy
    ``flow_trajectories_3d(trajs, "g", "lambda", "g")`` call pattern.
    """
    if z_coupling is None:
        return _TIME_AXIS
    if z_coupling in (x_coupling, y_coupling):
        warnings.warn(
            f"z_coupling={z_coupling!r} duplicates x or y; "
            "falling back to z = RG time t = log(k/k0). "
            "Pass z_coupling=None explicitly to silence this warning.",
            stacklevel=3,
        )
        return _TIME_AXIS
    return z_coupling


def _get_trajectory_values(
    traj: RGTrajectory,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract (x, y, z) arrays from a trajectory.

    The z axis is one of:

    - the value of the named coupling, when ``z_coupling`` matches a
      key in ``traj.coupling_values``
    - the trajectory's RG-time array ``traj.t_values`` when
      ``z_coupling == _TIME_AXIS``
    - all zeros when the coupling is absent (kept for backward
      compatibility, but :func:`_resolve_z` redirects to time first)
    """
    if traj.t_values is None or len(traj.t_values) == 0:
        return None
    n = len(traj.t_values)
    x = traj.coupling_values.get(x_coupling, np.zeros(n))
    y = traj.coupling_values.get(y_coupling, np.zeros(n))
    if z_coupling == _TIME_AXIS:
        z = np.asarray(traj.t_values, dtype=float)
    else:
        z = traj.coupling_values.get(z_coupling, np.zeros(n))
    return x, y, z


def _fp_coords(
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> tuple[float, float, float]:
    """Return (x, y, z) location of a fixed point.

    When ``z_coupling`` is the time-axis sentinel, the z coordinate is
    set to ``0`` — fixed points are at constant t in the (x, y, t) view.
    """
    z = (
        0.0 if z_coupling == _TIME_AXIS
        else fp.location.get(z_coupling, 0.0)
    )
    return (
        fp.location.get(x_coupling, 0.0),
        fp.location.get(y_coupling, 0.0),
        z,
    )


def _eigenvector_components(
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
) -> list[tuple[np.ndarray, complex]]:
    """Return list of (3-component eigenvector, critical exponent) pairs.

    Each eigenvector is projected onto the three chosen coupling axes.
    When ``z_coupling`` is the time-axis sentinel the z-component is
    zero (eigenvectors live in coupling space, not in t).
    """
    if fp.eigenvectors is None or fp.eigenvectors.size == 0:
        return []
    if fp.critical_exponents is None or fp.critical_exponents.size == 0:
        return []

    names = list(fp.location.keys())
    x_idx = _coupling_index(names, x_coupling)
    y_idx = _coupling_index(names, y_coupling)
    z_idx = (
        -1 if z_coupling == _TIME_AXIS
        else _coupling_index(names, z_coupling)
    )

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
    *,
    paper_label_for: dict[str, str] | None = None,
) -> None:
    """Mark fixed points using the project's marker conventions."""
    for fp in fixed_points:
        xv, yv, zv = _fp_coords(fp, x_coupling, y_coupling, z_coupling)
        if fp.is_gaussian:
            ax.plot(
                [xv], [yv], [zv],
                marker='o', color=COLOR_GFP, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label='GFP', zorder=5,
            )
            label_text = "GFP"
        else:
            ax.plot(
                [xv], [yv], [zv],
                marker='*', color=COLOR_NGFP, markersize=14,
                markeredgecolor='black', markeredgewidth=0.8,
                label=f'NGFP ({fp.relevant_directions} rel.)', zorder=5,
            )
            label_text = "NGFP"
        if paper_label_for and label_text in paper_label_for:
            ax.text(
                xv, yv, zv,
                f" {paper_label_for[label_text]}",
                fontsize=8, color="0.25", zorder=6,
            )


def _draw_eigenvectors(
    ax,
    fp: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    scale: float,
    *,
    show_theta_labels: bool = False,
) -> None:
    """Draw eigenvector arrows at *fp* using ``ax.quiver``.

    Blue arrows for relevant directions (Re(theta) > 0),
    orange arrows for irrelevant (Re(theta) < 0).
    Thickness and alpha scale with ``|Re theta| / max|Re theta|``.
    Optional theta labels are drawn at each arrow tip via
    :func:`asymsafety.visualization.style.theta_label`.
    """
    ev_pairs = _eigenvector_components(fp, x_coupling, y_coupling, z_coupling)
    if not ev_pairs:
        return

    xv, yv, zv = _fp_coords(fp, x_coupling, y_coupling, z_coupling)
    max_re = max(abs(theta.real) for _, theta in ev_pairs) or 1.0

    for vec, theta in ev_pairs:
        norm = np.linalg.norm(vec)
        if norm < 1e-14:
            continue
        direction = vec / norm * scale
        color = COLOR_RELEVANT if theta.real > 0 else COLOR_IRRELEVANT
        weight = abs(theta.real) / max_re
        lw = 1.0 + 2.0 * weight
        alpha = 0.45 + 0.55 * weight
        ax.quiver(
            xv, yv, zv,
            direction[0], direction[1], direction[2],
            color=color, arrow_length_ratio=0.15,
            linewidth=lw, alpha=alpha,
        )
        ax.quiver(
            xv, yv, zv,
            -direction[0], -direction[1], -direction[2],
            color=color, arrow_length_ratio=0.15,
            linewidth=lw, alpha=alpha * 0.5,
        )
        if show_theta_labels:
            tip = (
                xv + direction[0] * 1.08,
                yv + direction[1] * 1.08,
                zv + direction[2] * 1.08,
            )
            ax.text(
                *tip, theta_label(theta),
                fontsize=8, color=color, zorder=7,
            )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def flow_trajectories_3d(
    trajectories: list[RGTrajectory],
    x_coupling: str,
    y_coupling: str,
    z_coupling: str | None = None,
    fixed_points: list[FixedPoint] | None = None,
    show_eigenvectors: bool = True,
    eigenvector_scale: float = 0.3,
    *,
    show_theta_labels: bool = True,
    show_speed_widths: bool = True,
    show_references: bool = False,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    r"""Plot RG trajectories in 3D coupling (or coupling-time) space.

    Each trajectory is rendered with a colour gradient from blue (IR /
    early RG time) to red (UV / late RG time) along its length.

    Physics
    -------
    The default view (``z_coupling=None``) is the *world-line* picture:
    couplings flow on the (x, y) plane while RG time
    ``t = log(k/k_0)`` indexes the third axis. This avoids the silent
    zero-padded path that older code used for two-coupling truncations
    such as Einstein-Hilbert. Passing an explicit third coupling
    produces a true 3-coupling phase-space view, suitable for foliated
    EH ``(g, lambda, lambda_ADM)``, quadratic gravity ``(g, lambda, alpha)``,
    or matter-extended truncations.

    Parameters
    ----------
    trajectories : list of :class:`~asymsafety.analysis.flow.RGTrajectory`
        Pre-integrated RG trajectories.
    x_coupling, y_coupling : str
        Coupling names for the first two axes. Passed to
        :func:`asymsafety.visualization.style.coupling_label` for LaTeX.
    z_coupling : str or None, default ``None``
        Coupling name for the third axis. When ``None`` (the
        recommended default for 2-coupling truncations), the third
        axis is RG time ``t = log(k/k_0)``. When the same name as
        ``x_coupling`` or ``y_coupling``, a one-time warning is emitted
        and the axis falls back to RG time.
    fixed_points : list of :class:`~asymsafety.analysis.fixed_points.FixedPoint`, optional
        Fixed points to mark.
    show_eigenvectors : bool, default ``True``
        Draw bidirectional eigenvector arrows at each NGFP.
    eigenvector_scale : float, default ``0.3``
        Length scale of the eigenvector arrows in coupling-space units.
    show_theta_labels : bool, default ``True``
        Annotate each eigenvector arrow tip with the critical exponent
        ``theta`` via :func:`asymsafety.visualization.style.theta_label`.
    show_speed_widths : bool, default ``True``
        Modulate trajectory line widths by the local flow speed
        ``tanh(|beta|)`` so that fast / slow regions are visually
        distinguished.
    show_references : bool, default ``False``
        When ``True``, draw an inline reference box citing Reuter (1998)
        and Reuter & Saueressig (2012) in the lower-right corner.
    figsize : tuple, default ``(10, 8)``
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Always returns a figure even when no trajectories are supplied.

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    - See ``docs/LITERATURE.md`` for the full bibliography.

    See Also
    --------
    :func:`phase_portrait_3d`
        Vector-field view of the same coupling space.
    :func:`fixed_point_stability_3d`
        Zoomed view of a single fixed point with eigenvector encoding.
    :func:`asymsafety.visualization.phase_portrait.phase_portrait_2d`
        2D companion plot.
    :mod:`asymsafety.validation.reuter_1998`
        Benchmark fixed-point coordinates.
    """
    z_axis = _resolve_z(z_coupling, x_coupling, y_coupling)
    using_time_axis = z_axis == _TIME_AXIS

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.coolwarm

    for traj in trajectories:
        vals = _get_trajectory_values(traj, x_coupling, y_coupling, z_axis)
        if vals is None:
            continue
        xs, ys, zs = vals
        n = len(xs)
        if n < 2:
            continue
        t = np.asarray(traj.t_values, dtype=float)
        t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-30)

        widths: np.ndarray | None = None
        if show_speed_widths:
            # Approximate flow speed |beta| ~ ||d g|| / |dt| along the
            # integrated trajectory (in coupling space; the time axis
            # contribution is excluded so the encoding stays meaningful
            # in the (x, y, t) world-line view).
            dx = np.diff(xs)
            dy = np.diff(ys)
            dz = np.zeros_like(dx) if using_time_axis else np.diff(zs)
            dt = np.diff(t)
            step = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = np.where(
                    np.abs(dt) > 1e-30,
                    step / np.abs(dt),
                    0.0,
                )
            widths = 0.7 + 1.6 * np.tanh(speed)

        for k in range(n - 1):
            ax.plot(
                xs[k:k + 2], ys[k:k + 2], zs[k:k + 2],
                color=cmap(t_norm[k]),
                linewidth=float(widths[k]) if widths is not None else 1.4,
                alpha=0.85,
            )

    if fixed_points:
        paper_label_for = None
        # Heuristic: when the user is looking at the (g, lambda, t) EH view,
        # tag the NGFP with "Reuter 1998" — this is the canonical figure.
        if (
            using_time_axis
            and {x_coupling, y_coupling} == {"g", "lambda"}
        ):
            paper_label_for = {"NGFP": "Reuter 1998"}
        _draw_fixed_points(
            ax, fixed_points, x_coupling, y_coupling, z_axis,
            paper_label_for=paper_label_for,
        )
        if show_eigenvectors:
            for fp in fixed_points:
                _draw_eigenvectors(
                    ax, fp,
                    x_coupling, y_coupling, z_axis,
                    scale=eigenvector_scale,
                    show_theta_labels=show_theta_labels,
                )

    # Colorbar for RG-time gradient
    all_t = [traj.t_values for traj in trajectories
             if traj.t_values is not None and len(traj.t_values) > 0]
    if all_t:
        t_min = min(t.min() for t in all_t)
        t_max = max(t.max() for t in all_t)
        add_colorbar(
            fig, ax, mappable=None,
            label=r"$t = \log(k/k_0)$",
            vmin=t_min, vmax=t_max, cmap="coolwarm",
        )

    ax.set_xlabel(coupling_label(x_coupling), fontsize=13)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=13)
    z_label = (
        r"$t = \log(k/k_0)$" if using_time_axis
        else coupling_label(z_axis)
    )
    ax.set_zlabel(z_label, fontsize=13)
    title_suffix = (
        " — world-line view " if using_time_axis else ""
    )
    ax.set_title(f"RG Flow Trajectories (3D){title_suffix}", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    if unique_l:
        ax.legend(unique_h, unique_l, fontsize=10)

    if show_references:
        # 3D axes accept .text2D via inheritance; use the 2D-projection
        # fallback in add_reference_box instead by anchoring to the figure.
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                format_arxiv("Reuter (1998)", "hep-th/9605030"),
                format_arxiv("Reuter & Saueressig (2012)", "1202.2274"),
            ],
            loc="lower right",
        )

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
    *,
    show_singularity_planes: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    r"""Create a 3D quiver phase portrait of the RG flow.

    Beta-function vectors are evaluated on a regular grid and rendered
    as arrows coloured by the logarithmic speed ``log(1 + |beta|)``.
    Singularity / boundary planes are drawn semi-transparently when
    they appear in the chosen coupling triple.

    Physics
    -------
    The vector field ``beta(g_1, g_2, g_3)`` characterises local UV/IR
    motion in coupling space. Trajectories follow the field from IR
    (small ``k``) to UV (large ``k``); fixed points are zeros of
    ``beta``. For the Einstein-Hilbert truncation the line
    ``lambda = 1/2`` is a propagator-pole singularity; for foliated
    truncations the plane ``lambda_ADM = 1`` marks full-Diff
    restoration (Manrique 2011).

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
        The beta-function system to evaluate.
    x_coupling, y_coupling, z_coupling : str
        Coupling names for the three axes. ``z_coupling`` is required
        (vector-field plots have no natural 2D fallback). Passing the
        same name twice raises ``ValueError``.
    x_range, y_range, z_range : tuple, default see signature
        ``(min, max)`` bounds for each axis.
    n_grid : int, default ``8``
        Number of grid points per axis (cubic ``n_grid**3`` total).
    fixed_points : list of :class:`~asymsafety.analysis.fixed_points.FixedPoint`, optional
        Fixed points to mark.
    show_singularity_planes : bool, default ``True``
        Draw the ``lambda = 1/2`` and / or ``lambda_ADM = 1`` planes
        when the corresponding coupling appears among the three axes.
    figsize : tuple, default ``(10, 8)``
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Manrique et al. (2011), Phys. Rev. Lett. 106, 251302 [1003.5129] —
      ``lambda_ADM = 1`` plane.
    - See ``docs/LITERATURE.md``.

    See Also
    --------
    :func:`flow_trajectories_3d`
        Trajectory view (with a coupling or RG-time third axis).
    :func:`foliated_phase_portrait_3d`
        Specialisation to ``(g, lambda, lambda_ADM)`` with the
        full-Diff plane highlighted.
    """
    if z_coupling is None:
        raise ValueError(
            "phase_portrait_3d requires three distinct couplings; "
            "vector-field plots have no natural 2D fallback. "
            "Use flow_trajectories_3d(..., z_coupling=None) for a "
            "world-line view of a 2-coupling truncation."
        )
    if z_coupling in (x_coupling, y_coupling):
        raise ValueError(
            f"z_coupling={z_coupling!r} duplicates x_coupling or "
            "y_coupling; pick three distinct couplings."
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

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

    speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    log_speed = np.log1p(speed)
    ls_min, ls_max = log_speed.min(), log_speed.max()
    if ls_max - ls_min > 1e-30:
        c_norm = (log_speed - ls_min) / (ls_max - ls_min)
    else:
        c_norm = np.zeros_like(log_speed)

    cmap = plt.cm.coolwarm
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    Uf, Vf, Wf = U.ravel(), V.ravel(), W.ravel()
    cf = c_norm.ravel()

    max_speed = speed.max()
    if max_speed > 1e-30:
        arrow_len = (x_range[1] - x_range[0]) / n_grid * 0.4
        norm_factor = arrow_len / max_speed
        Uf = Uf * norm_factor
        Vf = Vf * norm_factor
        Wf = Wf * norm_factor

    n_bins = 16
    bins = np.linspace(0, 1, n_bins + 1)
    for b in range(n_bins):
        mask = (cf >= bins[b]) & (cf < bins[b + 1])
        if b == n_bins - 1:
            mask = mask | (cf >= bins[b + 1])
        if not np.any(mask):
            continue
        colour = cmap((bins[b] + bins[b + 1]) / 2)
        ax.quiver(
            Xf[mask], Yf[mask], Zf[mask],
            Uf[mask], Vf[mask], Wf[mask],
            color=colour, arrow_length_ratio=0.2,
            linewidth=0.8, alpha=0.8,
        )

    if fixed_points:
        _draw_fixed_points(ax, fixed_points, x_coupling, y_coupling, z_coupling)

    add_colorbar(
        fig, ax, mappable=None,
        label=r"$\log(1 + |\beta|)$",
        vmin=float(ls_min), vmax=float(ls_max), cmap="coolwarm",
    )

    if show_singularity_planes:
        _draw_singularity_planes(
            ax, x_coupling, y_coupling, z_coupling,
            x_range, y_range, z_range,
        )

    ax.set_xlabel(coupling_label(x_coupling), fontsize=13)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=13)
    ax.set_zlabel(coupling_label(z_coupling), fontsize=13)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_title("RG Flow Phase Portrait (3D)", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    if unique_l:
        ax.legend(unique_h, unique_l, fontsize=10)

    fig.tight_layout()
    return fig


def _draw_singularity_planes(
    ax,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    """Render lambda = 1/2 and lambda_ADM = 1 as semi-transparent planes.

    Only draws a plane when the corresponding coupling appears among
    the three axes and the plane value is inside the plotted box.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    couplings = (x_coupling, y_coupling, z_coupling)
    ranges = (x_range, y_range, z_range)

    def _plane_corners(axis: int, value: float) -> list[tuple[float, float, float]]:
        # Build a rectangle in the plane axis = value, spanning the
        # other two axes' ranges.
        others = [i for i in range(3) if i != axis]
        a_min, a_max = ranges[others[0]]
        b_min, b_max = ranges[others[1]]
        corners = []
        for a, b in [(a_min, b_min), (a_max, b_min),
                     (a_max, b_max), (a_min, b_max)]:
            pt = [0.0, 0.0, 0.0]
            pt[axis] = value
            pt[others[0]] = a
            pt[others[1]] = b
            corners.append(tuple(pt))
        return corners

    # lambda = 1/2 propagator-pole plane (Einstein-Hilbert / quadratic)
    for axis, name in enumerate(couplings):
        if name == "lambda" and ranges[axis][0] <= 0.5 <= ranges[axis][1]:
            corners = _plane_corners(axis, 0.5)
            poly = Poly3DCollection(
                [corners], alpha=0.10, facecolor=COLOR_SINGULARITY,
                edgecolor=COLOR_SINGULARITY, linewidths=0.8,
            )
            ax.add_collection3d(poly)
            # Label hint: place a tiny annotation near a corner
            ax.text(
                corners[0][0], corners[0][1], corners[0][2],
                r"$\lambda=1/2$",
                fontsize=8, color=COLOR_SINGULARITY,
            )

    # lambda_ADM = 1 full-Diff restoration plane (foliated truncations)
    for axis, name in enumerate(couplings):
        if name.lower() == "lambda_adm" and ranges[axis][0] <= 1.0 <= ranges[axis][1]:
            corners = _plane_corners(axis, 1.0)
            poly = Poly3DCollection(
                [corners], alpha=0.12, facecolor=COLOR_NGFP,
                edgecolor=COLOR_NGFP, linewidths=0.8,
            )
            ax.add_collection3d(poly)
            ax.text(
                corners[0][0], corners[0][1], corners[0][2],
                r"$\lambda_{\mathrm{ADM}}=1$ (full Diff)",
                fontsize=8, color="0.25",
            )


def fixed_point_stability_3d(
    fixed_point: FixedPoint,
    stability: StabilityAnalysis,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    scale: float = 0.5,
    *,
    show_theta_labels: bool = True,
    show_references: bool = False,
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    r"""Zoomed 3D view of a single fixed point with eigenvector arrows.

    Physics
    -------
    Linearised flow near a fixed point ``g*`` is governed by the
    stability matrix ``M_ij = d beta_i / d g_j``. Critical exponents
    ``theta_i = -eig(M)`` are positive (relevant, UV-attractive) or
    negative (irrelevant, UV-repulsive). Complex-conjugate pairs
    indicate spiralling flow. This view encodes:

    - **Colour** — blue for relevant, orange for irrelevant.
    - **Thickness and alpha** — proportional to
      ``|Re theta_i| / max|Re theta|``; the dominant relevant
      direction reads as the boldest arrow.
    - **Dashed arc at arrow tip** — a complex-conjugate eigenvalue
      pair, signalling spiral flow in the corresponding 2-plane.
    - **Tip label** (optional) — the numerical value of ``theta_i``
      formatted via :func:`asymsafety.visualization.style.theta_label`.

    Parameters
    ----------
    fixed_point : :class:`~asymsafety.analysis.fixed_points.FixedPoint`
        Fixed point to visualise.
    stability : :class:`~asymsafety.analysis.stability.StabilityAnalysis`
        Stability analysis (eigenvectors and critical exponents).
    x_coupling, y_coupling, z_coupling : str
        Coupling names for the three axes.
    scale : float, default ``0.5``
        Length of the eigenvector arrows in coupling-space units.
    show_theta_labels : bool, default ``True``
        Annotate each arrow tip with the numerical critical exponent.
    show_references : bool, default ``False``
        Draw an inline reference box citing Reuter (1998) and the
        relevant validation module.
    figsize : tuple, default ``(8, 8)``
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040]
    - Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810] — review
      of relevance / irrelevance and predictivity in asymptotic safety.

    See Also
    --------
    :func:`asymsafety.analysis.stability.analyze_stability`
        Computes the :class:`StabilityAnalysis` consumed here.
    :func:`asymsafety.visualization.fixed_point_plot.plot_critical_exponents`
        2D plot of ``theta_i`` along a continuation parameter.
    """
    if z_coupling in (x_coupling, y_coupling):
        raise ValueError(
            f"z_coupling={z_coupling!r} duplicates x_coupling or "
            "y_coupling; pick three distinct couplings."
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xc, yc, zc = _fp_coords(fixed_point, x_coupling, y_coupling, z_coupling)

    if fixed_point.is_gaussian:
        ax.plot(
            [xc], [yc], [zc],
            marker='o', color=COLOR_GFP, markersize=16,
            markeredgecolor='black', markeredgewidth=2,
            zorder=5,
        )
        fp_label = "Gaussian FP"
    else:
        ax.plot(
            [xc], [yc], [zc],
            marker='*', color=COLOR_NGFP, markersize=18,
            markeredgecolor='black', markeredgewidth=1,
            zorder=5,
        )
        fp_label = f"NGFP ({fixed_point.relevant_directions} relevant)"

    evecs = stability.eigenvectors
    thetas = stability.critical_exponents

    if (
        evecs is not None and evecs.size > 0
        and thetas is not None and thetas.size > 0
    ):
        names = list(fixed_point.location.keys())
        x_idx = _coupling_index(names, x_coupling)
        y_idx = _coupling_index(names, y_coupling)
        z_idx = _coupling_index(names, z_coupling)
        n_dim = evecs.shape[0]

        max_re = max(abs(theta.real) for theta in thetas)
        if max_re < 1e-30:
            max_re = 1.0

        drawn_as_pair: set[int] = set()
        has_complex_pair = False

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

            color = COLOR_RELEVANT if theta.real > 0 else COLOR_IRRELEVANT
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

            if show_theta_labels:
                tip = (
                    xc + direction[0] * 1.10,
                    yc + direction[1] * 1.10,
                    zc + direction[2] * 1.10,
                )
                ax.text(
                    *tip, theta_label(theta),
                    fontsize=8, color=color, zorder=7,
                )

            if abs(theta.imag) > 1e-10 and col not in drawn_as_pair:
                has_complex_pair = True
                for col2 in range(col + 1, evecs.shape[1]):
                    if col2 in drawn_as_pair:
                        continue
                    if (
                        abs(thetas[col2].real - theta.real) < 1e-8
                        and abs(thetas[col2].imag + theta.imag) < 1e-8
                    ):
                        drawn_as_pair.add(col)
                        drawn_as_pair.add(col2)
                        break

                tip = np.array([xc, yc, zc]) + direction
                d_hat = direction / np.linalg.norm(direction)
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

        # Build a structured legend explaining all visual encodings.
        legend_entries: list[tuple[str, str]] = [
            (
                "marker_star" if not fixed_point.is_gaussian
                else "marker_circle",
                fp_label,
            ),
            (f"line|{COLOR_RELEVANT}", r"relevant ($\mathrm{Re}\,\theta>0$)"),
            (f"line|{COLOR_IRRELEVANT}", r"irrelevant ($\mathrm{Re}\,\theta<0$)"),
        ]
        if has_complex_pair:
            legend_entries.append(
                (f"line:dashed|{COLOR_RELEVANT}",
                 r"spiral pair (complex $\theta$)")
            )
        legend_entries.append(("line|0.55", "thickness ∝ |Re θ|"))
        add_legend_panel(ax, legend_entries, loc="upper left", fontsize=9)

    margin = scale * 1.5
    ax.set_xlim(xc - margin, xc + margin)
    ax.set_ylim(yc - margin, yc + margin)
    ax.set_zlim(zc - margin, zc + margin)

    ax.set_xlabel(coupling_label(x_coupling), fontsize=13)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=13)
    ax.set_zlabel(coupling_label(z_coupling), fontsize=13)

    loc_str = ", ".join(
        f"${k}^*$={v:.4f}" for k, v in fixed_point.location.items()
    )
    title_lines = [fp_label, f"Location: {loc_str}"]
    if thetas is not None and thetas.size > 0:
        title_lines.append(
            r"$\theta_i$ = ["
            + ", ".join(
                f"{th.real:.3f}" if abs(th.imag) < 1e-10
                else f"{th.real:.3f}{th.imag:+.3f}i"
                for th in thetas
            )
            + "]"
        )
    ax.set_title("\n".join(title_lines), fontsize=12)

    if show_references:
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                format_arxiv("Reuter (1998)", "hep-th/9605030"),
                format_arxiv("Lauscher & Reuter (2002)", "hep-th/0108040"),
            ],
            loc="lower right",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Foliated 3D phase portrait (g, lambda, lambda_ADM)
# ---------------------------------------------------------------------------


def foliated_phase_portrait_3d(
    system: BetaFunctionSystem,
    *,
    x_range: tuple[float, float] = (0.0, 1.6),
    y_range: tuple[float, float] = (-0.3, 0.5),
    z_range: tuple[float, float] = (0.5, 1.4),
    n_grid: int = 6,
    fixed_points: list[FixedPoint] | None = None,
    show_references: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    r"""3D phase portrait of the foliated Einstein-Hilbert truncation.

    Specialisation of :func:`phase_portrait_3d` to the
    ``(g, lambda, lambda_ADM)`` coupling triple of the foliated EH
    truncation on ``S^1 x S^3`` (Manrique 2011). Marks the
    ``lambda_ADM = 1`` plane that signals full-Diff restoration at the
    NGFP and embeds an inline citation box.

    Physics
    -------
    The ADM decomposition splits the metric into lapse, shift, and
    spatial part; the additional coupling ``lambda_ADM`` measures
    foliation anisotropy. At the NGFP ``lambda_ADM -> 1``, recovering
    the covariant (full-Diff) limit, in agreement with Manrique 2011
    and confirmed under Wick rotation by Saueressig 2025.

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
        A foliated-EH beta-function system, typically built via
        :func:`asymsafety.beta.foliated.build_foliated_eh_beta_system`.
    x_range, y_range, z_range : tuple
        Plotting bounds for ``g``, ``lambda``, ``lambda_adm``.
    n_grid : int, default ``6``
        Grid points per axis.
    fixed_points : list, optional
        Fixed points to mark.
    show_references : bool, default ``True``
        Embed an inline reference box.
    figsize : tuple, default ``(10, 8)``
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Manrique, Rechenberger & Saueressig (2011),
      Phys. Rev. Lett. 106, 251302 [1003.5129]
    - Biemans et al. (2017), JHEP 05, 093 [1609.02803]
    - Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752]

    See Also
    --------
    :mod:`asymsafety.validation.manrique_2011`
        Benchmark fixed-point coordinates ``g* ≈ 0.96, lambda* ≈ 0.20,
        lambda_ADM* = 1.0``.
    :func:`asymsafety.beta.foliated.build_foliated_eh_beta_system`
        Builder for the foliated beta-function system.
    """
    fig = phase_portrait_3d(
        system,
        x_coupling="g", y_coupling="lambda", z_coupling="lambda_adm",
        x_range=x_range, y_range=y_range, z_range=z_range,
        n_grid=n_grid, fixed_points=fixed_points,
        show_singularity_planes=True, figsize=figsize,
    )
    ax = fig.axes[0]
    ax.set_title(
        r"Foliated EH RG Flow $(g, \lambda, \lambda_{\mathrm{ADM}})$",
        fontsize=14,
    )
    if show_references:
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                format_arxiv("Manrique et al. (2011)", "1003.5129"),
                format_arxiv("Biemans et al. (2017)", "1609.02803"),
                format_arxiv("Saueressig et al. (2025)", "2501.03752"),
            ],
            loc="lower right",
        )
    return fig


# ---------------------------------------------------------------------------
# Basin of attraction (Monte-Carlo IC sphere around the NGFP)
# ---------------------------------------------------------------------------


def flow_basin_3d(
    system: BetaFunctionSystem,
    fixed_point: FixedPoint,
    x_coupling: str,
    y_coupling: str,
    z_coupling: str,
    *,
    n_samples: int = 240,
    radius: float = 0.25,
    t_span: tuple[float, float] = (0.0, 6.0),
    escape_threshold: float = 2.0,
    seed: int = 0,
    show_references: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    r"""Monte-Carlo visualisation of the basin of attraction of a fixed point.

    Samples ``n_samples`` points on a sphere of radius ``radius`` around
    *fixed_point*, integrates the flow forward (toward the IR) for
    ``t_span``, and colour-codes whether the trajectory remained near
    the fixed point ("captured") or escaped beyond ``escape_threshold``
    ("escaped"). The result approximates the local basin of attraction
    in the ``(x, y, z)`` projection.

    Physics
    -------
    The UV-critical surface — the set of trajectories that hit the NGFP
    in the UV — has codimension equal to the number of irrelevant
    directions. Forward (IR-directed) integration starting *near* the
    NGFP traces out the IR flow leaving the surface; whether or not a
    trajectory escapes depends on which direction it was perturbed in.
    Captured (blue) and escaped (orange) clouds together delineate the
    local basin geometry.

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
    fixed_point : :class:`~asymsafety.analysis.fixed_points.FixedPoint`
    x_coupling, y_coupling, z_coupling : str
    n_samples : int, default ``240``
    radius : float, default ``0.25``
    t_span : tuple, default ``(0.0, 6.0)``
    escape_threshold : float, default ``2.0``
    seed : int, default ``0``
    show_references : bool, default ``True``
    figsize : tuple, default ``(10, 8)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    - Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810]

    See Also
    --------
    :func:`fixed_point_stability_3d`
        Local linearised stability — the analytic counterpart.
    """
    from asymsafety.analysis.flow import FlowIntegrator

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xc, yc, zc = _fp_coords(fixed_point, x_coupling, y_coupling, z_coupling)

    rng = np.random.default_rng(seed)
    # Random points on a unit sphere in the 3 chosen axes
    pts = rng.normal(size=(n_samples, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-30
    pts *= radius
    pts[:, 0] += xc
    pts[:, 1] += yc
    pts[:, 2] += zc

    integrator = FlowIntegrator(system)

    captured: list[tuple[float, float, float]] = []
    escaped: list[tuple[float, float, float]] = []

    for px, py, pz in pts:
        ic: dict[str, float] = dict(fixed_point.location)
        ic[x_coupling] = float(px)
        ic[y_coupling] = float(py)
        if z_coupling in fixed_point.location:
            ic[z_coupling] = float(pz)
        try:
            traj = integrator.integrate(ic, t_span=t_span)
        except Exception:
            escaped.append((px, py, pz))
            continue
        # Distance from FP at the end of the trajectory
        try:
            xs = traj.coupling_values.get(x_coupling)
            ys = traj.coupling_values.get(y_coupling)
            zs = (
                traj.coupling_values.get(z_coupling)
                if z_coupling in fixed_point.location
                else np.full_like(xs, pz)
            )
            d_end = np.sqrt(
                (xs[-1] - xc) ** 2
                + (ys[-1] - yc) ** 2
                + (zs[-1] - zc) ** 2
            )
        except Exception:
            d_end = np.inf
        if d_end > escape_threshold:
            escaped.append((px, py, pz))
        else:
            captured.append((px, py, pz))

    if captured:
        cap = np.asarray(captured)
        ax.scatter(
            cap[:, 0], cap[:, 1], cap[:, 2],
            c=COLOR_RELEVANT, s=14, alpha=0.55, label="captured",
            edgecolor="none",
        )
    if escaped:
        esc = np.asarray(escaped)
        ax.scatter(
            esc[:, 0], esc[:, 1], esc[:, 2],
            c=COLOR_IRRELEVANT, s=14, alpha=0.45, label="escaped",
            edgecolor="none",
        )

    ax.plot(
        [xc], [yc], [zc],
        marker='*', color=COLOR_NGFP, markersize=16,
        markeredgecolor='black', markeredgewidth=0.8,
        label="NGFP", zorder=5,
    )

    ax.set_xlabel(coupling_label(x_coupling), fontsize=13)
    ax.set_ylabel(coupling_label(y_coupling), fontsize=13)
    ax.set_zlabel(coupling_label(z_coupling), fontsize=13)
    ax.set_title(
        f"Basin of attraction near NGFP "
        f"({len(captured)}/{n_samples} captured)",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    if show_references:
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                format_arxiv("Reuter & Saueressig (2012)", "1202.2274"),
                format_arxiv("Bonanno et al. (2020)", "2004.06810"),
            ],
            loc="lower right",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quantum / cosmology 3D extensions
# ---------------------------------------------------------------------------


def vqrg_cost_isosurface_3d(
    cost,
    base_params: np.ndarray,
    *,
    dims: tuple[int, int, int] = (0, 1, 2),
    p_range: tuple[float, float] = (-np.pi, np.pi),
    n_grid: int = 24,
    isolevels: Sequence[float] = (0.05, 0.2, 0.8),
    show_references: bool = True,
) -> Figure:
    """3D isosurfaces of a VQRG cost-function slice over three parameters.

    Renders translucent iso-cost surfaces around an NGFP. Uses
    :func:`skimage.measure.marching_cubes` when scikit-image is
    installed; otherwise falls back to a stack of 2D contour slices
    along the first chosen axis (visually comparable, no extra
    dependency).

    Args:
        cost: A :class:`~asymsafety.quantum.vqrg.cost.VQRGCostFunction`
            instance.
        base_params: Reference parameter vector. Copies are taken before
            mutation; the input is not modified.
        dims: Triple of parameter indices to sweep.
        p_range: ``(p_lo, p_hi)`` sweep range, applied to all three axes.
        n_grid: Grid resolution per axis (cubic — total
            ``n_grid**3`` evaluations).
        isolevels: Iso-cost levels, in increasing order. Innermost
            surface (smallest level) is most transparent.
        show_references: Whether to add the citation box.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    grid = np.linspace(p_range[0], p_range[1], n_grid)
    P0, P1, P2 = np.meshgrid(grid, grid, grid, indexing="ij")
    cost_grid = np.empty_like(P0)
    params = base_params.astype(float).copy()
    i0, i1, i2 = dims
    for a in range(n_grid):
        for b in range(n_grid):
            for c in range(n_grid):
                params[i0] = P0[a, b, c]
                params[i1] = P1[a, b, c]
                params[i2] = P2[a, b, c]
                cost_grid[a, b, c] = float(cost.evaluate(params))

    cost_min = float(cost_grid.min())
    cost_max = float(cost_grid.max())
    norm = lambda v: cost_min + v * (cost_max - cost_min)

    rendered = False
    try:
        from skimage.measure import marching_cubes  # type: ignore
    except ImportError:
        marching_cubes = None  # noqa: N816

    if marching_cubes is not None:
        spacing = (
            (p_range[1] - p_range[0]) / max(1, n_grid - 1),
            (p_range[1] - p_range[0]) / max(1, n_grid - 1),
            (p_range[1] - p_range[0]) / max(1, n_grid - 1),
        )
        for k, level_frac in enumerate(isolevels):
            level = norm(level_frac)
            if not (cost_min < level < cost_max):
                continue
            try:
                verts, faces, _, _ = marching_cubes(
                    cost_grid, level=level, spacing=spacing,
                )
            except (RuntimeError, ValueError):
                continue
            verts = verts + np.array([p_range[0], p_range[0], p_range[0]])
            tri = ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                color=COLOR_RELEVANT,
                alpha=0.18 + 0.4 * (1.0 - k / max(1, len(isolevels) - 1)),
                edgecolor="none",
                linewidth=0,
            )
            tri.set_label(rf"$C={level:.3g}$")
            rendered = True

    if not rendered:
        # Fallback: contour stack along axis 0
        n_slices = min(8, n_grid)
        slice_idx = np.linspace(0, n_grid - 1, n_slices, dtype=int)
        for s in slice_idx:
            cs_x = grid[s] * np.ones_like(P1[s])
            cs = ax.contour(
                cs_x, P1[s], P2[s],
                cost_grid[s],
                levels=[norm(lvl) for lvl in isolevels],
                cmap="coolwarm", alpha=0.7,
            )
            del cs

    # Mark the slice minimum (3D coordinates)
    flat = int(np.argmin(cost_grid))
    a, b, c = np.unravel_index(flat, cost_grid.shape)
    ax.scatter(
        [P0[a, b, c]], [P1[a, b, c]], [P2[a, b, c]],
        marker="*", s=180, color=COLOR_NGFP,
        edgecolor="black", linewidth=0.6, zorder=10,
        label="slice minimum",
    )

    ax.set_xlabel(rf"$\theta_{{{i0}}}$")
    ax.set_ylabel(rf"$\theta_{{{i1}}}$")
    ax.set_zlabel(rf"$\theta_{{{i2}}}$")
    ax.set_title("VQRG cost isosurfaces")
    ax.legend(fontsize=9, loc="upper left")

    if show_references:
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                "Cerezo et al. (2021) [2012.09265]",
                "Stokes et al. (2020) [1909.02108]",
            ],
            loc="lower right",
        )

    fig.tight_layout()
    return fig


def koopman_spectrum_3d(
    edmd_result: dict,
    *,
    show_references: bool = True,
) -> Figure:
    """3D scatter of Koopman eigenvalues lifted to ``(Re mu, Im mu, |mu|)``.

    Useful when many decay rates collapse on top of one another in the
    flat 2D spectrum plot. Adds a translucent unit-cylinder mesh as
    the stability boundary.

    Args:
        edmd_result: Result of
            :meth:`~asymsafety.quantum.koopman.operator.KoopmanOperator.compute_edmd`.
        show_references: Whether to add the citation box.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    eigenvalues = np.atleast_1d(edmd_result.get("eigenvalues", []))
    modes = edmd_result.get("modes", np.zeros((0, 0)))
    if eigenvalues.size == 0:
        ax.text(0.5, 0.5, 0.5, "no eigenvalues",
                transform=ax.transAxes)
        return fig

    if modes.size > 0:
        n_take = min(eigenvalues.size, modes.shape[1])
        weights = np.linalg.norm(modes[:, :n_take], axis=0)
    else:
        n_take = eigenvalues.size
        weights = np.ones(n_take)
    eigenvalues = eigenvalues[:n_take]
    weights = weights / max(weights.max(), 1e-30)
    sizes = 30.0 + 230.0 * weights

    re_vals = eigenvalues.real
    im_vals = eigenvalues.imag
    abs_vals = np.abs(eigenvalues)

    inside = abs_vals < 1.0
    ax.scatter(
        re_vals[inside], im_vals[inside], abs_vals[inside],
        s=sizes[inside], c=COLOR_IRRELEVANT, alpha=0.85,
        edgecolor="black", linewidth=0.5, label="decaying",
    )
    ax.scatter(
        re_vals[~inside], im_vals[~inside], abs_vals[~inside],
        s=sizes[~inside], c=COLOR_RELEVANT, alpha=0.85,
        edgecolor="black", linewidth=0.5, label="growing",
    )

    # Unit cylinder
    theta = np.linspace(0, 2 * np.pi, 60)
    z_cyl = np.linspace(0, max(1.0, abs_vals.max() * 1.1), 12)
    Theta, Zc = np.meshgrid(theta, z_cyl)
    Xc = np.cos(Theta)
    Yc = np.sin(Theta)
    ax.plot_surface(
        Xc, Yc, Zc,
        color=COLOR_SINGULARITY, alpha=0.12, edgecolor="none",
    )

    ax.set_xlabel(r"$\mathrm{Re}(\mu)$")
    ax.set_ylabel(r"$\mathrm{Im}(\mu)$")
    ax.set_zlabel(r"$|\mu|$")
    ax.set_title("Koopman spectrum (3D)")
    ax.legend(fontsize=9, loc="upper left")

    if show_references:
        fig_ax = fig.add_axes([0.02, 0.02, 0.001, 0.001])
        fig_ax.set_axis_off()
        add_reference_box(
            fig_ax,
            [
                "Williams, Kevrekidis & Rowley (2015) [1408.4408]",
                "Mezic (2020) [1906.07401]",
            ],
            loc="lower right",
        )

    fig.tight_layout()
    return fig


def quadratic_stability_3d(
    fixed_point: FixedPoint,
    stability: StabilityAnalysis,
    *,
    x_coupling: str = "g",
    y_coupling: str = "lambda",
    z_coupling: str = "alpha",
    scale: float = 0.4,
    show_theta_labels: bool = True,
    show_references: bool = True,
) -> Figure:
    """3D eigenvector view of a quadratic-gravity NGFP.

    Thin wrapper around :func:`fixed_point_stability_3d` that picks
    sensible default axes for the four-coupling ``(g, lambda, alpha,
    beta)`` quadratic truncation. Use ``z_coupling="beta"`` to switch
    the projection axis.

    Args:
        fixed_point: NGFP returned by :class:`FixedPointFinder`.
        stability: Companion :class:`StabilityAnalysis`.
        x_coupling, y_coupling, z_coupling: Coupling axes.
        scale: Eigenvector arrow length scaling.
        show_theta_labels: Annotate arrows with critical exponents.
        show_references: Add the citation box.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    fig = fixed_point_stability_3d(
        fixed_point, stability,
        x_coupling, y_coupling, z_coupling,
        scale=scale,
        show_theta_labels=show_theta_labels,
        show_references=show_references,
    )
    if show_references and fig.axes:
        # The stability_3d already drew a reference box; we override the
        # fig title to mention quadratic gravity for context.
        fig.axes[0].set_title("Quadratic gravity NGFP — eigenvector view")
    return fig
