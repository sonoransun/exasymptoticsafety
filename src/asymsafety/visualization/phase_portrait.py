"""Phase portrait visualization for RG flows.

Produces 2D streamplots showing the RG flow in coupling space, with
fixed points marked, separatrices highlighted, and inline literature
provenance for the canonical Einstein-Hilbert truncation.

Public API
----------
- :func:`phase_portrait_2d` — generic streamplot for any 2-coupling slice.
- :func:`flow_diagram` — running couplings vs RG scale ``t``.
- :func:`annotated_eh_phase_portrait` — publication-grade EH figure with
  fixed points, eigenvector arrows labelled by critical exponents,
  back-integrated separatrix, and a citation footer.
- :func:`separatrix_overlay` — adds a back-integrated UV-critical-surface
  separatrix to an existing phase-portrait axes (or builds one).
- :func:`regulator_comparison_panels` — side-by-side phase portraits for
  Litim and Exponential regulators with NGFP-drift summary.
- :func:`quadratic_pairwise_grid` — 2x3 grid projecting the quadratic
  gravity ``(g, lambda, alpha, beta)`` flow onto the 6 pairwise planes.

References
----------
- Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
- Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040]
- Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195]
- Codello et al. (2009), Ann. Phys. 324, 414 [0812.0785]
- Wetterich (1993), Phys. Lett. B 301, 90
- See ``docs/LITERATURE.md`` for the full bibliography.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import FlowIntegrator, RGTrajectory
from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.visualization.style import (
    add_colorbar,
    add_reference_box,
    annotate_fixed_point,
    apply_style,
    coupling_label,
    format_arxiv,
    theta_label,
    COLOR_NGFP,
    COLOR_GFP,
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    COLOR_SEPARATRIX,
    COLOR_SINGULARITY,
    COLOR_TRAJECTORY,
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
    r"""Create a 2D phase portrait of the RG flow on a coupling slice.

    Physics
    -------
    Renders the beta-function vector field as a streamplot, colouring
    streamlines by ``log(1 + |beta|)`` so high-flow regions stand out.
    For Einstein-Hilbert, the line ``lambda = 1/2`` (a propagator-pole
    singularity) is automatically drawn as a dashed boundary when
    ``y_coupling == "lambda"`` and the line lies inside ``y_range``.
    Fixed points are marked using
    :func:`asymsafety.visualization.style.annotate_fixed_point`.

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
        Beta function system.
    x_coupling, y_coupling : str
        Coupling names for the two axes.
    x_range, y_range : tuple, default see signature
        ``(min, max)`` plotting bounds.
    n_grid : int, default ``25``
        Streamplot grid resolution.
    fixed_points : list, optional
        Fixed points to mark.
    trajectories : list, optional
        Pre-integrated trajectories to overlay.
    title : str, optional
        Plot title (default ``"RG Flow Phase Portrait"``).
    figsize : tuple, default ``(8, 6)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195]
    - See ``docs/LITERATURE.md`` for the full bibliography.

    See Also
    --------
    :func:`annotated_eh_phase_portrait`
        Self-contained, publication-grade Einstein-Hilbert variant
        with eigenvectors, separatrix and citation footer.
    :func:`asymsafety.gui.visualization_3d.flow_trajectories_3d`
        3D world-line view.
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
            label=r"$\lambda = 1/2$ singularity",
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
    r"""Plot running couplings as a function of RG scale ``t = log(k/k_0)``.

    Physics
    -------
    Each coupling ``g_i(k)`` solves ``dg_i/dt = beta_i(g)``. This figure
    shows multiple solution curves on a shared time axis, with IR
    (small ``k``) on the left and UV (large ``k``) on the right.
    Asymptotic safety is the statement that all curves approach finite
    values ``g_i^*`` as ``t -> infty``.

    Parameters
    ----------
    trajectories : list of :class:`~asymsafety.analysis.flow.RGTrajectory`
    coupling_names : list of str, optional
        Subset of couplings to show. Defaults to every coupling
        present in the trajectories.
    title : str, default ``"Running Couplings"``
    figsize : tuple, default ``(10, 6)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Wetterich (1993), Phys. Lett. B 301, 90 — exact RG flow equation.
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] — first
      Einstein-Hilbert NGFP.

    See Also
    --------
    :func:`asymsafety.gui.visualization_3d.flow_trajectories_3d`
        3D world-line view of the same trajectories.
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
    *,
    show_separatrix: bool = True,
    show_references: bool = True,
) -> Figure:
    r"""Publication-grade annotated phase portrait of Einstein-Hilbert RG flow.

    Physics
    -------
    The Einstein-Hilbert truncation ``Gamma_k = (16 pi G)^{-1} integral
    sqrt(g) (R - 2 Lambda)`` produces a 2-coupling beta-function system
    in the dimensionless variables ``g = G k^2`` and
    ``lambda = Lambda k^{-2}``. With the optimised Litim regulator
    (Litim 2001), Reuter (1998) found a non-Gaussian UV fixed point at
    approximately ``g* ≈ 0.707, lambda* ≈ 0.193`` with two
    UV-attractive (relevant) directions, giving a finite-parameter UV
    completion of quantum gravity. This figure overlays:

    - The streamplot of the beta-function vector field, coloured by
      ``log(1 + |beta|)``.
    - The Gaussian (perturbative) FP and the Reuter NGFP, both
      labelled with their reference paper.
    - Eigenvector arrows at the NGFP, blue for relevant
      (``Re theta > 0``) and orange for irrelevant; annotated with the
      numerical value of ``theta_i`` via
      :func:`asymsafety.visualization.style.theta_label`.
    - The propagator-pole boundary ``lambda = 1/2``.
    - Optionally, the back-integrated separatrix of the relevant
      eigendirection (the local UV-critical-surface trace).
    - A citation footer linking to Reuter (1998), Lauscher & Reuter
      (2002), and Litim (2001).

    Parameters
    ----------
    figsize : tuple, default ``(10, 8)``
    show_separatrix : bool, default ``True``
        Back-integrate the relevant eigendirection from the NGFP and
        draw the resulting separatrix curve. Disable for cleaner views.
    show_references : bool, default ``True``
        Include the corner citation box.

    Returns
    -------
    matplotlib.figure.Figure
        Always returns a figure, even when individual analysis steps
        (FP finder, stability, integrator) fail. Failures are silent.

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040]
    - Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195]
    - Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    - See ``docs/LITERATURE.md`` for the full bibliography.

    See Also
    --------
    :func:`phase_portrait_2d`
        Generic streamplot version (no inline citations).
    :func:`asymsafety.gui.visualization_3d.flow_trajectories_3d`
        3D world-line view.
    :func:`asymsafety.gui.visualization_3d.fixed_point_stability_3d`
        Local zoom on a fixed point with eigenvector arrows.
    :mod:`asymsafety.validation.reuter_1998`
        Benchmark fixed-point coordinates ``g* ≈ 0.707, lambda* ≈ 0.193``.
    """
    from asymsafety.beta.einstein_hilbert import build_eh_beta_system
    from asymsafety.analysis.fixed_points import FixedPointFinder
    from asymsafety.analysis.stability import analyze_stability
    from asymsafety.visualization.style import CMAP_FLOW

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
    # 6. Draw fixed points (with paper provenance for the NGFP)
    # ------------------------------------------------------------------
    if gfp is not None:
        annotate_fixed_point(ax, gfp, x_coupling, y_coupling)
    if ngfp is not None:
        annotate_fixed_point(
            ax, ngfp, x_coupling, y_coupling,
            paper_label="Reuter 1998",
        )

    # ------------------------------------------------------------------
    # 7. Eigenvector arrows at the NGFP, with theta values inline
    # ------------------------------------------------------------------
    relevant_eigvec_for_separatrix: np.ndarray | None = None
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

            if (
                is_relevant
                and relevant_eigvec_for_separatrix is None
            ):
                relevant_eigvec_for_separatrix = np.array(
                    [evec[x_idx] / norm, evec[y_idx] / norm]
                )

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
            # Inline theta value near the positive arrow tip
            relevance_word = "rel." if is_relevant else "irrel."
            ax.text(
                fp_g + dx * 1.15, fp_lam + dy * 1.15,
                f"{theta_label(theta)}\n({relevance_word})",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white", ec="0.85", alpha=0.85,
                ),
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
            label=r"$\lambda = 1/2$ singularity",
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
    # 11. UV-critical-surface separatrix (back-integrate the relevant
    #     eigendirection from the NGFP)
    # ------------------------------------------------------------------
    if (
        show_separatrix
        and ngfp is not None
        and relevant_eigvec_for_separatrix is not None
    ):
        eps = 0.01
        v = relevant_eigvec_for_separatrix
        for sign in (+1.0, -1.0):
            try:
                ic = {
                    "g": ngfp.location["g"] + sign * eps * v[0],
                    "lambda": ngfp.location["lambda"] + sign * eps * v[1],
                }
                sep_traj = integrator.integrate(ic, t_span=(-12, 0))
                sep_g = sep_traj.coupling_values.get("g")
                sep_lam = sep_traj.coupling_values.get("lambda")
                if sep_g is None or sep_lam is None:
                    continue
                ax.plot(
                    sep_g, sep_lam,
                    color=COLOR_SEPARATRIX, lw=2.0, alpha=0.7,
                    label="UV-critical separatrix" if sign > 0 else None,
                    zorder=4,
                )
            except Exception:
                continue
        # Update legend so the separatrix entry appears
        h, l = ax.get_legend_handles_labels()
        seen: set[str] = set()
        h2, l2 = [], []
        for hh, ll in zip(h, l):
            if ll in seen:
                continue
            seen.add(ll)
            h2.append(hh)
            l2.append(ll)
        ax.legend(h2, l2, loc="upper left", fontsize=10)

    # ------------------------------------------------------------------
    # 12. Reference box
    # ------------------------------------------------------------------
    if show_references:
        add_reference_box(
            ax,
            [
                format_arxiv("Reuter (1998)", "hep-th/9605030"),
                format_arxiv("Lauscher & Reuter (2002)", "hep-th/0108040"),
                format_arxiv("Litim (2001)", "hep-th/0103195"),
            ],
            loc="lower right",
        )

    # ------------------------------------------------------------------
    # 13. Finalize
    # ------------------------------------------------------------------
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Separatrix overlay
# ---------------------------------------------------------------------------


def separatrix_overlay(
    system: BetaFunctionSystem,
    fixed_point: FixedPoint,
    x_coupling: str = "g",
    y_coupling: str = "lambda",
    *,
    epsilon: float = 0.01,
    t_span: tuple[float, float] = (-12.0, 0.0),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    r"""Add (or build) a UV-critical-surface separatrix overlay on a phase portrait.

    Physics
    -------
    The UV-critical surface at a fixed point is the set of trajectories
    that hit the FP in the UV; in two dimensions it is one-dimensional
    and locally tangent to the relevant eigenvector(s). This function
    seeds two trajectories at ``fp +/- epsilon * v_rel`` and integrates
    backwards in RG time, producing a curve that traces the local
    separatrix outward from the NGFP. The result delineates the UV-
    attractive basin from the irrelevant escape directions.

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
    fixed_point : :class:`~asymsafety.analysis.fixed_points.FixedPoint`
    x_coupling, y_coupling : str, default ``"g", "lambda"``
    epsilon : float, default ``0.01``
        Initial offset from the FP along the relevant eigendirection.
    t_span : tuple, default ``(-12.0, 0.0)``
        RG-time integration interval (negative = backward = toward IR).
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to draw on; a new figure is created when ``None``.
    figsize : tuple
        Figure size used only when ``ax`` is ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    - Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810]

    See Also
    --------
    :func:`annotated_eh_phase_portrait`
        Already includes a separatrix overlay (toggle via
        ``show_separatrix=True``).
    :func:`asymsafety.gui.visualization_3d.flow_basin_3d`
        Monte-Carlo basin visualisation in 3D.
    """
    from asymsafety.analysis.stability import analyze_stability

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    try:
        sa = analyze_stability(system, fixed_point)
    except Exception:
        sa = None

    if sa is None or sa.eigenvectors.size == 0:
        return fig

    names = system.coupling_names
    try:
        x_idx = names.index(x_coupling)
        y_idx = names.index(y_coupling)
    except ValueError:
        return fig

    integrator = FlowIntegrator(system)

    # Use the dominant relevant eigendirection (largest Re(theta) > 0).
    rel_cols = [
        c for c in range(sa.eigenvectors.shape[1])
        if sa.critical_exponents[c].real > 0
    ]
    if not rel_cols:
        return fig
    col = max(rel_cols, key=lambda c: sa.critical_exponents[c].real)
    evec = sa.eigenvectors[:, col].real
    norm = np.sqrt(evec[x_idx] ** 2 + evec[y_idx] ** 2)
    if norm < 1e-12:
        return fig
    v = np.array([evec[x_idx] / norm, evec[y_idx] / norm])

    fp_x = fixed_point.location.get(x_coupling, 0.0)
    fp_y = fixed_point.location.get(y_coupling, 0.0)

    drew_one = False
    for sign in (+1.0, -1.0):
        ic = dict(fixed_point.location)
        ic[x_coupling] = fp_x + sign * epsilon * v[0]
        ic[y_coupling] = fp_y + sign * epsilon * v[1]
        try:
            traj = integrator.integrate(ic, t_span=t_span)
        except Exception:
            continue
        sx = traj.coupling_values.get(x_coupling)
        sy = traj.coupling_values.get(y_coupling)
        if sx is None or sy is None:
            continue
        ax.plot(
            sx, sy,
            color=COLOR_SEPARATRIX, lw=2.0, alpha=0.75,
            label="UV-critical separatrix" if not drew_one else None,
            zorder=4,
        )
        drew_one = True

    if drew_one:
        ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regulator comparison panels
# ---------------------------------------------------------------------------


def regulator_comparison_panels(
    system_litim: BetaFunctionSystem,
    system_exponential: BetaFunctionSystem,
    *,
    x_coupling: str = "g",
    y_coupling: str = "lambda",
    x_range: tuple[float, float] = (0.0, 1.5),
    y_range: tuple[float, float] = (-0.3, 0.45),
    n_grid: int = 30,
    fp_guess: dict[str, float] | None = None,
    figsize: tuple[float, float] = (15, 5),
) -> Figure:
    r"""Side-by-side phase portraits for Litim vs Exponential regulators.

    Three-panel comparison: (left) Litim streamplot, (centre)
    Exponential streamplot, (right) summary table reporting NGFP
    coordinates and critical exponents under each scheme. Inline
    citations: Litim (2001) and Wetterich (1993).

    Physics
    -------
    The Wetterich equation is exact, but truncations and regulator
    choices introduce scheme dependence. Comparing the optimised Litim
    cutoff (compact support, closed-form beta functions) with a smooth
    Exponential cutoff (``C^infty``, numerical integration) probes how
    much the NGFP coordinates and critical exponents drift between
    schemes — small drifts signal that universal physics is being
    captured.

    Parameters
    ----------
    system_litim, system_exponential : :class:`~asymsafety.beta.system.BetaFunctionSystem`
        Two beta-function systems differing only in the regulator.
    x_coupling, y_coupling : str, default ``"g", "lambda"``
    x_range, y_range : tuple
    n_grid : int, default ``30``
    fp_guess : dict, optional
        Initial guess for the NGFP search. Default is
        ``{"g": 0.7, "lambda": 0.14}``.
    figsize : tuple, default ``(15, 5)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Wetterich (1993), Phys. Lett. B 301, 90 — exact RG equation.
    - Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195] — optimised
      regulator with closed-form beta functions.
    - Codello et al. (2009), Ann. Phys. 324, 414 [0812.0785] — explicit
      scheme-dependence study.
    - See ``docs/LITERATURE.md``.

    See Also
    --------
    :func:`asymsafety.visualization.conceptual.regulator_comparison`
        1D plot of the regulator shape functions.
    """
    from asymsafety.analysis.fixed_points import FixedPointFinder
    from asymsafety.analysis.stability import analyze_stability

    if fp_guess is None:
        fp_guess = {"g": 0.7, "lambda": 0.14}

    fig, (ax_lit, ax_exp, ax_sum) = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 3, 2]},
    )

    fps: dict[str, FixedPoint | None] = {}
    sa_by_scheme: dict[str, object | None] = {}
    for label, system, ax in (
        ("Litim", system_litim, ax_lit),
        ("Exponential", system_exponential, ax_exp),
    ):
        # Streamplot
        x = np.linspace(x_range[0], x_range[1], n_grid)
        y = np.linspace(y_range[0], y_range[1], n_grid)
        X, Y = np.meshgrid(x, y)
        rhs = system.rhs_vector()
        names = system.coupling_names
        try:
            x_idx = names.index(x_coupling)
            y_idx = names.index(y_coupling)
        except ValueError:
            continue
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(n_grid):
            for j in range(n_grid):
                pt = np.zeros(len(names))
                pt[x_idx] = X[i, j]
                pt[y_idx] = Y[i, j]
                try:
                    bv = rhs(0, pt)
                    U[i, j] = bv[x_idx]
                    V[i, j] = bv[y_idx]
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
        speed = np.sqrt(U ** 2 + V ** 2)
        speed_safe = np.where(speed == 0, 1, speed)
        ax.streamplot(
            X, Y, U, V,
            color=np.log1p(speed_safe), cmap="coolwarm",
            density=1.4, linewidth=0.8, arrowsize=1.2,
        )
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel(coupling_label(x_coupling), fontsize=12)
        ax.set_ylabel(coupling_label(y_coupling), fontsize=12)
        ax.set_title(label, fontsize=13)

        # Find NGFP and annotate
        try:
            fp = FixedPointFinder(system).find_fixed_point(fp_guess)
        except Exception:
            fp = None
        fps[label] = fp
        if fp is not None:
            annotate_fixed_point(ax, fp, x_coupling, y_coupling)
            try:
                sa_by_scheme[label] = analyze_stability(system, fp)
            except Exception:
                sa_by_scheme[label] = None

    # Summary panel
    ax_sum.set_axis_off()
    ax_sum.set_xlim(0, 1)
    ax_sum.set_ylim(0, 1)
    ax_sum.set_title("Scheme dependence", fontsize=13)
    rows = [
        ("Scheme", "Litim", "Exponential"),
    ]
    for key, label in (
        ("g", r"$g^*$"), ("lambda", r"$\lambda^*$"),
    ):
        a = fps.get("Litim")
        b = fps.get("Exponential")
        rows.append((
            label,
            f"{a.location.get(key, float('nan')):.3f}" if a else "—",
            f"{b.location.get(key, float('nan')):.3f}" if b else "—",
        ))
    sa_a = sa_by_scheme.get("Litim")
    sa_b = sa_by_scheme.get("Exponential")
    rows.append((
        r"$\theta_1$",
        f"{sa_a.critical_exponents[0].real:.2f}" if sa_a is not None else "—",
        f"{sa_b.critical_exponents[0].real:.2f}" if sa_b is not None else "—",
    ))
    if sa_a is not None and len(sa_a.critical_exponents) > 1:
        rows.append((
            r"$\theta_2$",
            f"{sa_a.critical_exponents[1].real:.2f}",
            f"{sa_b.critical_exponents[1].real:.2f}"
            if sa_b is not None and len(sa_b.critical_exponents) > 1 else "—",
        ))

    n_rows = len(rows)
    row_h = 0.85 / n_rows
    for r, row in enumerate(rows):
        y_top = 0.92 - r * row_h
        is_header = r == 0
        for c, txt in enumerate(row):
            x_c = 0.18 + c * 0.32
            ax_sum.text(
                x_c, y_top, txt,
                fontsize=10,
                fontweight="bold" if is_header else "normal",
                color="0.2" if is_header else "0.15",
                ha="center", va="top",
            )
        if not is_header:
            ax_sum.axhline(
                y_top + row_h * 0.5, color="0.85", lw=0.5,
                xmin=0.05, xmax=0.95,
            )

    add_reference_box(
        ax_sum,
        [
            format_arxiv("Wetterich (1993)", "—"),
            format_arxiv("Litim (2001)", "hep-th/0103195"),
            format_arxiv("Codello et al. (2009)", "0812.0785"),
        ],
        loc="lower left",
    )

    fig.suptitle(
        r"Regulator dependence of the EH RG flow",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quadratic gravity pairwise grid
# ---------------------------------------------------------------------------


def quadratic_pairwise_grid(
    system: BetaFunctionSystem,
    couplings: tuple[str, str, str, str] = ("g", "lambda", "alpha", "beta"),
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_grid: int = 22,
    fp_guess: dict[str, float] | None = None,
    figsize: tuple[float, float] = (14, 9),
) -> Figure:
    r"""2x3 grid of pairwise streamplots for the quadratic gravity truncation.

    Physics
    -------
    Quadratic gravity ``Gamma_k = (16 pi G)^{-1} (R - 2 Lambda) +
    alpha R^2 + beta C_{munurhosigma}^2`` lives in the four-coupling
    space ``(g, lambda, alpha, beta)``. This grid plots all six
    pairwise projections, freezing the remaining two couplings at their
    NGFP values (Codello et al. 2009). Of particular note: the
    ``C^2`` coupling ``beta`` is asymptotically free at the GFP,
    visible as flow toward ``beta = 0`` in any panel containing it.

    Parameters
    ----------
    system : :class:`~asymsafety.beta.system.BetaFunctionSystem`
        Quadratic-gravity beta-function system, typically built via
        :func:`asymsafety.beta.quadratic.build_quadratic_beta_system`.
    couplings : tuple of 4 str, default ``("g", "lambda", "alpha", "beta")``
        Names of the four couplings.
    bounds : dict, optional
        Per-coupling ``(min, max)`` plotting ranges. Defaults are
        sensible for the canonical NGFP near ``(0.97, 0.14, 0.006, 0.002)``.
    n_grid : int, default ``22``
    fp_guess : dict, optional
        Seed for the four-coupling NGFP search. Default is the
        Codello (2009) approximate location.
    figsize : tuple, default ``(14, 9)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785]
    - See ``docs/LITERATURE.md``.

    See Also
    --------
    :mod:`asymsafety.validation.codello_2009`
        Benchmark fixed-point coordinates and the asymptotic-freedom
        result for ``beta``.
    :func:`asymsafety.beta.quadratic.build_quadratic_beta_system`
        Builder for the beta-function system.
    """
    from asymsafety.analysis.fixed_points import FixedPointFinder

    if bounds is None:
        bounds = {
            "g": (0.0, 1.6),
            "lambda": (-0.3, 0.45),
            "alpha": (-0.05, 0.05),
            "beta": (-0.05, 0.05),
        }
    if fp_guess is None:
        fp_guess = {"g": 0.97, "lambda": 0.14, "alpha": 0.006, "beta": 0.002}

    # Try to anchor frozen couplings at the NGFP for the most
    # representative slices.
    try:
        ngfp = FixedPointFinder(system).find_fixed_point(fp_guess)
        anchor = dict(ngfp.location)
    except Exception:
        ngfp = None
        anchor = dict(fp_guess)

    pairs = [
        (couplings[0], couplings[1]),
        (couplings[0], couplings[2]),
        (couplings[0], couplings[3]),
        (couplings[1], couplings[2]),
        (couplings[1], couplings[3]),
        (couplings[2], couplings[3]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()
    rhs = system.rhs_vector()
    names = system.coupling_names

    for ax, (xc, yc) in zip(axes_flat, pairs):
        try:
            x_idx = names.index(xc)
            y_idx = names.index(yc)
        except ValueError:
            ax.set_axis_off()
            continue
        x_lo, x_hi = bounds.get(xc, (-1.0, 1.0))
        y_lo, y_hi = bounds.get(yc, (-1.0, 1.0))
        X, Y = np.meshgrid(
            np.linspace(x_lo, x_hi, n_grid),
            np.linspace(y_lo, y_hi, n_grid),
        )
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(n_grid):
            for j in range(n_grid):
                pt = np.zeros(len(names))
                for k, n in enumerate(names):
                    pt[k] = anchor.get(n, 0.0)
                pt[x_idx] = X[i, j]
                pt[y_idx] = Y[i, j]
                try:
                    bv = rhs(0, pt)
                    U[i, j] = bv[x_idx]
                    V[i, j] = bv[y_idx]
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
        speed = np.sqrt(U ** 2 + V ** 2)
        speed_safe = np.where(speed == 0, 1, speed)
        ax.streamplot(
            X, Y, U, V,
            color=np.log1p(speed_safe), cmap="coolwarm",
            density=1.2, linewidth=0.7, arrowsize=1.0,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(coupling_label(xc), fontsize=11)
        ax.set_ylabel(coupling_label(yc), fontsize=11)
        if ngfp is not None:
            try:
                annotate_fixed_point(ax, ngfp, xc, yc)
            except Exception:
                pass
        # Frozen-coupling annotation
        frozen = [n for n in couplings if n not in (xc, yc)]
        frozen_text = ", ".join(
            f"{coupling_label(n).strip('$')}={anchor.get(n, 0.0):.3f}"
            for n in frozen
        )
        ax.text(
            0.02, 0.98, f"frozen: ${frozen_text}$",
            transform=ax.transAxes, fontsize=8, va="top",
            color="0.35",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85"),
        )

    fig.suptitle(
        r"Quadratic Gravity: Pairwise Phase Portraits "
        r"$(g, \lambda, \alpha, \beta)$",
        fontsize=14, fontweight="bold",
    )
    add_reference_box(
        axes_flat[-1],
        [format_arxiv("Codello et al. (2009)", "0812.0785")],
        loc="lower right",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectral / heat-kernel / batch-norm helpers
# ---------------------------------------------------------------------------


def plot_spectral_sum_convergence(
    spectral_sum,
    *,
    n_max_values: Sequence[int] | None = None,
    field_type: str = "scalar",
    R_bar_val: float = 12.0,
    W_func=None,
    ax=None,
):
    """Plot how a spectral sum converges as the angular-momentum cutoff grows.

    Useful for choosing a defensible ``l_max`` truncation: the curve
    typically saturates exponentially for sufficiently smooth weight
    functions ``W``. Default weight is the dimensionless Litim threshold
    ``W(z) = 1 / (1 + z)``.

    Args:
        spectral_sum: A
            :class:`asymsafety.frg.spectral.SpectralSumEvaluator` instance.
        n_max_values: Sequence of ``l_max`` truncation levels to sweep.
            Defaults to ``[2, 4, 8, 16, 32, 64, 128, 256]``.
        field_type: Field type passed to the spectrum
            (``"scalar"``, ``"vector"``, ``"TT"``).
        R_bar_val: Background Ricci scalar (controls the sphere radius).
        W_func: Optional user weight function. Defaults to
            ``lambda z: 1.0 / (1.0 + z)``.
        ax: Optional existing axes.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.get_figure()

    if n_max_values is None:
        n_max_values = [2, 4, 8, 16, 32, 64, 128, 256]

    if W_func is None:
        W_func = lambda z: 1.0 / (1.0 + z)  # noqa: E731

    original_lmax = spectral_sum.l_max
    try:
        sums = []
        for n in n_max_values:
            spectral_sum.l_max = int(n)
            sums.append(
                float(spectral_sum.trace_on_sphere(W_func, field_type, R_bar_val))
            )
    finally:
        spectral_sum.l_max = original_lmax

    sums = np.asarray(sums, dtype=float)

    ax.plot(n_max_values, sums, color=COLOR_RELEVANT, lw=2.0,
            marker="o", markersize=5, label=field_type)
    if sums.size > 1:
        asymptote = sums[-1]
        ax.axhline(asymptote, color=COLOR_TRAJECTORY, lw=1.0, ls="--",
                   label=rf"$l_{{\max}}={n_max_values[-1]}$ value")

    ax.set_xscale("log")
    ax.set_xlabel(r"angular-momentum cutoff $l_{\max}$")
    ax.set_ylabel(r"$\mathrm{Tr}\,W(-\bar D^2)$")
    ax.set_title("Spectral-sum convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Manrique et al. (2011)", "1003.5129")],
        loc="upper left",
    )
    return fig


def plot_heat_kernel_coefficients(
    sdw,
    *,
    field_types: Sequence[str] = ("scalar", "vector", "TT"),
    coefficients: Sequence[str] = ("b0", "b2", "b4"),
    ax=None,
):
    """Bar chart of Seeley-DeWitt coefficients ``b0, b2, b4`` per field type.

    Compares the Vassilevich heat-kernel expansion coefficients for
    several field species. Useful for a quick sanity check that a new
    truncation's matter content has the expected ``b_2`` / ``b_4``
    contributions.

    Args:
        sdw: A
            :class:`asymsafety.frg.heat_kernel.SeeleyDeWittCoefficients`
            instance.
        field_types: Field types to compare.
        coefficients: Coefficient names; one of
            ``("b0", "b2", "b4")``.
        ax: Optional existing axes.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        fig = ax.get_figure()

    n_fields = len(field_types)
    n_coeffs = len(coefficients)
    width = 0.8 / max(1, n_fields)
    x = np.arange(n_coeffs)

    for fi, ft in enumerate(field_types):
        vals = []
        for c in coefficients:
            try:
                if c == "b0":
                    val = float(sdw.b0(ft))
                elif c == "b2":
                    val = float(sdw.b2(ft))
                elif c == "b4":
                    val = float(sdw.b4_on_sphere(ft))
                else:
                    val = float("nan")
            except Exception:
                val = float("nan")
            vals.append(val)
        offset = (fi - (n_fields - 1) / 2.0) * width
        ax.bar(
            x + offset, vals, width=width,
            color=TRAJECTORY_COLORS[fi % len(TRAJECTORY_COLORS)],
            edgecolor="black", linewidth=0.4, label=ft,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(coefficients))
    ax.set_ylabel(r"coefficient $b_n$")
    ax.set_title("Seeley-DeWitt heat-kernel coefficients")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Vassilevich (2003)", "hep-th/0306138")],
        loc="lower left",
    )
    return fig


def plot_beta_norm_grid(
    arrays: dict,
    *,
    log10: bool = True,
    coupling_x: str | None = None,
    coupling_y: str | None = None,
    ax=None,
):
    """Heatmap of ``|beta(g)|`` from an ``asymsafety eval`` ``.npz`` payload.

    Reconstructs the 2D grid from a dict-like payload exposing
    ``axis_<name>`` and ``betas`` keys, computes the per-point Euclidean
    norm of the beta vector, and renders it as a heatmap. Used by
    notebook 05 (the acceleration / sweep tutorial) and by figure
    generators that operate on saved sweep outputs.

    Args:
        arrays: Mapping of field name to array, in the format produced
            by :func:`asymsafety.cli.io.write_results` (or returned by
            :func:`np.load` on the ``.npz`` output of
            ``asymsafety eval``). Required keys: ``"betas"``, plus one
            ``"axis_<name>"`` per coupling axis.
        log10: Render ``log10(|beta| + epsilon)`` instead of raw
            magnitude.
        coupling_x, coupling_y: Names of the two axes to plot. Defaults
            to the first two axes detected.
        ax: Optional existing axes.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
    else:
        fig = ax.get_figure()

    axis_keys = [k for k in arrays if str(k).startswith("axis_")]
    axis_keys.sort()
    if len(axis_keys) < 2:
        raise ValueError(
            "plot_beta_norm_grid requires at least two axis_* keys; got "
            f"{axis_keys}"
        )

    if coupling_x is None:
        coupling_x = axis_keys[0][len("axis_"):]
    if coupling_y is None:
        coupling_y = axis_keys[1][len("axis_"):]

    x_vals = np.asarray(arrays[f"axis_{coupling_x}"]).ravel()
    y_vals = np.asarray(arrays[f"axis_{coupling_y}"]).ravel()
    betas = np.asarray(arrays["betas"])

    nx = x_vals.size
    ny = y_vals.size
    n_couplings = betas.shape[-1]

    # Reshape (nx*ny, n_c) → (nx, ny, n_c). Reverse if cartesian product
    # was created with ``indexing='ij'`` (the default in eval_cmd).
    grid = betas.reshape(nx, ny, n_couplings)
    norms = np.linalg.norm(grid, axis=-1)
    plotted = np.log10(norms + 1e-12) if log10 else norms

    mesh = ax.imshow(
        plotted.T, origin="lower",
        extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
        aspect="auto", cmap="viridis",
    )
    add_colorbar(
        fig, ax, mesh,
        label=(r"$\log_{10}|\beta|$" if log10 else r"$|\beta|$"),
    )
    ax.set_xlabel(coupling_label(coupling_x))
    ax.set_ylabel(coupling_label(coupling_y))
    ax.set_title(r"$\beta$-function magnitude on the coupling grid")
    return fig
