"""Conceptual physics diagrams for asymptotic safety.

Publication-quality schematic figures illustrating the core concepts of
asymptotic safety, the Wetterich equation, regulator shape functions,
and fixed-point stability.  These are *pedagogical* diagrams — they
do not require running full FRG computations (except
:func:`regulator_comparison`, which evaluates symbolic shape functions).

References
----------
- Weinberg (1979), in *General Relativity: An Einstein Centenary
  Survey*, 790 — the asymptotic-safety conjecture.
- Wetterich (1993), Phys. Lett. B 301, 90 — exact RG flow equation.
- Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] — first NGFP.
- Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195] — optimised
  regulator with closed-form beta functions.
- Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810] — review.
- See ``docs/LITERATURE.md`` for the full bibliography.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe

from asymsafety.visualization.style import (
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_GFP,
    COLOR_SEPARATRIX,
    COLOR_TRAJECTORY,
    COLOR_LITIM,
    COLOR_EXPONENTIAL,
    add_reference_box,
    apply_style,
    format_arxiv,
)


# ── helpers ────────────────────────────────────────────────────────────


def _bezier_curve(
    pts: list[tuple[float, float]], n: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a cubic Bezier curve through *pts* (4 control points)."""
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(pts, codes)
    verts = path.interpolated(n).vertices
    return verts[:, 0], verts[:, 1]


def _add_flow_arrow(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    lw: float = 1.4,
    alpha: float = 0.7,
    idx: float = 0.55,
) -> None:
    """Draw a curve with a mid-point arrowhead."""
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=2)
    n = len(x)
    i = int(n * idx)
    dx = x[min(i + 1, n - 1)] - x[max(i - 1, 0)]
    dy = y[min(i + 1, n - 1)] - y[max(i - 1, 0)]
    ax.annotate(
        "",
        xy=(x[i] + dx * 0.01, y[i] + dy * 0.01),
        xytext=(x[i], y[i]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
        zorder=3,
    )


# ── public API ─────────────────────────────────────────────────────────


def asymptotic_safety_concept(
    figsize: tuple[float, float] = (12, 5),
    *,
    show_references: bool = True,
) -> Figure:
    r"""Two-panel figure explaining the core idea of asymptotic safety.

    Physics
    -------
    Left panel: schematic theory space showing the Gaussian (GFP) and
    non-Gaussian (NGFP) fixed points, RG flow trajectories, and the
    UV-critical surface — the codimension-``d_irr`` set of trajectories
    that hit the NGFP in the UV. Right panel: the dimensionless Newton
    coupling ``g(k) = G(k) k^2`` as a function of scale, contrasting
    the perturbative Landau-pole divergence (red dashed) with the
    asymptotically safe plateau ``g -> g^* ≈ 0.7`` (blue).

    Parameters
    ----------
    figsize : tuple, default ``(12, 5)``
    show_references : bool, default ``True``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Weinberg (1979), in *General Relativity: An Einstein Centenary
      Survey*, 790 — original asymptotic-safety conjecture.
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] — first
      explicit NGFP calculation.
    - Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810] — review.

    See Also
    --------
    :func:`asymsafety.visualization.phase_portrait.annotated_eh_phase_portrait`
        Data-driven counterpart on the (g, lambda) plane.
    :func:`fixed_point_stability_concept`
        Companion diagram for relevant / irrelevant directions.
    """
    apply_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
    )

    # ── Left panel: Theory Space ──────────────────────────────────────

    ax = ax_left
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.25, 0.55)
    ax.set_xlabel(r"$g$ (Newton)", fontsize=13)
    ax.set_ylabel(r"$\lambda$ (cosmological)", fontsize=13)
    ax.set_title("Theory Space & UV Fixed Point", fontsize=14)

    # UV-critical surface band (diagonal strip through NGFP)
    band_x = np.array([-0.1, 0.3, 0.7, 1.1, 1.25])
    band_center = np.array([0.38, 0.28, 0.15, 0.02, -0.05])
    bw = 0.08
    ax.fill_between(
        band_x, band_center - bw, band_center + bw,
        color=COLOR_RELEVANT, alpha=0.15, zorder=0,
    )
    ax.text(
        0.95, 0.30,
        r"UV-critical surface ($d_{\mathrm{UV}}$ free parameters)",
        fontsize=8, color=COLOR_RELEVANT, ha="right", style="italic",
    )

    # GFP
    ax.plot(
        0, 0, "o", color=COLOR_GFP, markersize=11,
        markeredgecolor="black", markeredgewidth=1.5, zorder=5,
    )
    ax.annotate(
        "GFP", xy=(0, 0), xytext=(-0.08, -0.07),
        fontsize=10, fontweight="bold", ha="center",
    )

    # NGFP
    ngfp_x, ngfp_y = 0.7, 0.15
    ax.plot(
        ngfp_x, ngfp_y, "*", color=COLOR_NGFP, markersize=18,
        markeredgecolor="black", markeredgewidth=0.8, zorder=5,
    )
    ax.annotate(
        "NGFP", xy=(ngfp_x, ngfp_y), xytext=(ngfp_x + 0.08, ngfp_y + 0.08),
        fontsize=10, fontweight="bold",
    )

    # Trajectories toward NGFP (UV direction)
    traj_in = [
        [(1.20, 0.50), (1.00, 0.35), (0.85, 0.22), (ngfp_x, ngfp_y)],
        [(1.15, 0.00), (1.00, 0.05), (0.85, 0.10), (ngfp_x, ngfp_y)],
        [(1.10, -0.15), (0.95, -0.02), (0.80, 0.08), (ngfp_x, ngfp_y)],
    ]
    for pts in traj_in:
        cx, cy = _bezier_curve(pts)
        _add_flow_arrow(ax, cx, cy, COLOR_TRAJECTORY, lw=1.4, alpha=0.65)

    # Trajectories away from GFP
    traj_out = [
        [(0, 0), (-0.02, -0.04), (-0.06, -0.10), (-0.10, -0.20)],
        [(0, 0), (0.08, -0.06), (0.15, -0.12), (0.22, -0.22)],
    ]
    for pts in traj_out:
        cx, cy = _bezier_curve(pts)
        _add_flow_arrow(ax, cx, cy, COLOR_SEPARATRIX, lw=1.2, alpha=0.5)

    # UV / IR labels
    ax.text(
        ngfp_x - 0.12, ngfp_y + 0.12, "UV",
        fontsize=11, color="0.35", fontweight="bold",
    )
    ax.text(
        -0.05, -0.22, "IR",
        fontsize=11, color="0.35", fontweight="bold",
    )

    # Text box
    ax.text(
        0.03, 0.97,
        r"Fixed point: $\beta_i(g^*) = 0$ for all $i$",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9),
    )

    # ── Right panel: Running Newton coupling ──────────────────────────

    ax = ax_right
    k = np.logspace(-1, 2.5, 300)
    g_star = 0.7

    # Asymptotically safe curve: tanh approach to plateau
    g_safe = g_star * np.tanh(1.2 * np.log10(k / k[0]))
    g_safe = np.clip(g_safe, 0.01, None)

    # Perturbative curve: diverges
    k_landau = 10 ** 1.7
    g_pert = 0.3 / (1.0 - (k / k_landau) ** 2)
    g_pert = np.where(k < k_landau * 0.92, g_pert, np.nan)

    ax.semilogx(k, g_safe, color=COLOR_RELEVANT, lw=2.2,
                label="Asymptotically safe")
    ax.semilogx(k, g_pert, color=COLOR_SEPARATRIX, lw=2.0, ls="--",
                label="Perturbative (Landau pole)")

    ax.axhline(g_star, color="0.5", ls=":", lw=1.0)
    ax.text(k[5], g_star + 0.05, r"$g^* \approx 0.7$", fontsize=10,
            color="0.4")

    ax.set_xlabel(r"$k$ (RG scale)", fontsize=13)
    ax.set_ylabel(r"$G(k)\,k^2$  (dimensionless)", fontsize=13)
    ax.set_title("Running Newton Coupling", fontsize=14)
    ax.set_ylim(-0.05, 2.0)
    ax.legend(fontsize=10, loc="upper left")

    # IR / UV annotations
    ax.annotate(
        r"IR $\leftarrow$", xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=9, color="0.4", va="top",
    )
    ax.annotate(
        r"$\rightarrow$ UV", xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=9, color="0.4", va="top", ha="right",
    )

    if show_references:
        add_reference_box(
            ax_left,
            [
                format_arxiv("Weinberg (1979)", "—"),
                format_arxiv("Reuter (1998)", "hep-th/9605030"),
                format_arxiv("Bonanno et al. (2020)", "2004.06810"),
            ],
            loc="lower right",
        )

    return fig


# ── regulator comparison ──────────────────────────────────────────────


def regulator_comparison(
    figsize: tuple[float, float] = (12, 5),
    *,
    show_references: bool = True,
) -> Figure:
    r"""Two-panel figure comparing Litim and Exponential regulators.

    Physics
    -------
    The IR regulator ``R_k(z)`` damps low-momentum modes ``z < k^2``
    and drives the Wetterich flow. The optimised Litim regulator
    ``R_k(z) = (k^2 - z) Theta(k^2 - z)`` (left panel, blue) has compact
    support and yields rational, closed-form beta functions, while the
    smooth ``C^infty`` exponential regulator (orange) requires numerical
    integration. Their qualitative differences and shared physics
    content (universal critical exponents) are summarised in the right
    panel.

    Parameters
    ----------
    figsize : tuple, default ``(12, 5)``
    show_references : bool, default ``True``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Wetterich (1993), Phys. Lett. B 301, 90 — exact RG equation.
    - Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195] — optimised
      regulator with rational beta functions.

    See Also
    --------
    :func:`asymsafety.visualization.phase_portrait.regulator_comparison_panels`
        Phase-portrait-level comparison of the same regulators.
    """
    import sympy
    from sympy import Symbol

    from asymsafety.frg.regulator import ExponentialRegulator, LitimRegulator

    apply_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
    )

    # ── Left panel: shape functions r(y) ──────────────────────────────

    y_sym = Symbol("y", positive=True)
    litim = LitimRegulator()
    exponential = ExponentialRegulator()

    r_litim_sym = litim.shape_function(y_sym)
    r_exp_sym = exponential.shape_function(y_sym)

    r_litim_fn = sympy.lambdify(
        y_sym, r_litim_sym,
        modules=[{"Heaviside": lambda x, h=0.5: np.heaviside(x, h)}, "numpy"],
    )
    r_exp_fn = sympy.lambdify(y_sym, r_exp_sym, modules="numpy")

    y_vals = np.linspace(0.01, 3.0, 500)
    r_litim_vals = np.array([float(r_litim_fn(yv)) for yv in y_vals])
    r_exp_vals = r_exp_fn(y_vals)

    ax = ax_left
    ax.plot(y_vals, r_litim_vals, color=COLOR_LITIM, lw=2.2,
            label="Litim (optimized)")
    ax.plot(y_vals, r_exp_vals, color=COLOR_EXPONENTIAL, lw=2.2,
            label="Exponential")
    ax.set_xlabel(r"$y = z/k^2$", fontsize=13)
    ax.set_ylabel(r"$r(y)$", fontsize=13)
    ax.set_title("Regulator Shape Functions", fontsize=14)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.1, 5.0)
    ax.legend(fontsize=10)

    # Annotations
    ax.annotate(
        "Step cutoff",
        xy=(0.5, float(r_litim_fn(0.5))),
        xytext=(1.3, 3.5),
        fontsize=10, color=COLOR_LITIM,
        arrowprops=dict(arrowstyle="->", color=COLOR_LITIM, lw=1.2),
    )
    ax.annotate(
        r"Smooth ($C^\infty$)",
        xy=(0.5, float(r_exp_fn(0.5))),
        xytext=(1.6, 2.0),
        fontsize=10, color=COLOR_EXPONENTIAL,
        arrowprops=dict(arrowstyle="->", color=COLOR_EXPONENTIAL, lw=1.2),
    )

    # ── Right panel: properties comparison table ──────────────────────

    ax = ax_right
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Properties Comparison", fontsize=14)

    col_x = [0.35, 0.70]  # column centres
    row_labels = ["Shape", "Beta functions", "Scheme dep.", "Reference"]
    litim_vals = [
        "Step (compact support)",
        "Rational / closed-form",
        "Minimized",
        "Litim (2001)",
    ]
    exp_vals = [
        r"Smooth ($C^\infty$)",
        "Requires numerical int.",
        "Moderate",
        "Wetterich (1993)",
    ]

    n_rows = len(row_labels)
    row_h = 0.13
    top_y = 0.88

    # Header bars
    hw = 0.27
    hh = 0.07
    ax.add_patch(mpatches.FancyBboxPatch(
        (col_x[0] - hw / 2, top_y - hh / 2), hw, hh,
        boxstyle="round,pad=0.02", fc=COLOR_LITIM, ec="none", alpha=0.8,
    ))
    ax.text(col_x[0], top_y, "Litim", fontsize=11, fontweight="bold",
            color="white", ha="center", va="center")
    ax.add_patch(mpatches.FancyBboxPatch(
        (col_x[1] - hw / 2, top_y - hh / 2), hw, hh,
        boxstyle="round,pad=0.02", fc=COLOR_EXPONENTIAL, ec="none", alpha=0.8,
    ))
    ax.text(col_x[1], top_y, "Exponential", fontsize=11, fontweight="bold",
            color="white", ha="center", va="center")

    # Row label column
    label_x = 0.08
    for i, label in enumerate(row_labels):
        y = top_y - (i + 1) * row_h
        ax.text(label_x, y, label, fontsize=10, fontweight="bold",
                ha="left", va="center", color="0.25")
        ax.text(col_x[0], y, litim_vals[i], fontsize=9.5, ha="center",
                va="center", color="0.15")
        ax.text(col_x[1], y, exp_vals[i], fontsize=9.5, ha="center",
                va="center", color="0.15")

    # Light grid lines
    for i in range(n_rows + 1):
        y = top_y - (i + 0.5) * row_h
        ax.axhline(y, color="0.85", lw=0.6, xmin=0.03, xmax=0.97)

    if show_references:
        add_reference_box(
            ax_left,
            [
                format_arxiv("Wetterich (1993)", "—"),
                format_arxiv("Litim (2001)", "hep-th/0103195"),
            ],
            loc="upper right",
        )

    return fig


# ── fixed point stability ────────────────────────────────────────────


def fixed_point_stability_concept(
    figsize: tuple[float, float] = (12, 5),
    *,
    show_references: bool = True,
) -> Figure:
    r"""Two-panel figure explaining fixed-point stability and predictivity.

    Physics
    -------
    Linearised flow near a fixed point ``g^*`` is governed by the
    stability matrix ``M_ij = d beta_i / d g_j``; critical exponents
    ``theta_i = -eig(M)`` separate UV-attractive (relevant,
    ``Re theta > 0``) from UV-repulsive (irrelevant, ``Re theta < 0``)
    directions. The number of relevant directions equals the number of
    free parameters in the asymptotically safe theory — its
    predictivity. Left panel: schematic linearised flow with shaded
    UV-critical surface. Right panel: bar chart of representative
    ``theta_i`` values colour-coded by relevance.

    Parameters
    ----------
    figsize : tuple, default ``(12, 5)``
    show_references : bool, default ``True``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Bonanno et al. (2020), Front. Phys. 8, 269 [2004.06810]

    See Also
    --------
    :func:`asymsafety.gui.visualization_3d.fixed_point_stability_3d`
        Data-driven 3D counterpart at a real fixed point.
    """
    apply_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
    )

    # ── Left panel: linearised flow ───────────────────────────────────

    ax = ax_left
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Linearised RG Flow", fontsize=14)
    ax.set_xlabel(r"$\delta g_1$", fontsize=13)
    ax.set_ylabel(r"$\delta g_2$", fontsize=13)

    # FP marker
    ax.plot(
        0, 0, "*", color=COLOR_NGFP, markersize=20,
        markeredgecolor="black", markeredgewidth=0.8, zorder=10,
    )

    # Eigenvector 1: relevant (theta > 0), UV-attractive
    angle_rel = np.deg2rad(30)
    ev_len = 1.25
    dx_r, dy_r = ev_len * np.cos(angle_rel), ev_len * np.sin(angle_rel)

    # Draw relevant eigenvector axis
    ax.annotate(
        "", xy=(dx_r, dy_r), xytext=(-dx_r, -dy_r),
        arrowprops=dict(arrowstyle="-", color=COLOR_RELEVANT, lw=2.5,
                        linestyle="-"),
        zorder=3,
    )
    ax.text(
        dx_r * 1.08, dy_r * 1.08,
        r"Relevant ($\theta > 0$)",
        fontsize=10, color=COLOR_RELEVANT, fontweight="bold",
        ha="left", va="bottom",
    )

    # Small flow arrows along relevant direction (toward FP)
    for frac in [0.35, 0.65, -0.35, -0.65]:
        px, py = frac * dx_r, frac * dy_r
        sign = -1.0 if frac > 0 else 1.0
        adx = sign * 0.12 * np.cos(angle_rel)
        ady = sign * 0.12 * np.sin(angle_rel)
        ax.annotate(
            "", xy=(px + adx, py + ady), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_RELEVANT, lw=1.5),
            zorder=4,
        )

    # Eigenvector 2: irrelevant (theta < 0), UV-repulsive
    angle_irr = angle_rel + np.pi / 2
    dx_i, dy_i = ev_len * np.cos(angle_irr), ev_len * np.sin(angle_irr)
    ax.annotate(
        "", xy=(dx_i, dy_i), xytext=(-dx_i, -dy_i),
        arrowprops=dict(arrowstyle="-", color=COLOR_IRRELEVANT, lw=2.5,
                        linestyle="-"),
        zorder=3,
    )
    ax.text(
        dx_i * 1.08, dy_i * 1.08,
        r"Irrelevant ($\theta < 0$)",
        fontsize=10, color=COLOR_IRRELEVANT, fontweight="bold",
        ha="right", va="bottom",
    )

    # Small flow arrows along irrelevant direction (away from FP)
    for frac in [0.35, 0.65, -0.35, -0.65]:
        px, py = frac * dx_i, frac * dy_i
        sign = 1.0 if frac > 0 else -1.0
        adx = sign * 0.12 * np.cos(angle_irr)
        ady = sign * 0.12 * np.sin(angle_irr)
        ax.annotate(
            "", xy=(px + adx, py + ady), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_IRRELEVANT, lw=1.5),
            zorder=4,
        )

    # Curved trajectory lines (spiralling)
    t_param = np.linspace(0, 1.0, 200)
    for start_angle_deg in [10, 80, 190, 260]:
        sa = np.deg2rad(start_angle_deg)
        # Inward along relevant, outward along irrelevant
        proj_r = 1.1 * np.cos(sa - angle_rel) * np.exp(-2.0 * t_param)
        proj_i = 0.3 * np.sin(sa - angle_rel) * np.exp(1.5 * t_param)
        proj_i = np.clip(proj_i, -1.4, 1.4)
        cx = proj_r * np.cos(angle_rel) + proj_i * np.cos(angle_irr)
        cy = proj_r * np.sin(angle_rel) + proj_i * np.sin(angle_irr)
        mask = (np.abs(cx) < 1.45) & (np.abs(cy) < 1.45)
        ax.plot(cx[mask], cy[mask], color=COLOR_TRAJECTORY, lw=0.9,
                alpha=0.45, zorder=1)

    # UV-critical surface shading along relevant direction
    perp_x, perp_y = -np.sin(angle_rel), np.cos(angle_rel)
    strip_w = 0.22
    strip_corners = np.array([
        [-dx_r - strip_w * perp_x, -dy_r - strip_w * perp_y],
        [dx_r - strip_w * perp_x, dy_r - strip_w * perp_y],
        [dx_r + strip_w * perp_x, dy_r + strip_w * perp_y],
        [-dx_r + strip_w * perp_x, -dy_r + strip_w * perp_y],
    ])
    from matplotlib.patches import Polygon
    strip = Polygon(
        strip_corners, closed=True,
        fc=COLOR_RELEVANT, ec="none", alpha=0.10, zorder=0,
    )
    ax.add_patch(strip)
    ax.text(
        -1.0, -1.3, "UV-critical surface",
        fontsize=9, color=COLOR_RELEVANT, style="italic",
    )

    # ── Right panel: critical exponents bar chart ─────────────────────

    ax = ax_right
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    values = [2.5, 1.2, -1.5]
    colors = [
        COLOR_RELEVANT if v > 0 else COLOR_IRRELEVANT for v in values
    ]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="0.3",
                  linewidth=0.8, zorder=3)
    ax.axhline(0, color="0.4", ls="--", lw=1.0, zorder=2)

    ax.set_title(r"Critical Exponents $\theta_i$", fontsize=14)
    ax.set_ylabel(r"$\theta_i$", fontsize=13)
    ax.set_ylim(-2.5, 3.5)

    # Annotations next to bars
    for bar_rect, val in zip(bars, values):
        x_c = bar_rect.get_x() + bar_rect.get_width() / 2
        y_tip = val
        label = "UV-attractive" if val > 0 else "UV-repulsive"
        offset_y = 0.25 if val > 0 else -0.35
        ax.text(
            x_c, y_tip + offset_y, label,
            fontsize=9, ha="center", va="bottom" if val > 0 else "top",
            color="0.3", style="italic",
        )

    # Bottom equation annotation
    ax.text(
        0.5, 0.02,
        r"$\theta_i = -\mathrm{eigenvalues}(M_{ij})$"
        r" where $M_{ij} = \partial\beta_i / \partial g_j$",
        transform=ax.transAxes, fontsize=9, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.65",
                  alpha=0.9),
    )

    if show_references:
        add_reference_box(
            ax_left,
            [
                format_arxiv("Reuter (1998)", "hep-th/9605030"),
                format_arxiv("Bonanno et al. (2020)", "2004.06810"),
            ],
            loc="lower right",
        )

    return fig


# ── Wetterich equation diagram ───────────────────────────────────────


def wetterich_equation_diagram(
    figsize: tuple[float, float] = (10, 4),
    *,
    show_references: bool = True,
) -> Figure:
    r"""Schematic of the Wetterich equation with a one-loop diagram.

    Physics
    -------
    The Wetterich equation
    ``d_t Gamma_k = (1/2) Tr [(Gamma_k^(2) + R_k)^{-1} d_t R_k]``
    is the exact one-loop RG flow equation for the effective average
    action ``Gamma_k``: the trace runs over the full regularised
    propagator ``(Gamma_k^(2) + R_k)^{-1}`` with a single regulator
    insertion ``d_t R_k`` (cross). Only modes near the running scale
    ``k`` contribute to the flow.

    Parameters
    ----------
    figsize : tuple, default ``(10, 4)``
    show_references : bool, default ``True``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Wetterich (1993), Phys. Lett. B 301, 90 — exact RG equation.
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] —
      application to gravity.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(
        5.0, 3.75, "The Wetterich Equation (1993)",
        fontsize=15, fontweight="bold", ha="center", va="top",
    )

    # ── Main equation ─────────────────────────────────────────────────
    eq_text = (
        r"$\partial_t \Gamma_k = \frac{1}{2}\,"
        r"\mathrm{Tr}\!\left[\left(\Gamma_k^{(2)} + R_k\right)^{-1}"
        r"\partial_t R_k\right]$"
    )
    ax.text(5.0, 3.15, eq_text, fontsize=16, ha="center", va="center")

    # ── One-loop diagram ──────────────────────────────────────────────
    cx, cy, r = 3.0, 1.5, 0.7

    # The loop (full circle)
    theta = np.linspace(0, 2 * np.pi, 200)
    loop_x = cx + r * np.cos(theta)
    loop_y = cy + r * np.sin(theta)
    ax.plot(
        loop_x, loop_y, color=COLOR_TRAJECTORY, lw=2.5, zorder=3,
        path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
    )

    # Regulator insertion: cross (X) at bottom of loop
    ins_angle = -np.pi / 2
    ins_x = cx + r * np.cos(ins_angle)
    ins_y = cy + r * np.sin(ins_angle)
    cross_sz = 0.12
    ax.plot(
        [ins_x - cross_sz, ins_x + cross_sz],
        [ins_y - cross_sz, ins_y + cross_sz],
        color=COLOR_SEPARATRIX, lw=3, zorder=5,
    )
    ax.plot(
        [ins_x - cross_sz, ins_x + cross_sz],
        [ins_y + cross_sz, ins_y - cross_sz],
        color=COLOR_SEPARATRIX, lw=3, zorder=5,
    )

    # Label: regulator insertion
    ax.annotate(
        r"$\partial_t R_k$",
        xy=(ins_x, ins_y),
        xytext=(ins_x - 1.0, ins_y - 0.55),
        fontsize=12, color=COLOR_SEPARATRIX, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_SEPARATRIX, lw=1.2),
    )

    # Label: propagator line
    prop_angle = np.pi / 3
    prop_x = cx + r * np.cos(prop_angle)
    prop_y = cy + r * np.sin(prop_angle)
    ax.annotate(
        r"$\left(\Gamma_k^{(2)} + R_k\right)^{-1}$",
        xy=(prop_x, prop_y),
        xytext=(prop_x + 0.5, prop_y + 0.55),
        fontsize=11, color=COLOR_TRAJECTORY,
        arrowprops=dict(arrowstyle="->", color=COLOR_TRAJECTORY, lw=1.0),
    )

    # Label: trace (loop)
    ax.text(
        cx, cy, "Tr",
        fontsize=12, ha="center", va="center",
        color="0.4", fontweight="bold",
    )

    # ── Right-side annotation block ───────────────────────────────────
    ann_x = 7.0
    annotations = [
        (r"$\Gamma_k^{(2)}$: Hessian of effective average action", 2.1),
        (r"$R_k$: IR regulator (suppresses modes below $k$)", 1.55),
        (r"$\partial_t R_k$: drives the flow — only modes near $k$ contribute",
         1.0),
    ]
    for text, y_pos in annotations:
        ax.text(
            ann_x, y_pos, text,
            fontsize=10, ha="center", va="center", color="0.2",
        )

    if show_references:
        ax.text(
            5.0, 0.25,
            "Wetterich (1993), Phys. Lett. B 301, 90",
            fontsize=8, ha="center", va="center", color="0.45",
            style="italic",
        )

    return fig
