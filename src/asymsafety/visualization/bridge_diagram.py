"""Cross-analogue bridge and hydraulic analogy diagrams.

Publication-quality schematic figures illustrating the commutative
relationships between different mathematical representations of the
renormalization group (classical RG, control theory / hydraulics,
integral transforms, and quantum computing), as well as the detailed
hydraulic network analogy for RG flow.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from asymsafety.visualization.style import (
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_GFP,
    COLOR_SEPARATRIX,
    COLOR_TRAJECTORY,
    TRAJECTORY_COLORS,
    apply_style,
)


# ── helpers ────────────────────────────────────────────────────────────

# Domain colours for the four analogue boxes
_CLR_RG = "#2176AE"           # Classical RG — blue
_CLR_HYDRAULIC = "#06D6A0"    # Hydraulic / control — green
_CLR_TRANSFORMS = "#F77F00"   # Integral transforms — orange
_CLR_QUANTUM = "#9B5DE5"      # Quantum computing — purple


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    header: str,
    body_lines: list[str],
    header_color: str,
    *,
    fontsize_header: int = 11,
    fontsize_body: int = 9,
) -> None:
    """Draw a rounded-rectangle box with a coloured header bar and body text."""
    # Header bar
    hdr_h = h * 0.30
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle="round,pad=0.08",
        fc=header_color, ec="0.3", lw=1.2, alpha=0.85, zorder=3,
    ))
    ax.text(
        x + w / 2, y + h - hdr_h / 2, header,
        fontsize=fontsize_header, fontweight="bold", color="white",
        ha="center", va="center", zorder=4,
    )

    # Body background
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h - hdr_h,
        boxstyle="round,pad=0.08",
        fc="white", ec="0.3", lw=1.2, zorder=2,
    ))

    # Body text
    body_text = "\n".join(body_lines)
    ax.text(
        x + w / 2, y + (h - hdr_h) / 2, body_text,
        fontsize=fontsize_body, ha="center", va="center",
        color="0.15", zorder=4, linespacing=1.45,
    )


def _curved_arrow(
    ax: plt.Axes,
    xy_a: tuple[float, float],
    xy_b: tuple[float, float],
    label: str,
    *,
    color: str = "0.35",
    connectionstyle: str = "arc3,rad=0.15",
    fontsize: int = 8,
    label_offset: tuple[float, float] = (0, 0),
) -> None:
    """Draw a curved labelled arrow between two points."""
    arrow = FancyArrowPatch(
        xy_a, xy_b,
        connectionstyle=connectionstyle,
        arrowstyle="-|>",
        mutation_scale=14,
        lw=1.4,
        color=color,
        zorder=5,
    )
    ax.add_patch(arrow)
    mx = (xy_a[0] + xy_b[0]) / 2 + label_offset[0]
    my = (xy_a[1] + xy_b[1]) / 2 + label_offset[1]
    ax.text(
        mx, my, label,
        fontsize=fontsize, ha="center", va="center",
        color=color, style="italic",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        zorder=6,
    )


# ── public API ─────────────────────────────────────────────────────────


def cross_analogue_bridge_diagram(
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    r"""Commutative diagram connecting four analogue representations of RG flow.

    Physics
    -------
    The package's :class:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`
    enforces a commutative diagram: every path through the boxes
    (classical RG ↔ hydraulic network ↔ quantum circuit ↔ integral
    transforms) yields the same critical exponents ``theta_i`` at a
    fixed point. Boxes: Classical RG (top), Hydraulic / Control Theory
    (right), Integral Transforms (bottom), Quantum Computing (left).
    Arrows show the mathematical bridges between domains. Used as the
    cover-figure for the cross-analogue subsystem.

    Parameters
    ----------
    figsize : tuple, default ``(10, 8)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - Wetterich (1993), Phys. Lett. B 301, 90 — exact RG equation
      (anchor of the classical-RG box).
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].
    - See ``docs/LITERATURE.md`` and the README section
      "Physical Computational Analogues".

    See Also
    --------
    :func:`hydraulic_analogy_diagram`
        Side-by-side companion comparing RG and hydraulic networks.
    :class:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`
        Numerical implementation of the diagram.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_title(
        "Cross-Analogue Bridge: Commutative Diagram",
        fontsize=15, fontweight="bold", pad=12,
    )

    # Box dimensions
    bw, bh = 3.2, 1.6

    # Positions (lower-left corner of each box)
    pos_top = (5 - bw / 2, 6.3)       # Classical RG
    pos_right = (7.2, 3.4 - bh / 2)   # Hydraulic
    pos_bottom = (5 - bw / 2, 0.0)    # Transforms
    pos_left = (0.0, 3.4 - bh / 2)    # Quantum

    _rounded_box(
        ax, *pos_top, bw, bh,
        "Classical RG",
        [r"$\beta$ functions", "Fixed points", r"Critical exponents $\theta_i$"],
        _CLR_RG,
    )
    _rounded_box(
        ax, *pos_right, bw, bh,
        "Control / Hydraulic",
        ["Impedance Z(s)", "Bode plots", "Pipe network"],
        _CLR_HYDRAULIC,
    )
    _rounded_box(
        ax, *pos_bottom, bw, bh,
        "Integral Transforms",
        ["Laplace, Mellin", "Wavelet, Fourier"],
        _CLR_TRANSFORMS,
    )
    _rounded_box(
        ax, *pos_left, bw, bh,
        "Quantum Computing",
        ["VQRG circuits", "Grover search", "Koopman simulation"],
        _CLR_QUANTUM,
    )

    # Connection points (midpoints of box edges)
    top_bottom_edge = (5, pos_top[1])
    top_right_edge = (pos_top[0] + bw, pos_top[1] + bh / 2)
    top_left_edge = (pos_top[0], pos_top[1] + bh / 2)

    right_top_edge = (pos_right[0] + bw / 2, pos_right[1] + bh)
    right_bottom_edge = (pos_right[0] + bw / 2, pos_right[1])

    bottom_top_edge = (5, pos_bottom[1] + bh)
    bottom_right_edge = (pos_bottom[0] + bw, pos_bottom[1] + bh / 2)
    bottom_left_edge = (pos_bottom[0], pos_bottom[1] + bh / 2)

    left_top_edge = (pos_left[0] + bw / 2, pos_left[1] + bh)
    left_bottom_edge = (pos_left[0] + bw / 2, pos_left[1])

    # Top -> Right: impedance bridge
    _curved_arrow(
        ax,
        (pos_top[0] + bw, pos_top[1] + bh * 0.3),
        (pos_right[0], pos_right[1] + bh * 0.7),
        "Impedance bridge",
        color=_CLR_HYDRAULIC, connectionstyle="arc3,rad=-0.2",
        label_offset=(0.3, 0.3),
    )

    # Top -> Bottom: Laplace / Mellin
    _curved_arrow(
        ax,
        (pos_top[0] + bw * 0.65, pos_top[1]),
        (pos_bottom[0] + bw * 0.65, pos_bottom[1] + bh),
        "Laplace / Mellin transform",
        color=_CLR_TRANSFORMS, connectionstyle="arc3,rad=-0.15",
        label_offset=(0.9, 0),
    )

    # Top -> Left: variational encoding
    _curved_arrow(
        ax,
        (pos_top[0], pos_top[1] + bh * 0.3),
        (pos_left[0] + bw, pos_left[1] + bh * 0.7),
        "Variational encoding",
        color=_CLR_QUANTUM, connectionstyle="arc3,rad=0.2",
        label_offset=(-0.3, 0.3),
    )

    # Right -> Bottom: transfer functions
    _curved_arrow(
        ax,
        (pos_right[0], pos_right[1] + bh * 0.2),
        (pos_bottom[0] + bw, pos_bottom[1] + bh * 0.7),
        "Transfer functions H(s)",
        color="0.45", connectionstyle="arc3,rad=-0.2",
        label_offset=(0.3, -0.2),
    )

    # Left -> Bottom: Koopman evolution
    _curved_arrow(
        ax,
        (pos_left[0] + bw, pos_left[1] + bh * 0.2),
        (pos_bottom[0], pos_bottom[1] + bh * 0.7),
        "Koopman evolution",
        color="0.45", connectionstyle="arc3,rad=0.2",
        label_offset=(-0.3, -0.2),
    )

    # Centre text
    ax.text(
        5, 3.4,
        r"All paths yield consistent $\theta_i$",
        fontsize=12, ha="center", va="center",
        fontweight="bold", color="0.25",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="0.6",
                  alpha=0.95),
        zorder=7,
    )

    return fig


# ── hydraulic analogy ─────────────────────────────────────────────────


def hydraulic_analogy_diagram(
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    r"""Side-by-side comparison of RG flow and a hydraulic network.

    Physics
    -------
    Maps RG-flow concepts onto a control-theoretic / hydraulic
    framework: couplings ``g_i`` ↔ pressures ``P_i``, beta functions
    ↔ flow equations, fixed points ↔ steady states, regulators ↔
    valves, critical exponents ↔ impedance eigenvalues. Left panel:
    schematic RG vector field with the NGFP and inflow trajectories.
    Right panel: pipe network with nodes, pressures, and flow arrows.
    Centre: correspondence table with double-headed arrows.

    Parameters
    ----------
    figsize : tuple, default ``(14, 6)``

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - See README "Physical Computational Analogues".
    - The hydraulic mapping itself is documented in
      :class:`asymsafety.hydraulic.mapping.RGToHydraulicMapper`.

    See Also
    --------
    :func:`asymsafety.hydraulic.visualization.plot_hydraulic_network`
        Data-driven plot of the actual generated network.
    :func:`cross_analogue_bridge_diagram`
        Higher-level commutative-diagram view.
    """
    apply_style()

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Three panels: left (RG), centre (correspondence), right (hydraulic)
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 2.5, 4])
    ax_rg = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_hyd = fig.add_subplot(gs[0, 2])

    fig.suptitle("The Hydraulic Analogy", fontsize=16, fontweight="bold")

    # ── Left panel: RG Flow ───────────────────────────────────────────

    ax = ax_rg
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.3, 0.6)
    ax.set_xlabel(r"$g$", fontsize=13)
    ax.set_ylabel(r"$\lambda$", fontsize=13)
    ax.set_title("RG Flow", fontsize=13)

    # Simple schematic arrows (vector field)
    arrow_pts_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1,
                            0.1, 0.3, 0.5, 0.7, 0.9, 1.1,
                            0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    arrow_pts_y = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
                            0.40, 0.40, 0.40, 0.40, 0.40, 0.40])
    # Flow toward NGFP at (0.7, 0.15)
    fp_x, fp_y = 0.7, 0.15
    dx = (fp_x - arrow_pts_x) * 0.12
    dy = (fp_y - arrow_pts_y) * 0.12
    ax.quiver(
        arrow_pts_x, arrow_pts_y, dx, dy,
        color=COLOR_TRAJECTORY, alpha=0.35, scale=1, width=0.005, zorder=1,
    )

    # NGFP star
    ax.plot(
        fp_x, fp_y, "*", color=COLOR_NGFP, markersize=18,
        markeredgecolor="black", markeredgewidth=0.8, zorder=5,
    )
    ax.text(fp_x + 0.06, fp_y + 0.06, "NGFP", fontsize=10, fontweight="bold")

    # Trajectories
    from matplotlib.path import Path as MplPath

    traj_sets = [
        [(1.2, 0.45), (1.0, 0.30), (0.85, 0.20), (fp_x, fp_y)],
        [(0.1, -0.15), (0.3, -0.05), (0.5, 0.05), (fp_x, fp_y)],
        [(1.15, -0.10), (0.95, 0.00), (0.80, 0.10), (fp_x, fp_y)],
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    for pts in traj_sets:
        path = MplPath(pts, codes)
        verts = path.interpolated(60).vertices
        ax.plot(verts[:, 0], verts[:, 1], color=COLOR_RELEVANT, lw=1.4,
                alpha=0.7, zorder=2)

    # Labels
    ax.text(
        0.05, 0.55, r"Couplings $(g, \lambda)$",
        fontsize=9, color="0.3", style="italic",
    )
    ax.text(
        0.05, 0.50, r"$\beta$ functions $=$ flow",
        fontsize=9, color="0.3", style="italic",
    )
    ax.text(
        0.05, -0.22, r"Fixed point: $\beta = 0$",
        fontsize=9, color="0.3", style="italic",
    )

    # ── Right panel: Hydraulic Network ────────────────────────────────

    ax = ax_hyd
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.5, 3.0)
    ax.set_axis_off()
    ax.set_title("Hydraulic Network", fontsize=13)

    # Nodes
    node_pos = {
        "R": (0.3, 2.3),
        "P1": (2.0, 2.3),
        "P2": (2.0, 0.5),
        "P3": (0.3, 0.5),
    }

    # Reservoir (square, blue)
    rsv = mpatches.FancyBboxPatch(
        (node_pos["R"][0] - 0.3, node_pos["R"][1] - 0.25), 0.6, 0.5,
        boxstyle="square,pad=0.05", fc=COLOR_RELEVANT, ec="0.2",
        lw=1.5, alpha=0.6, zorder=3,
    )
    ax.add_patch(rsv)
    ax.text(
        node_pos["R"][0], node_pos["R"][1], "Reservoir",
        fontsize=9, ha="center", va="center", fontweight="bold",
        color="white", zorder=4,
    )

    # Pressure nodes (circles)
    for name, (nx, ny) in node_pos.items():
        if name == "R":
            continue
        circle = plt.Circle(
            (nx, ny), 0.22, fc="white", ec="0.3", lw=1.5, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            nx, ny, f"$P_{name[-1]}$", fontsize=11, ha="center",
            va="center", zorder=4,
        )

    # Pipes (thick lines with flow arrows)
    pipes = [
        ("R", "P1"),
        ("P1", "P2"),
        ("P2", "P3"),
        ("P3", "R"),
        ("R", "P2"),  # diagonal
    ]
    for n_from, n_to in pipes:
        x0, y0 = node_pos[n_from]
        x1, y1 = node_pos[n_to]
        ax.plot(
            [x0, x1], [y0, y1],
            color="0.55", lw=4, solid_capstyle="round", zorder=1,
        )
        # Flow arrow at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ddx, ddy = (x1 - x0) * 0.08, (y1 - y0) * 0.08
        ax.annotate(
            "", xy=(mx + ddx, my + ddy), xytext=(mx - ddx, my - ddy),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_TRAJECTORY, lw=1.8),
            zorder=5,
        )

    # Valve symbol on one pipe (R -> P2 diagonal): small diamond
    vx = (node_pos["R"][0] + node_pos["P2"][0]) / 2
    vy = (node_pos["R"][1] + node_pos["P2"][1]) / 2
    valve_sz = 0.15
    valve = plt.Polygon(
        [
            (vx, vy + valve_sz),
            (vx + valve_sz, vy),
            (vx, vy - valve_sz),
            (vx - valve_sz, vy),
        ],
        closed=True, fc=COLOR_SEPARATRIX, ec="0.2", lw=1.2, zorder=6,
    )
    ax.add_patch(valve)
    ax.text(
        vx + 0.25, vy, "Valve", fontsize=8, color=COLOR_SEPARATRIX,
        va="center",
    )

    # Labels below
    ax.text(
        1.5, -0.15, r"Pressures $(P_1, P_2)$",
        fontsize=9, color="0.3", style="italic", ha="center",
    )
    ax.text(
        1.5, -0.35, "Flow equations",
        fontsize=9, color="0.3", style="italic", ha="center",
    )

    # ── Centre panel: correspondence ──────────────────────────────────

    ax = ax_mid
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    correspondences = [
        (r"Coupling $g_i$", r"Pressure $P_i$"),
        (r"$\beta_i(g)$", "Flow at node $i$"),
        ("Fixed point", "Steady state"),
        (r"$\theta_i$", "Impedance eigenval."),
    ]

    n = len(correspondences)
    y_start = 0.82
    y_step = 0.17

    for i, (left, right) in enumerate(correspondences):
        y = y_start - i * y_step
        ax.text(0.15, y, left, fontsize=10, ha="center", va="center",
                color=_CLR_RG, fontweight="bold")
        ax.annotate(
            "", xy=(0.72, y), xytext=(0.30, y),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.5),
        )
        ax.text(0.87, y, right, fontsize=10, ha="center", va="center",
                color=_CLR_HYDRAULIC, fontweight="bold")

    # Separator label
    ax.text(
        0.50, y_start - n * y_step + 0.02,
        "net flow $= 0$",
        fontsize=9, ha="center", va="center", color="0.4", style="italic",
    )

    return fig
