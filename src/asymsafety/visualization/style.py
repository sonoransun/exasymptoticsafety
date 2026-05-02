"""Centralized visual styling for all asymptotic safety plots.

Provides a unified color palette, font settings, and helper functions
so that every visualization in the package has a consistent look.

The module also exposes citation helpers (:func:`format_arxiv`,
:func:`add_reference_box`) and a proxy-artist legend builder
(:func:`add_legend_panel`) so that 2D and 3D plots can carry inline
literature provenance and structured legend explanations.

References
----------
- Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030] — the canonical
  Einstein-Hilbert NGFP whose coordinates seed many of the package's
  reference-box annotations.
- See ``docs/LITERATURE.md`` for the full topic-organised bibliography.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    "lambda_ADM": r"$\lambda_{\mathrm{ADM}}$",
    "n_s": r"$N_s$",
    "n_v": r"$N_v$",
    "n_d": r"$N_D$",
}


def coupling_label(name: str) -> str:
    """Return a LaTeX-formatted label for a coupling name.

    Falls back case-insensitively, then to a generic ``$<name>$``
    LaTeX rendering for unknown couplings.
    """
    if name in _COUPLING_LABELS:
        return _COUPLING_LABELS[name]
    if name.lower() in _COUPLING_LABELS:
        return _COUPLING_LABELS[name.lower()]
    return f"${name}$"


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
    paper_label: str | None = None,
) -> None:
    """Draw a marker and coordinate label at a fixed point.

    Args:
        ax: Matplotlib axes.
        fp: The fixed point to annotate.
        x_coupling: Coupling on the x-axis.
        y_coupling: Coupling on the y-axis.
        fontsize: Annotation text size.
        offset: (dx, dy) pixel offset for the annotation text.
        paper_label: Optional short citation appended below the
            coordinate label (e.g. ``"Reuter 1998"``). Renders inside
            the same annotation box on a new line.
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
    annotation_text = f"${coord_text}$"
    if paper_label:
        annotation_text = f"{annotation_text}\n{paper_label}"
    ax.annotate(
        annotation_text,
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


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

# Canonical short labels for the most-cited papers in the project.
# Keys are the arXiv ID; values are the "Author Year" short label.
CITATION_LABELS: dict[str, str] = {
    "hep-th/9605030": "Reuter (1998)",
    "hep-th/0108040": "Lauscher & Reuter (2002)",
    "hep-th/0103195": "Litim (2001)",
    "hep-th/0306138": "Vassilevich (2003)",
    "0812.0785": "Codello et al. (2009)",
    "1003.5129": "Manrique et al. (2011)",
    "1202.2274": "Reuter & Saueressig (2012)",
    "1311.2898": "Dona et al. (2014)",
    "1609.02803": "Biemans et al. (2017)",
    "2004.06810": "Bonanno et al. (2020)",
    "2212.07456": "Eichhorn & Schiffer (2022)",
    "2306.10408": "Knorr et al. (2023)",
    "2302.14152": "Saueressig (2023)",
    "2310.20603": "D'Angelo et al. (2024)",
    "2402.01260": "Korver et al. (2024)",
    "2412.13800": "Draper et al. (2025)",
    "2501.03752": "Saueressig et al. (2025)",
}


def format_arxiv(label: str, arxiv_id: str) -> str:
    """Format a citation as ``"Reuter (1998) [hep-th/9605030]"``.

    Matches the project's citation convention used in
    ``src/asymsafety/validation/*.py`` and the docstrings of
    ``src/asymsafety/beta/einstein_hilbert.py``.

    Args:
        label: Short ``"Author Year"`` label, e.g. ``"Reuter (1998)"``.
        arxiv_id: arXiv identifier, e.g. ``"hep-th/9605030"`` or
            ``"2402.01260"``.

    Returns:
        Concatenated citation string ``"<label> [<arxiv_id>]"``.
    """
    return f"{label} [{arxiv_id}]"


def add_reference_box(
    ax: Axes,
    refs: Sequence[str],
    *,
    loc: str = "lower right",
    fontsize: int = 7,
    title: str = "References",
    alpha: float = 0.85,
) -> None:
    """Draw a small footnote-style citation box on *ax*.

    Used to embed paper provenance directly inside scientific figures
    so they remain self-explanatory even when rendered in isolation
    (i.e. without README context).

    Args:
        ax: Matplotlib axes to annotate.
        refs: List of citation strings, typically produced by
            :func:`format_arxiv`. Each entry renders on its own line.
        loc: One of ``"lower right"``, ``"lower left"``,
            ``"upper right"``, ``"upper left"``. Determines the corner
            of *ax* that anchors the box.
        fontsize: Font size for the citation lines.
        title: Header label drawn in bold above the citations. Pass an
            empty string to suppress the header.
        alpha: Background opacity in ``[0, 1]``.
    """
    if not refs:
        return
    body = "\n".join(refs)
    text = f"{title}\n{body}" if title else body

    anchors = {
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
    }
    if loc not in anchors:
        loc = "lower right"
    x, y, ha, va = anchors[loc]

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha, va=va,
        color="0.25",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white", ec="0.7", lw=0.6, alpha=alpha,
        ),
        zorder=20,
    )


# ---------------------------------------------------------------------------
# Legend panel (proxy artists for 3D / mixed-encoding plots)
# ---------------------------------------------------------------------------


def add_legend_panel(
    ax: Axes,
    entries: Sequence[tuple[str, str]],
    *,
    loc: str = "upper left",
    fontsize: int = 9,
    title: str | None = None,
) -> None:
    """Build a proxy-artist legend on *ax*.

    A wrapper around ``ax.legend`` that takes a list of
    ``(kind, label)`` pairs and produces the appropriate matplotlib
    proxy artist for each one. This is needed for 3D axes
    (:class:`mpl_toolkits.mplot3d.Axes3D`) which do not support
    line-style legend entries via ``text2D`` and which often lose the
    mapping between artists and labels when arrows/quivers are drawn.

    Args:
        ax: Matplotlib axes (2D or 3D).
        entries: Sequence of ``(kind, label)`` pairs. ``kind`` is one of
            the strings below, optionally followed by a colour spec
            separated by ``"|"``:

            - ``"line"`` — solid line
            - ``"line:dashed"`` — dashed line
            - ``"marker_star"`` — green NGFP-style star
            - ``"marker_circle"`` — white GFP-style circle
            - ``"patch"`` — filled rectangle
            - ``"arrow"`` — solid line proxy with arrow-style colour

            Examples: ``("line|#2176AE", "relevant")``,
            ``("marker_star", "NGFP")``,
            ``("patch|#06D6A0", "UV-attractive basin")``.
        loc: Legend location passed to ``ax.legend``.
        fontsize: Legend text size.
        title: Optional legend title.
    """
    handles: list[object] = []
    labels: list[str] = []

    for kind_spec, label in entries:
        if "|" in kind_spec:
            kind, color = kind_spec.split("|", 1)
        else:
            kind, color = kind_spec, "0.3"

        if kind == "line":
            handles.append(Line2D([0], [0], color=color, lw=2.0))
        elif kind == "line:dashed":
            handles.append(
                Line2D([0], [0], color=color, lw=1.6, linestyle="--")
            )
        elif kind == "marker_star":
            handles.append(Line2D(
                [0], [0], marker="*", linestyle="none",
                color=color or COLOR_NGFP, markersize=12,
                markeredgecolor="black", markeredgewidth=0.8,
            ))
        elif kind == "marker_circle":
            handles.append(Line2D(
                [0], [0], marker="o", linestyle="none",
                color=color or COLOR_GFP, markersize=10,
                markeredgecolor="black", markeredgewidth=1.0,
            ))
        elif kind == "patch":
            handles.append(Patch(facecolor=color, edgecolor="0.4", alpha=0.45))
        elif kind == "arrow":
            handles.append(Line2D(
                [0], [0], color=color, lw=2.4, marker=">", markersize=6,
            ))
        else:
            handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(label)

    leg = ax.legend(
        handles, labels,
        loc=loc, fontsize=fontsize, title=title, framealpha=0.9,
    )
    if title:
        leg.get_title().set_fontsize(fontsize)


# ---------------------------------------------------------------------------
# Critical-exponent label formatting
# ---------------------------------------------------------------------------


def theta_label(theta: complex, *, precision: int = 2) -> str:
    """Format a critical exponent for inline display next to plot artists.

    Real exponents render as ``"θ=1.94"``; complex-conjugate-pair members
    render as ``"θ=1.94+3.15i"``. Used for arrow-tip labels on
    eigenvector quivers in :func:`asymsafety.gui.visualization_3d.fixed_point_stability_3d`
    and :func:`asymsafety.visualization.phase_portrait.annotated_eh_phase_portrait`.

    Args:
        theta: Critical exponent (complex). Imaginary parts smaller than
            ``1e-10`` are treated as zero.
        precision: Number of decimal places (default 2).

    Returns:
        LaTeX-ready label such as ``r"$\\theta=1.94+3.15i$"``.
    """
    re = float(theta.real)
    im = float(theta.imag)
    if abs(im) < 1e-10:
        return rf"$\theta={re:+.{precision}f}$"
    sign = "+" if im >= 0 else "-"
    return rf"$\theta={re:+.{precision}f}{sign}{abs(im):.{precision}f}i$"
