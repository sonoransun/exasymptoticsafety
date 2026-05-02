"""Grover-domain plotting (success-probability curve, measurement heatmap)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from asymsafety.visualization.style import (
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_RELEVANT,
    CMAP_FLOW,
    add_colorbar,
    add_reference_box,
    apply_style,
    coupling_label,
    format_arxiv,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from asymsafety.quantum.grover.encoding import CouplingGridEncoding
    from asymsafety.quantum.grover.search import (
        GroverFixedPointSearch, GroverSearchResult,
    )


def plot_grover_success_probability(
    search: GroverFixedPointSearch,
    *,
    n_iter_max: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    r"""Curve of theoretical success probability vs Grover iteration count.

    Plots ``P(k) = sin^2((2k+1) * arcsin(sqrt(M/N)))`` over
    ``k = 0..n_iter_max``, with the optimal iteration count
    ``k_opt = floor(pi/4 * sqrt(N/M))`` highlighted.

    Args:
        search: A :class:`GroverFixedPointSearch` instance whose oracle
            has been pre-computed (so ``oracle.n_marked`` is known).
        n_iter_max: Upper bound on iteration count to plot. Defaults to
            ``2 * k_opt`` so the first oscillation is visible.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.get_figure()

    k_opt = search.optimal_iterations()
    if n_iter_max is None:
        n_iter_max = max(8, 2 * k_opt)

    iterations = np.arange(0, n_iter_max + 1)
    probs = np.array([
        search.success_probability_at(int(k)) for k in iterations
    ])

    ax.plot(iterations, probs, color=COLOR_RELEVANT, lw=2.0,
            marker="o", markersize=4, label=r"$P(k)$")
    if k_opt > 0 and k_opt <= n_iter_max:
        p_opt = search.success_probability_at(k_opt)
        ax.axvline(k_opt, color=COLOR_IRRELEVANT, lw=1.2, ls=":")
        ax.plot(k_opt, p_opt, marker="*", color=COLOR_NGFP,
                markersize=16, markeredgecolor="black",
                zorder=5, label=rf"$k_{{\mathrm{{opt}}}}={k_opt}$")

    ax.set_xlabel("Grover iterations $k$")
    ax.set_ylabel(r"$P(k) = \sin^2((2k+1)\theta)$")
    ax.set_title(r"Grover success probability vs iteration count")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Grover (1996)", "quant-ph/9605043")],
        loc="upper left",
    )
    return fig


def plot_grover_measurement_distribution(
    result: GroverSearchResult,
    encoding: CouplingGridEncoding,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Heatmap of Grover measurement counts over the discretised coupling grid.

    For 2-coupling encodings the counts reshape to a 2D heatmap
    ``(2^n_bits, 2^n_bits)``; otherwise the function falls back to a
    1D bar chart over the flat grid index. The refined NGFP location
    (when available) is overlaid as a green star.

    Args:
        result: The :class:`GroverSearchResult` from a completed search.
            Requires ``result.counts`` to be populated.
        encoding: The encoding used for the search (needed to map
            bitstrings back to coupling values).
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.get_figure()

    counts = result.counts or {}
    if not counts:
        ax.text(0.5, 0.5, "no measurement counts captured",
                transform=ax.transAxes, ha="center", va="center")
        return fig

    n_per_dim = 1 << encoding.n_bits
    coupling_names = encoding.coupling_names

    if len(coupling_names) == 2:
        grid = np.zeros((n_per_dim, n_per_dim), dtype=float)
        for bitstring, shots in counts.items():
            try:
                idx = encoding.couplings_to_index(
                    encoding.bitstring_to_couplings(bitstring)
                )
            except (ValueError, KeyError):
                continue
            i = idx % n_per_dim
            j = idx // n_per_dim
            grid[j, i] += shots

        x_name, y_name = coupling_names
        x_lo, x_hi = encoding._coupling_ranges[x_name]  # noqa: SLF001
        y_lo, y_hi = encoding._coupling_ranges[y_name]  # noqa: SLF001
        extent = (x_lo, x_hi, y_lo, y_hi)

        mesh = ax.imshow(
            grid, origin="lower", extent=extent, aspect="auto",
            cmap=CMAP_FLOW,
        )
        add_colorbar(fig, ax, mesh, label="shots")

        ax.set_xlabel(coupling_label(x_name))
        ax.set_ylabel(coupling_label(y_name))

        if result.refined_fixed_point is not None:
            fp = result.refined_fixed_point
            ax.plot(
                fp.location.get(x_name, np.nan),
                fp.location.get(y_name, np.nan),
                marker="*", color=COLOR_NGFP, markersize=16,
                markeredgecolor="black", zorder=5,
                label="refined NGFP",
            )
            ax.legend(loc="upper right", framealpha=0.85)
    else:
        keys, vals = zip(*sorted(counts.items(), key=lambda kv: -kv[1]))
        keys = list(keys)[:32]
        vals = np.array(list(vals)[:32], dtype=float)
        ax.bar(range(len(keys)), vals,
               color=COLOR_RELEVANT, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=80, fontsize=7)
        ax.set_ylabel("shots")

    ax.set_title("Grover measurement distribution")
    add_reference_box(
        ax,
        [format_arxiv("Grover (1996)", "quant-ph/9605043")],
        loc="lower left",
    )
    return fig


__all__ = [
    "plot_grover_success_probability",
    "plot_grover_measurement_distribution",
]
