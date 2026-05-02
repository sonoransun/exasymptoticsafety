"""Koopman-domain plotting (spectrum, modes, QFT power spectrum).

The Koopman operator linearises the (nonlinear) RG flow on a lifted
observable space; its eigenvalues encode decay/growth rates of those
observables, and its modes describe how the observables project back
onto the original couplings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from asymsafety.visualization.style import (
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_RELEVANT,
    COLOR_SEPARATRIX,
    TRAJECTORY_COLORS,
    add_reference_box,
    apply_style,
    coupling_label,
    format_arxiv,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from asymsafety.quantum.koopman.qft_rg import QuantumFourierRG


def plot_koopman_spectrum(
    edmd_result: dict,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Scatter plot of Koopman eigenvalues on the complex plane.

    The unit circle is the stability boundary: eigenvalues inside
    correspond to decaying observables (irrelevant), eigenvalues outside
    to growing ones (relevant). Marker size encodes the magnitude of the
    associated mode in the original coupling basis.

    Args:
        edmd_result: Result of
            :meth:`asymsafety.quantum.koopman.operator.KoopmanOperator.compute_edmd`.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.get_figure()

    eigenvalues = np.atleast_1d(edmd_result.get("eigenvalues", []))
    modes = edmd_result.get("modes", np.zeros((0, 0)))
    if eigenvalues.size == 0:
        ax.text(0.5, 0.5, "no eigenvalues", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    if modes.size > 0:
        # modes shape: (n_couplings, dict_size) — pick the first
        # dict_size eigenvalues, weight by coupling-projection norm.
        n_take = min(modes.shape[1], eigenvalues.size)
        weights = np.linalg.norm(modes[:, :n_take], axis=0)
    else:
        n_take = eigenvalues.size
        weights = np.ones(n_take)
    eigenvalues = eigenvalues[:n_take]
    weights = weights / max(weights.max(), 1e-30)
    sizes = 30.0 + 250.0 * weights

    inside = np.abs(eigenvalues) < 1.0
    ax.scatter(
        eigenvalues.real[inside], eigenvalues.imag[inside],
        s=sizes[inside], c=COLOR_IRRELEVANT, alpha=0.85,
        edgecolor="black", linewidth=0.5, label="decaying",
    )
    ax.scatter(
        eigenvalues.real[~inside], eigenvalues.imag[~inside],
        s=sizes[~inside], c=COLOR_RELEVANT, alpha=0.85,
        edgecolor="black", linewidth=0.5, label="growing",
    )

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            color=COLOR_SEPARATRIX, lw=1.2, ls="--",
            label="unit circle")

    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.axvline(0.0, color="0.5", lw=0.6)

    max_extent = max(1.05, float(np.max(np.abs(eigenvalues))) * 1.1)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\mu)$")
    ax.set_ylabel(r"$\mathrm{Im}(\mu)$")
    ax.set_title("Koopman spectrum (EDMD)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)
    add_reference_box(
        ax,
        [
            "Williams, Kevrekidis & Rowley (2015) [1408.4408]",
            "Mezic (2020) [1906.07401]",
        ],
        loc="lower right",
    )
    return fig


def plot_koopman_modes(
    edmd_result: dict,
    coupling_names: list[str],
    *,
    top_k: int = 8,
    ax: Axes | None = None,
) -> Figure:
    """Bar chart of the top-``k`` Koopman modes' coupling support.

    Each bar group corresponds to a Koopman eigenfunction (sorted by
    ``|mu|`` from the EDMD output); within a group, one bar per coupling
    shows how strongly that eigenfunction loads onto that coupling.

    Args:
        edmd_result: Result of EDMD.
        coupling_names: Coupling names in the order used to build the
            EDMD dictionary.
        top_k: Number of leading modes to display.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
    else:
        fig = ax.get_figure()

    eigenvalues = np.atleast_1d(edmd_result.get("eigenvalues", []))
    modes = edmd_result.get("modes", np.zeros((0, 0)))
    if eigenvalues.size == 0 or modes.size == 0:
        ax.text(0.5, 0.5, "empty Koopman result", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    n_couplings = len(coupling_names)
    n_take = min(top_k, modes.shape[1], eigenvalues.size)
    abs_modes = np.abs(modes[:n_couplings, :n_take])

    x = np.arange(n_take)
    width = 0.8 / max(1, n_couplings)
    for ci in range(n_couplings):
        offset = (ci - (n_couplings - 1) / 2.0) * width
        ax.bar(
            x + offset, abs_modes[ci], width=width,
            color=TRAJECTORY_COLORS[ci % len(TRAJECTORY_COLORS)],
            edgecolor="black", linewidth=0.4,
            label=coupling_label(coupling_names[ci]),
        )

    eig_labels = [
        f"$|\\mu_{{{i}}}|={abs(e):.3f}$"
        for i, e in enumerate(eigenvalues[:n_take])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(eig_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\mathrm{mode}_{c,i}|$")
    ax.set_title("Koopman modes — top-{} by $|\\mu|$".format(n_take))
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25)
    add_reference_box(
        ax,
        ["Williams, Kevrekidis & Rowley (2015) [1408.4408]"],
        loc="upper left",
    )
    return fig


def plot_qft_power_spectrum(
    qft: QuantumFourierRG,
    coupling_name: str,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Stem-plot the QFT amplitude spectrum of a single running coupling.

    Calls :meth:`QuantumFourierRG.spectral_analysis` (qiskit-backed) on
    the named coupling and renders the result with peak frequencies
    highlighted.

    Args:
        qft: The :class:`QuantumFourierRG` instance.
        coupling_name: Coupling whose trajectory should be transformed.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.

    Raises:
        ImportError: Propagated from the underlying qiskit-dependent
            method when qiskit is not installed.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = ax.get_figure()

    result = qft.spectral_analysis(coupling_name)
    freqs = np.asarray(result["frequencies"])
    amps = np.asarray(result["amplitudes"])
    peaks = np.asarray(result.get("peak_frequencies", []), dtype=int)

    markerline, stemlines, baseline = ax.stem(
        freqs, amps, basefmt=" ",
    )
    plt.setp(stemlines, color=COLOR_RELEVANT, alpha=0.85, linewidth=1.2)
    plt.setp(markerline, color=COLOR_RELEVANT, markersize=4)

    if peaks.size > 0:
        ax.scatter(freqs[peaks], amps[peaks],
                   color=COLOR_NGFP, marker="*", s=130,
                   edgecolor="black", linewidth=0.8,
                   zorder=5, label="dominant peaks")
        ax.legend(loc="upper right", framealpha=0.85)

    ax.set_xlabel("frequency")
    ax.set_ylabel(rf"$|\hat g(\omega)|^2$ for {coupling_label(coupling_name)}")
    ax.set_title(rf"QFT spectrum of {coupling_label(coupling_name)}$(t)$")
    ax.grid(True, alpha=0.25)
    add_reference_box(
        ax,
        [format_arxiv("Reuter & Saueressig (2012)", "1202.2274")],
        loc="upper left",
    )
    return fig


__all__ = [
    "plot_koopman_spectrum",
    "plot_koopman_modes",
    "plot_qft_power_spectrum",
]
