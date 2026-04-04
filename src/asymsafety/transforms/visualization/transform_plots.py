"""Plotting utilities for transforms: Bode plots, scalograms, pseudospectra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from asymsafety.transforms._types import WaveletResult
    from asymsafety.transforms.linear.resolvent import ResolventOperator


def plot_bode(
    bode_data: dict,
    entry: tuple[int, int] = (0, 0),
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Bode plot (magnitude + phase) for a specific transfer function entry.

    Args:
        bode_data: Dict from :meth:`ImpedanceBridge.bode_data` with keys
            ``"omega"``, ``"magnitude"``, ``"phase"``.
        entry: ``(i, j)`` indices of the transfer function matrix.
        ax: Optional axes.  When *None*, a new two-subplot figure is
            created (magnitude on top, phase below).  When provided,
            magnitude is plotted on *ax* and phase on a twin y-axis.

    Returns:
        The matplotlib figure containing the plot.
    """
    i, j = entry

    if ax is None:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True
        )
    else:
        fig = ax.get_figure()
        ax_mag = ax
        ax_phase = ax.twinx()  # fallback if single axis

    omega = bode_data["omega"]
    mag = bode_data["magnitude"][i, j, :]
    phase = bode_data["phase"][i, j, :]

    ax_mag.semilogx(omega, mag)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_title(f"Bode Plot: H_{{{i}{j}}}(j\u03c9)")

    ax_phase.semilogx(omega, phase)
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel("Frequency \u03c9")
    ax_phase.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_scalogram(
    wavelet_result: WaveletResult,
    coupling_index: int = 0,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot wavelet scalogram |W(a,b)|^2.

    Args:
        wavelet_result: :class:`WaveletResult` from
            ``RGFlowWavelet.transform()``.
        coupling_index: Which coupling to plot.
        ax: Optional axes; a new figure is created when *None*.

    Returns:
        The matplotlib figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    energy = np.abs(wavelet_result.coefficients[coupling_index]) ** 2

    mesh = ax.pcolormesh(
        wavelet_result.positions,
        wavelet_result.scales,
        energy,
        shading="auto",
        cmap="viridis",
    )
    ax.set_ylabel("Scale a")
    ax.set_xlabel("RG time t")
    ax.set_yscale("log")
    ax.set_title(f"Wavelet Scalogram: coupling {coupling_index}")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|W(a,b)|\u00b2")

    return fig


def plot_pseudospectrum(
    resolvent_op: ResolventOperator,
    epsilon_values: list[float] | None = None,
    real_range: tuple[float, float] = (-5, 5),
    imag_range: tuple[float, float] = (-5, 5),
    n_grid: int = 100,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot pseudospectrum contours.

    Args:
        resolvent_op: :class:`ResolventOperator` instance.
        epsilon_values: Contour levels; defaults to ``[1.0, 0.1, 0.01]``.
        real_range: ``(re_min, re_max)`` bounds on the real axis.
        imag_range: ``(im_min, im_max)`` bounds on the imaginary axis.
        n_grid: Grid resolution per axis.
        ax: Optional axes; a new figure is created when *None*.

    Returns:
        The matplotlib figure containing the plot.
    """
    if epsilon_values is None:
        epsilon_values = [1.0, 0.1, 0.01]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    real_grid, imag_grid, norm_grid = resolvent_op.pseudospectrum(
        epsilon=min(epsilon_values),
        real_range=real_range,
        imag_range=imag_range,
        n_grid=n_grid,
    )

    # Plot log10(||R(s)||) as filled contour
    log_norm = np.log10(norm_grid + 1e-30)
    ax.contourf(real_grid, imag_grid, log_norm, levels=20, cmap="hot_r")

    # Plot epsilon-pseudospectrum boundaries
    for eps in epsilon_values:
        ax.contour(
            real_grid,
            imag_grid,
            norm_grid,
            levels=[1.0 / eps],
            colors="white",
            linewidths=1.5,
        )

    # Plot eigenvalues
    poles = resolvent_op.poles()
    ax.plot(poles.real, poles.imag, "wx", markersize=10, markeredgewidth=2)

    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("\u03b5-Pseudospectrum")
    ax.set_aspect("equal")

    return fig


def plot_comparison_table(
    table: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot critical exponents from all methods side-by-side as bar chart.

    Args:
        table: Dict from
            :meth:`CrossAnalogueBridge.full_comparison_table`, mapping
            method names to arrays of critical exponents (or *None*).
        ax: Optional axes; a new figure is created when *None*.

    Returns:
        The matplotlib figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    methods = [k for k, v in table.items() if v is not None]
    n_methods = len(methods)

    if n_methods == 0:
        return fig

    # Get max number of exponents
    n_exp = max(len(table[m]) for m in methods)

    x = np.arange(n_exp)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        vals = table[method]
        real_vals = np.real(vals)[:n_exp]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x[: len(real_vals)] + offset,
            real_vals,
            width,
            label=method,
            alpha=0.8,
        )

    ax.set_xlabel("Exponent index")
    ax.set_ylabel("Re(\u03b8)")
    ax.set_title("Critical Exponents: Cross-Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    return fig
