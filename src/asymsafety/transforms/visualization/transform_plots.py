"""Plotting utilities for transforms: Bode plots, scalograms, pseudospectra.

These plots accompany the integral- and linear-transform domain of the
cross-analogue bridge (see
:class:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`).
Each transform exposes a different facet of RG flow:

- Bode plots → frequency response of the impedance bridge.
- Scalograms → wavelet localisation of the running couplings in
  time-scale space.
- Pseudospectra → robustness of the spectrum (eigenvalues are critical
  exponents) under perturbations.
- Cross-method comparison → critical exponents from every analogue
  side-by-side; rows must agree at the NGFP for the bridge to commute.

References
----------
- See README "Physical Computational Analogues".
- ``docs/LITERATURE.md`` § resolvent / pseudospectra references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from asymsafety.visualization.style import (
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
)

if TYPE_CHECKING:
    from asymsafety.transforms._types import WaveletResult
    from asymsafety.transforms.linear.resolvent import ResolventOperator


def plot_bode(
    bode_data: dict,
    entry: tuple[int, int] = (0, 0),
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    r"""Bode plot (magnitude + phase) for a transfer-function entry.

    Physics
    -------
    The impedance bridge maps the linearised RG flow at a fixed point
    to a multi-input multi-output transfer function ``H(s)``. Its
    poles are the eigenvalues of the stability matrix; thus poles in
    the right half-plane indicate UV-attractive (relevant) directions.
    The magnitude / phase plot at logarithmic frequency renders the
    impedance directly.

    Parameters
    ----------
    bode_data : dict
        Result of :meth:`ImpedanceBridge.bode_data` with keys
        ``"omega"``, ``"magnitude"``, ``"phase"``.
    entry : tuple, default ``(0, 0)``
        Matrix entry ``(i, j)``.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to draw magnitude on; phase goes on a twin axis when
        provided. When ``None``, a new two-row subplot is created.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :class:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`
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

    ax_mag.semilogx(omega, mag, color=COLOR_RELEVANT, lw=1.8)
    ax_mag.set_ylabel("Magnitude (dB)", fontsize=11)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title(rf"Bode plot of $H_{{{i}{j}}}(j\omega)$",
                     fontsize=13, fontweight="bold")

    ax_phase.semilogx(omega, phase, color=COLOR_IRRELEVANT, lw=1.8)
    ax_phase.set_ylabel("Phase (degrees)", fontsize=11)
    ax_phase.set_xlabel(
        r"Frequency $\omega$ (in units of the reference RG scale $k_{\rm ref}$)",
        fontsize=11,
    )
    ax_phase.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_scalogram(
    wavelet_result: WaveletResult,
    coupling_index: int = 0,
    ax: matplotlib.axes.Axes | None = None,
    *,
    coupling_name: str | None = None,
) -> matplotlib.figure.Figure:
    r"""Plot wavelet scalogram :math:`|W(a, b)|^2` of a running coupling.

    Physics
    -------
    A wavelet decomposition localises the running coupling in both
    RG-time ``b`` and scale ``a``. Sharp horizontal bands at small
    scales reflect rapid crossover regimes; bright tails at large
    scales mark the asymptotic plateau toward the fixed point.

    Parameters
    ----------
    wavelet_result : :class:`~asymsafety.transforms._types.WaveletResult`
    coupling_index : int, default ``0``
    ax : :class:`matplotlib.axes.Axes`, optional

    Returns
    -------
    matplotlib.figure.Figure
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
    ax.set_ylabel(r"Scale $a$", fontsize=11)
    ax.set_xlabel(r"RG time $t = \log(k/k_0)$", fontsize=11)
    ax.set_yscale("log")
    label = coupling_name if coupling_name else f"coupling #{coupling_index}"
    ax.set_title(rf"Wavelet scalogram of ${label}(t)$",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$|W(a, t)|^2$", fontsize=11)

    return fig


def plot_pseudospectrum(
    resolvent_op: ResolventOperator,
    epsilon_values: list[float] | None = None,
    real_range: tuple[float, float] = (-5, 5),
    imag_range: tuple[float, float] = (-5, 5),
    n_grid: int = 100,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    r"""Plot the resolvent's :math:`\varepsilon`-pseudospectrum.

    Physics
    -------
    The pseudospectrum
    ``Lambda_eps(M) = { z : ||(zI - M)^{-1}|| > 1/eps }`` is the set of
    spectral values that perturbations of size ``eps`` can produce.
    Thin pseudospectral contours indicate well-conditioned eigenvalues
    (robust critical exponents); fat contours indicate that small
    truncation / regulator changes can shift exponents substantially.

    Parameters
    ----------
    resolvent_op : :class:`~asymsafety.transforms.linear.resolvent.ResolventOperator`
    epsilon_values : list of float, optional
        Contour levels. Defaults to ``[1.0, 0.1, 0.01]``.
    real_range, imag_range : tuple
    n_grid : int, default ``100``
    ax : :class:`matplotlib.axes.Axes`, optional

    Returns
    -------
    matplotlib.figure.Figure
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
    cf = ax.contourf(real_grid, imag_grid, log_norm, levels=20, cmap="hot_r")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$\log_{10}\,\|(sI-M)^{-1}\|$", fontsize=10)

    # Plot epsilon-pseudospectrum boundaries
    for eps in epsilon_values:
        contours = ax.contour(
            real_grid,
            imag_grid,
            norm_grid,
            levels=[1.0 / eps],
            colors="white",
            linewidths=1.5,
        )
        ax.clabel(contours, fmt={1.0 / eps: rf"$\varepsilon={eps}$"},
                  fontsize=8, colors="white")

    # Plot eigenvalues
    poles = resolvent_op.poles()
    ax.plot(poles.real, poles.imag, "wx", markersize=10, markeredgewidth=2,
            label="eigenvalues")

    ax.set_xlabel(r"Re($s$)", fontsize=11)
    ax.set_ylabel(r"Im($s$)", fontsize=11)
    ax.set_title(r"$\varepsilon$-pseudospectrum of $M = \partial\beta/\partial g$",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.85)

    return fig


def plot_comparison_table(
    table: dict,
    ax: matplotlib.axes.Axes | None = None,
    *,
    include_hydraulic: bool = False,
) -> matplotlib.figure.Figure:
    r"""Cross-method bar chart of critical exponents.

    Physics
    -------
    Side-by-side comparison of ``Re(theta_i)`` computed via different
    analogues (classical RG, Koopman / quantum, hydraulic / impedance,
    integral transforms). The cross-analogue bridge requires that all
    methods agree at the NGFP — the bar groups should align within
    numerical tolerance, with the relevant / irrelevant shaded bands
    indicating the sign of each exponent.

    Parameters
    ----------
    table : dict
        Output of
        :meth:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge.full_comparison_table`,
        mapping method names to arrays of critical exponents.
    ax : :class:`matplotlib.axes.Axes`, optional

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :class:`asymsafety.transforms.bridge.cross_analogue.CrossAnalogueBridge`
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    # Hydraulic path returns impedance eigenvalues of the pipe network,
    # not RG critical exponents directly \u2014 different dimensionality and
    # order of magnitude. Excluded by default; opt in with
    # ``include_hydraulic=True``. Open issue: see
    # CrossAnalogueBridge.hydraulic_exponents.
    methods = [
        k for k, v in table.items()
        if v is not None and (include_hydraulic or k != "hydraulic")
    ]
    n_methods = len(methods)

    if n_methods == 0:
        return fig

    # Length-align bar groups to the longest non-hydraulic array so
    # the x-indices correspond to the same RG critical exponents.
    rg_methods = [m for m in methods if m != "hydraulic"]
    n_exp = max(len(table[m]) for m in (rg_methods or methods))

    x = np.arange(n_exp)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        real_vals = np.real(table[method])[:n_exp]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x[: len(real_vals)] + offset,
            real_vals,
            width,
            label=method.replace("_", " "),
            alpha=0.85,
        )

    ax.axhline(y=0, color="0.5", ls="--", lw=1)

    ax.set_xlabel("Exponent index", fontsize=11)
    ax.set_ylabel(r"Re($\theta_i$)", fontsize=11)
    ax.set_title("Critical exponents: cross-method comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\theta_{{{j+1}}}$" for j in range(n_exp)])
    ax.legend(loc="best", framealpha=0.92)
    ax.grid(True, alpha=0.3, axis="y")

    # Shaded bands for relevant (above 0) / irrelevant (below 0)
    y_lo, y_hi = ax.get_ylim()
    ax.axhspan(0, y_hi, color=COLOR_RELEVANT, alpha=0.06)
    ax.axhspan(y_lo, 0, color=COLOR_IRRELEVANT, alpha=0.06)
    ax.set_ylim(y_lo, y_hi)

    return fig
