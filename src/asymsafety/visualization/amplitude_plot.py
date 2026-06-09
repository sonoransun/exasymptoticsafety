"""Plots for graviton-mediated scattering amplitudes in asymptotic safety.

Companion to :mod:`asymsafety.scattering`.  All functions return a
:class:`matplotlib.figure.Figure` and use the shared house style
(:mod:`asymsafety.visualization.style`).

* :func:`plot_amplitude_vs_energy` — |M| vs √s: classical GR grows, the
  asymptotically-safe amplitude plateaus (UV-finite).
* :func:`plot_form_factor` — the graviton form factor ``f(p²)`` and the
  softening ``G(p²)/G_N``.
* :func:`plot_partial_wave_unitarity` — ``|a_ℓ(s)|``: GR runs past the
  ``|a_ℓ|=1`` line, AS stays bounded.
* :func:`plot_regge_trajectory` — Chew–Frautschi plot of the string tower.
* :func:`plot_as_vs_string` — normalised high-energy comparison:
  UV-constant (AS) vs ultrasoft (string).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from asymsafety.visualization.style import (
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_RELEVANT,
    COLOR_SINGULARITY,
    add_reference_box,
    apply_style,
)

_DRAPER_REF = "Draper, Knorr, Ripken & Saueressig (2020) [2007.04396]"
_KNORR_REF = "Knorr (2026) [2602.21285]"
_CHEUNG_REF = "Cheung, Remmen, Sciotti & Tarquini (2025) [2508.09246]"


def plot_amplitude_vs_energy(
    amplitude,
    *,
    cos_theta: float = 0.3,
    s_range: tuple[float, float, int] = (1e-2, 1e8, 200),
    figsize: tuple[float, float] = (8.0, 5.0),
    show_references: bool = True,
) -> Figure:
    """|M(√s)| at fixed angle for the AS (dressed) and GR (undressed) amplitudes."""
    apply_style()
    lo, hi, n = s_range
    s = np.geomspace(lo, hi, n)
    sqrt_s = np.sqrt(s)
    M_as = np.abs(amplitude.amplitude_vs_s(s, cos_theta, dressed=True))
    M_gr = np.abs(amplitude.amplitude_vs_s(s, cos_theta, dressed=False))

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(sqrt_s, M_gr, color=COLOR_SINGULARITY, lw=2.0,
              label="classical GR (undressed)")
    ax.loglog(sqrt_s, M_as, color=COLOR_NGFP, lw=2.0,
              label="asymptotic safety (dressed)")
    ax.axvline(1.0, color="0.6", ls=":", lw=1.0)
    ax.text(1.05, ax.get_ylim()[0] * 2, r"$k_0$ (Planck)", fontsize=8,
            color="0.4", rotation=90, va="bottom")
    ax.set_xlabel(r"$\sqrt{s}\,/\,k_0$")
    ax.set_ylabel(r"$|\mathcal{M}(s,\cos\theta)|$")
    ax.set_title("Graviton-mediated scalar scattering: GR vs asymptotic safety")
    ax.legend(loc="upper left", frameon=True)
    if show_references:
        add_reference_box(ax, [_DRAPER_REF])
    fig.tight_layout()
    return fig


def plot_form_factor(
    form_factor,
    *,
    p2_range: tuple[float, float, int] = (1e-3, 1e6, 200),
    figsize: tuple[float, float] = (8.0, 5.0),
    show_references: bool = True,
) -> Figure:
    """Form factor ``f(p²)`` and the softening ``G(p²)/G_N`` vs ``p²``."""
    apply_style()
    lo, hi, n = p2_range
    p2 = np.geomspace(lo, hi, n)
    f = np.asarray(form_factor.f(p2), dtype=float)
    G = np.asarray(form_factor.G_of_psq(p2), dtype=float)
    G_N = form_factor.newton_constant()

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(p2, f, color=COLOR_RELEVANT, lw=2.0, label=r"form factor $f(p^2)$")
    ax.loglog(p2, G / G_N, color=COLOR_IRRELEVANT, lw=2.0,
              label=r"$G(p^2)/G_N$")
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"$p^2 / k_0^2$")
    ax.set_ylabel("dimensionless dressing")
    ax.set_title(r"Graviton form factor: $f\propto p^2$, $G\propto 1/p^2$ in the UV")
    ax.legend(loc="center left", frameon=True)
    if show_references:
        add_reference_box(ax, [_DRAPER_REF, _KNORR_REF])
    fig.tight_layout()
    return fig


def plot_partial_wave_unitarity(
    amplitude,
    *,
    ell: int = 0,
    s_range: tuple[float, float, int] = (1e-2, 1e6, 50),
    figsize: tuple[float, float] = (8.0, 5.0),
    show_references: bool = True,
) -> Figure:
    """``|a_ℓ(s)|`` for AS vs GR with the ``|a_ℓ| = 1`` unitarity line.

    Note the *absolute* normalisation of both curves depends on the
    trajectory's IR Newton constant and on the forward/backward angular
    cutoff ``cos_max`` applied to the massless graviton ``t``/``u``
    poles (see :func:`asymsafety.scattering.consistency.partial_wave`);
    the scheme-robust statement is GR's unbounded ``∝ s`` growth versus
    the finite AS plateau, not the literal elastic bound.
    """
    from asymsafety.scattering import consistency as C

    apply_style()
    lo, hi, n = s_range
    s = np.geomspace(lo, hi, n)
    a_as = np.array([abs(C.partial_wave(amplitude, ell, float(x), dressed=True))
                     for x in s])
    a_gr = np.array([abs(C.partial_wave(amplitude, ell, float(x), dressed=False))
                     for x in s])

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(np.sqrt(s), a_gr, color=COLOR_SINGULARITY, lw=2.0,
              label="classical GR")
    ax.loglog(np.sqrt(s), a_as, color=COLOR_NGFP, lw=2.0,
              label="asymptotic safety")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0, label=r"$|a_\ell|=1$")
    # Mark the Planck scale of the underlying trajectory when it sits
    # inside the plotted window (in k0 units, M_Pl = G_N^{-1/2}).
    G_N = float(amplitude.form_factor.newton_constant())
    if G_N > 0:
        m_pl = G_N ** -0.5
        if np.sqrt(lo) < m_pl < np.sqrt(hi):
            ax.axvline(m_pl, color="0.6", ls=":", lw=1.2,
                       label=r"$M_{\mathrm{Pl}} = G_N^{-1/2}$")
    ax.set_xlabel(r"$\sqrt{s}\,/\,k_0$")
    ax.set_ylabel(rf"$|a_{{{ell}}}(s)|$")
    ax.set_title("Partial-wave growth: GR is unbounded, asymptotic safety is not")
    ax.legend(loc="upper left", frameon=True)
    if show_references:
        add_reference_box(ax, [_KNORR_REF])
    fig.tight_layout()
    return fig


def plot_regge_trajectory(
    *,
    alpha0: float = 1.0,
    alphap: float = 1.0,
    n_max: int = 5,
    figsize: tuple[float, float] = (8.0, 5.0),
    show_references: bool = True,
) -> Figure:
    """Chew–Frautschi plot: spin ``J = α(m²)`` vs ``m²`` with the string tower."""
    from asymsafety.scattering import bootstrap as B

    apply_style()
    spec = B.mass_spectrum(n_max, alpha0=alpha0, alphap=alphap)
    spins = np.arange(n_max + 1)
    m2_line = np.linspace(spec[0] - 0.5, spec[-1] + 0.5, 100)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(m2_line, B.regge_trajectory(m2_line, alpha0, alphap),
            color=COLOR_RELEVANT, lw=2.0, label=r"$\alpha(m^2)=\alpha_0+\alpha' m^2$")
    ax.scatter(spec, spins, color=COLOR_NGFP, s=70, zorder=5, edgecolor="k",
               label="string states")
    for m2, J in zip(spec, spins):
        ax.annotate(f"$J={J}$", (m2, J), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_xlabel(r"$m^2\,/\,(\alpha')^{-1}$")
    ax.set_ylabel(r"spin $J$")
    ax.set_title("Regge trajectory and mass spectrum (Strings from Almost Nothing)")
    ax.legend(loc="upper left", frameon=True)
    if show_references:
        add_reference_box(ax, [_CHEUNG_REF])
    fig.tight_layout()
    return fig


def plot_as_vs_string(
    bridge,
    *,
    cos_theta: float = 0.3,
    s_range: tuple[float, float, int] = (5.0, 60.0, 80),
    figsize: tuple[float, float] = (8.0, 5.0),
    show_references: bool = True,
) -> Figure:
    """Normalised high-energy comparison: UV-constant (AS) vs ultrasoft (string)."""
    apply_style()
    lo, hi, n = s_range
    s = np.geomspace(lo, hi, n)
    M_as = np.abs(bridge.as_amplitude.amplitude_vs_s(s, cos_theta, dressed=True))
    M_str = np.abs(bridge.string_amplitude.amplitude_vs_s(s, cos_theta))
    # Normalise each curve to its low-energy value to compare shapes.
    M_as = M_as / M_as[0]
    M_str = M_str / M_str[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(s, M_as, color=COLOR_NGFP, lw=2.0,
              label="asymptotic safety (UV-constant)")
    ax.loglog(s, M_str, color=COLOR_RELEVANT, lw=2.0,
              label="string bootstrap (ultrasoft)")
    ax.set_xlabel(r"$s\,/\,k_0^2$")
    ax.set_ylabel(r"$|\mathcal{M}(s)| / |\mathcal{M}(s_0)|$")
    ax.set_title("Two routes to UV completeness: softening vs an infinite tower")
    ax.legend(loc="lower left", frameon=True)
    if show_references:
        add_reference_box(ax, [_CHEUNG_REF, _DRAPER_REF])
    fig.tight_layout()
    return fig
