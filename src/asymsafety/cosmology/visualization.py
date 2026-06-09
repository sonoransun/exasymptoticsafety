"""Plotting utilities for the RG-improved cosmology classes.

Six plotting functions exposing the asymptotic-safety predictions for
black-hole geometry and FLRW cosmology in the canonical visualization
style. Every function accepts an optional ``ax`` and returns the
``Figure`` (or new figure) so the plots compose freely with
:mod:`asymsafety.visualization.style` helpers.

Sub-domain coverage
-------------------
- :func:`plot_running_newton_constant` — running Newton constant
  ``G(r)`` on a Schwarzschild background.
- :func:`plot_lapse_with_horizons` — RG-improved lapse function with
  Cauchy and event horizons annotated.
- :func:`plot_classical_vs_rg_lapse` — side-by-side classical vs. RG
  lapse for several masses, illustrating the critical-mass remnant.
- :func:`plot_hawking_temperature` — Hawking temperature ``T_H(M)``
  with the asymptotic-safety zero at ``M_crit``.
- :func:`plot_scale_identification` — overlay of ``k(r)`` for the
  built-in :class:`ScaleIdentification` prescriptions.
- :func:`plot_flrw_evolution` — three-panel ``a(t)``, ``H(t)``,
  running couplings for an RG-improved FLRW universe.

References
----------
- Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196].
- Bonanno & Reuter (2002), Phys. Rev. D 65, 043508 [hep-th/0106133].
- Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274].
- Platania (2023), in *Handbook of Quantum Gravity* [2302.04272].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

from asymsafety.visualization.style import (
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_RELEVANT,
    COLOR_SEPARATRIX,
    COLOR_TRAJECTORY,
    TRAJECTORY_COLORS,
    add_legend_panel,
    add_reference_box,
    apply_style,
    format_arxiv,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
    from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW
    from asymsafety.cosmology.scale_identification import ScaleIdentification


# ---------------------------------------------------------------------------
# Black-hole plots
# ---------------------------------------------------------------------------


def plot_running_newton_constant(
    bh: RGImprovedSchwarzschild,
    *,
    r_range: tuple[float, float] = (1e-3, 50.0),
    n_points: int = 400,
    ax: Axes | None = None,
) -> Figure:
    """Log-log plot of the running Newton constant ``G(r)``.

    Compares the RG-improved ``G(r) = g(k(r)) / k(r)^2`` with a constant
    classical reference. The asymptotic-safety prediction is a softening
    of ``G`` toward the origin: ``G(r) -> 0`` as ``r -> 0`` once the
    fixed-point regime is reached.

    Args:
        bh: An :class:`RGImprovedSchwarzschild` instance.
        r_range: ``(r_min, r_max)`` radial window in geometric units.
        n_points: Number of log-spaced sample points.
        ax: Optional existing axes; new figure created if ``None``.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.get_figure()

    r = np.geomspace(r_range[0], r_range[1], n_points)
    G_r = np.atleast_1d(bh.G(r))

    # Classical reference: G_N at the IR end of the trajectory.
    G_N = float(np.max(G_r))

    ax.loglog(r, G_r, color=COLOR_RELEVANT, lw=2.0, label=r"$G(r)$ (RG-improved)")
    ax.axhline(G_N, color=COLOR_TRAJECTORY, lw=1.4, ls="--",
               label=r"$G_N$ (classical, IR limit)")

    # Mark the Planck length r_Pl ~ sqrt(G_N) (units G_N=1 -> r_Pl=1)
    r_pl = float(np.sqrt(G_N))
    if r_range[0] < r_pl < r_range[1]:
        ax.axvline(r_pl, color=COLOR_SEPARATRIX, lw=1.0, ls=":", alpha=0.7)
        ax.text(r_pl, ax.get_ylim()[0], r"$r_{\mathrm{Pl}}$",
                rotation=90, va="bottom", ha="right",
                color=COLOR_SEPARATRIX, fontsize=10)

    ax.set_xlabel(r"radius $r$ (geometric units)")
    ax.set_ylabel(r"Newton constant $G(r)$")
    ax.set_title(r"RG-improved Newton constant $G(r)$ on Schwarzschild")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Bonanno & Reuter (2000)", "hep-th/0002196")],
        loc="upper right",
    )
    return fig


def plot_lapse_with_horizons(
    bh: RGImprovedSchwarzschild,
    *,
    r_range: tuple[float, float] = (1e-3, 5.0),
    n_points: int = 600,
    y_clip: tuple[float, float] = (-2.0, 1.3),
    ax: Axes | None = None,
) -> Figure:
    """Semi-log-x plot of the lapse ``f(r)`` with horizons annotated.

    Vertical lines mark each root of ``f(r) = 0``: the Cauchy horizon
    ``r_-`` (inner) and event horizon ``r_+`` (outer) when ``M`` is
    above the critical mass; they merge into a single extremal horizon
    at ``M = M_crit``; no horizon at all for ``M < M_crit``.

    The default ``r_range`` covers the regime where the Bonanno-Reuter
    improvement acts (the softened core through a few Schwarzschild
    radii past the outer horizon). The trajectory should reach a
    classical IR regime ``g ≈ G_N k²`` so that ``G(r) = g/k²`` is
    constant at large ``r`` (see
    ``scripts/generate_figures._eh_trajectory_for_cosmology`` for the
    IR-upward construction); ``y_clip`` merely constrains the y-axis
    so a deep inter-horizon dip does not dominate the frame.

    When ``f(r)`` has no root *inside the plotted window*, the function
    re-searches a wider radial range before declaring the geometry a
    sub-critical naked-mass remnant, so the in-figure label is only
    drawn when no horizon exists at all.

    Args:
        bh: An :class:`RGImprovedSchwarzschild` instance.
        r_range: ``(r_min, r_max)`` radial window — default ``(1e-3, 5)``.
        n_points: Number of log-spaced sample points.
        y_clip: ``(y_min, y_max)`` lapse-axis window.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = ax.get_figure()

    r = np.geomspace(r_range[0], r_range[1], n_points)
    f = np.atleast_1d(bh.lapse(r))

    ax.semilogx(r, f, color=COLOR_RELEVANT, lw=2.0,
                label=rf"$f(r),\ M={bh.M:g}$")
    ax.axhline(0.0, color="0.3", lw=1.0)

    horizons = sorted(bh.horizons(r_min=r_range[0], r_max=r_range[1]))
    # Standard Bonanno-Reuter labelling: inner = Cauchy r_-, outer = event r_+.
    if len(horizons) >= 2:
        horizon_labels = [r"$r_{-}$ (Cauchy)", r"$r_{+}$ (event)"]
    elif len(horizons) == 1:
        horizon_labels = [r"$r_{+}$ (event)"]
    else:
        horizon_labels = []
    if len(horizons) > 2:
        horizon_labels += [r"$r_{\mathrm{cosmo}}$"]
    for idx, r_h in enumerate(horizons):
        label = horizon_labels[idx] if idx < len(horizon_labels) else None
        ax.axvline(r_h, color=COLOR_SEPARATRIX, lw=1.4, ls="--", alpha=0.85)
        if label is not None:
            ax.annotate(
                label, xy=(r_h, 0.0), xytext=(0, 14),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
                color=COLOR_SEPARATRIX, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="0.7", alpha=0.85),
            )

    if not horizons:
        # Distinguish a genuinely horizonless (sub-critical) geometry
        # from horizons that merely sit outside the plotted window.
        wide = bh.horizons(
            r_min=r_range[0], r_max=max(100.0, 10.0 * r_range[1]),
        )
        if wide:
            note = (
                "no horizon inside plotted window\n"
                f"(roots at r = {', '.join(f'{w:.3g}' for w in wide)})"
            )
        else:
            note = (
                "naked-mass remnant\n"
                r"$M < M_{\mathrm{crit}}$: no horizon"
            )
        ax.text(
            0.5, 0.5, note,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.3",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="0.7", alpha=0.85),
        )

    # Annotate the softened core: f(r) -> 1 (linearly) as r -> 0.
    # NOT a regular de Sitter core — with the k = xi/r identification
    # the curvature still diverges (mildly); see the module docstring
    # of :mod:`asymsafety.cosmology.rg_improved_bh`.
    f_origin = float(np.atleast_1d(bh.lapse(r_range[0]))[0])
    if abs(f_origin - 1.0) < 0.5:
        ax.annotate(
            "softened core: $f \\to 1$ (linear,\nsingularity milder, "
            "not removed)",
            xy=(r_range[0], f_origin), xytext=(48, -30),
            textcoords="offset points", fontsize=9, color="0.35",
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8),
        )

    ax.set_xlabel(r"radius $r$ (geometric units)")
    ax.set_ylabel(r"lapse $f(r) = -g_{tt}$")
    ax.set_title(r"RG-improved Schwarzschild lapse")
    ax.set_ylim(*y_clip)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.85)
    add_reference_box(
        ax,
        [
            format_arxiv("Bonanno & Reuter (2000)", "hep-th/0002196"),
            format_arxiv("Platania (2023)", "2302.04272"),
        ],
        loc="upper right",
    )
    return fig


def _estimate_G_newton(
    bh: RGImprovedSchwarzschild,
    r_probe: tuple[float, float] = (1e-2, 100.0),
    n_probe: int = 128,
) -> float:
    """Estimate the IR Newton constant of the trajectory behind ``bh``.

    Takes the maximum of ``G(r)`` over a log-spaced radial probe — for
    a trajectory with classical IR scaling ``g = G_N k²``, ``G(r)``
    increases monotonically toward the IR and saturates at ``G_N``.
    """
    r = np.geomspace(r_probe[0], r_probe[1], n_probe)
    return float(np.max(np.atleast_1d(bh.G(r))))


def plot_classical_vs_rg_lapse(
    bh: RGImprovedSchwarzschild,
    *,
    M_values: Sequence[float] = (0.5, 1.0, 2.0),
    r_range: tuple[float, float] = (1e-3, 30.0),
    n_points: int = 500,
    G_N: float | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Overlay RG-improved vs. classical Schwarzschild lapse for several masses.

    For each ``M``, draws the classical lapse ``1 - 2 G_N M / r``
    (dashed, with ``G_N`` the trajectory's IR Newton constant) and the
    RG-improved lapse (solid). Visualises how the asymptotic-safety
    correction lifts the singularity and produces the ``M_crit``
    remnant.

    The bound BH instance is mutated during evaluation and restored on
    exit (try/finally guarantee), so the call is non-destructive.

    Args:
        bh: An :class:`RGImprovedSchwarzschild` instance.
        M_values: Sequence of ADM masses to plot.
        r_range: ``(r_min, r_max)`` radial window.
        n_points: Number of log-spaced sample points.
        G_N: IR Newton constant used for the classical reference
            curves. ``None`` (default) estimates it from the
            trajectory via :func:`_estimate_G_newton`.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig = ax.get_figure()

    if G_N is None:
        G_N = _estimate_G_newton(bh)

    r = np.geomspace(r_range[0], r_range[1], n_points)
    original_M = bh.M

    palette = TRAJECTORY_COLORS
    try:
        for i, M in enumerate(M_values):
            color = palette[i % len(palette)]
            bh.M = float(M)
            f_rg = np.atleast_1d(bh.lapse(r))
            f_cl = 1.0 - 2.0 * G_N * M / r
            ax.plot(r, f_rg, color=color, lw=2.0,
                    label=rf"RG, $M={M:g}$")
            ax.plot(r, f_cl, color=color, lw=1.2, ls="--", alpha=0.65,
                    label=rf"classical, $M={M:g}$")
    finally:
        bh.M = original_M

    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"radius $r$ (geometric units)")
    ax.set_ylabel(r"lapse $f(r)$")
    ax.set_title("RG-improved vs. classical Schwarzschild lapse")
    ax.set_ylim(-1.0, 1.4)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85, ncol=2)
    add_reference_box(
        ax,
        [format_arxiv("Bonanno & Reuter (2000)", "hep-th/0002196")],
        loc="upper left",
    )
    return fig


def plot_hawking_temperature(
    bh: RGImprovedSchwarzschild,
    *,
    M_range: tuple[float, float] = (0.5, 5.0),
    n_masses: int = 60,
    G_N: float | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Hawking temperature ``T_H(M)`` of the RG-improved black hole.

    Computes the surface gravity ``T_H = |f'(r_+)| / (4 pi)``
    numerically via central differences on the lapse at the outer
    horizon (the absolute value guards against numerical sign flips
    near extremality). For a trajectory with a classical IR regime
    (``G(r) -> G_N`` at large ``r``) the canonical Bonanno-Reuter
    shape emerges: ``T_H`` tracks the classical ``1/(8 pi G_N M)``
    curve at large ``M``, peaks at a few times the critical mass, and
    drops to zero at ``M_crit`` where the inner and outer horizons
    merge (extremal remnant) — in contrast with the classical
    divergence as ``M -> 0``. No point is drawn for masses without a
    horizon (``M < M_crit``).

    Args:
        bh: An :class:`RGImprovedSchwarzschild` instance.
        M_range: ``(M_min, M_max)`` mass sweep.
        n_masses: Number of sample masses (linear).
        G_N: IR Newton constant used for the classical reference
            curve ``1/(8 pi G_N M)``. ``None`` (default) estimates it
            from the trajectory via :func:`_estimate_G_newton`.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.get_figure()

    if G_N is None:
        G_N = _estimate_G_newton(bh)

    M_values = np.linspace(M_range[0], M_range[1], n_masses)
    T_rg = np.full(n_masses, np.nan)
    T_cl = 1.0 / (8.0 * np.pi * G_N * np.maximum(M_values, 1e-12))

    # The Hawking temperature at the *outer event* horizon is positive by
    # definition; surface gravity ``κ = |f'(r_+)| / 2`` with ``T_H = κ /
    # (2π)``. The absolute value guards against the central difference
    # flipping sign from finite-step noise close to extremality (where
    # ``f'(r_+) -> 0``).
    original_M = bh.M
    try:
        for i, M in enumerate(M_values):
            bh.M = float(M)
            roots = bh.horizons()
            if not roots:
                continue
            # Outer event horizon = max real root
            r_plus = max(roots)
            dr = max(1e-4, 1e-3 * r_plus)
            f_plus = float(np.atleast_1d(bh.lapse(r_plus + dr))[0])
            f_minus = float(np.atleast_1d(bh.lapse(r_plus - dr))[0])
            kappa = abs(f_plus - f_minus) / (2.0 * dr)
            T_rg[i] = kappa / (4.0 * np.pi)
    finally:
        bh.M = original_M

    ax.plot(M_values, T_cl, color=COLOR_TRAJECTORY, lw=1.4, ls="--",
            label=r"classical $1/(8\pi G_N M)$")
    ax.plot(M_values, T_rg, color=COLOR_RELEVANT, lw=2.2,
            label="RG-improved $T_H$")

    valid = ~np.isnan(T_rg)
    if valid.any():
        peak_idx = int(np.nanargmax(T_rg))
        valid_idx = np.where(valid)[0]
        # Only mark a *genuine* interior maximum — a curve that is
        # monotone over the sampled window has its argmax on the
        # boundary, and starring that point would be dishonest.
        if valid_idx[0] < peak_idx < valid_idx[-1]:
            ax.plot(M_values[peak_idx], T_rg[peak_idx],
                    marker="*", markersize=14,
                    color=COLOR_NGFP, markeredgecolor="black",
                    label="peak temperature", zorder=5)

    # Mark the critical mass: smallest M with no horizon
    crit_idx = np.where(np.isnan(T_rg))[0]
    if crit_idx.size > 0:
        M_crit = float(M_values[crit_idx[-1]])
        ax.axvline(M_crit, color=COLOR_SEPARATRIX, lw=1.2, ls=":")
        ax.text(M_crit, ax.get_ylim()[1] * 0.85,
                r"$M_{\mathrm{crit}}$", color=COLOR_SEPARATRIX,
                rotation=90, ha="right", va="top", fontsize=10)

    ax.set_xlabel(r"ADM mass $M$ (geometric units)")
    ax.set_ylabel(r"Hawking temperature $T_H$")
    ax.set_title("Hawking temperature: classical vs. asymptotic safety")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)
    # "center right" keeps the citation box off the large-M tail of the
    # temperature curves (which hugs the lower-right corner).
    add_reference_box(
        ax,
        [
            format_arxiv("Bonanno & Reuter (2000)", "hep-th/0002196"),
            format_arxiv("Platania (2023)", "2302.04272"),
        ],
        loc="center right",
    )
    return fig


# ---------------------------------------------------------------------------
# Scale identification
# ---------------------------------------------------------------------------


def plot_scale_identification(
    scales: ScaleIdentification | Sequence[ScaleIdentification],
    *,
    r_range: tuple[float, float] = (1e-3, 50.0),
    n_points: int = 400,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Log-log overlay of ``k(r)`` for one or more scale-identification choices.

    Useful for visualising how :class:`InverseDistanceScale` (diverges
    at the origin) is regularised by :class:`GeodesicDistanceScale`
    (Bonanno-Reuter softening) and :class:`ProperDistanceScale`
    (geodesic distance from a horizon).

    Args:
        scales: One scale or a sequence of scales to overlay.
        r_range: ``(r_min, r_max)`` radial window.
        n_points: Number of log-spaced sample points.
        labels: Optional explicit labels (defaults to class name).
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.get_figure()

    if not isinstance(scales, (list, tuple)):
        scales = [scales]

    r = np.geomspace(r_range[0], r_range[1], n_points)
    for i, scale in enumerate(scales):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        try:
            k_r = np.asarray(scale.k_of_r(r), dtype=float)
        except NotImplementedError:
            continue
        label = (
            labels[i] if labels is not None and i < len(labels)
            else type(scale).__name__
        )
        ax.loglog(r, k_r, color=color, lw=2.0, label=label)

    ax.set_xlabel(r"radius $r$")
    ax.set_ylabel(r"RG scale $k(r)$")
    ax.set_title("Scale-identification prescriptions")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Reuter & Saueressig (2012)", "1202.2274")],
        loc="lower left",
    )
    return fig


# ---------------------------------------------------------------------------
# FLRW
# ---------------------------------------------------------------------------


def plot_flrw_evolution(
    flrw: RGImprovedFLRW,
    *,
    t_span: tuple[float, float] = (1e-2, 10.0),
    n_steps: int = 200,
    a0: float = 1e-4,
    rho0: float = 1.0,
) -> Figure:
    """Three-panel summary of an RG-improved FLRW evolution.

    Stacks ``a(t)``, ``H(t)``, and ``G(t)/G_0`` plus dimensionless
    ``Lambda(t)`` on a shared time axis. Captures the Bonanno-Reuter
    Planck-scale signature: ``G(t) -> 0`` and bounded ``H(t)`` near
    ``t = 0`` as the fixed-point regime regulates the early-time
    singularity.

    The plot is laid out with three rows and a single column to keep
    panel comparison straightforward; passing an ``ax`` is therefore
    not supported (the function always creates its own figure).

    Args:
        flrw: An :class:`RGImprovedFLRW` instance.
        t_span: Cosmic-time integration interval.
        n_steps: Number of stored time points.
        a0: Initial scale factor.
        rho0: Initial energy density.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()

    sol = flrw.integrate(a0=a0, t_span=t_span, rho0=rho0, n_steps=n_steps)

    fig, axs = plt.subplots(
        3, 1, figsize=(7.5, 8.5), sharex=True,
    )
    ax_a, ax_H, ax_couplings = axs

    ax_a.plot(sol["t"], sol["a"], color=COLOR_RELEVANT, lw=2.0)
    ax_a.set_ylabel(r"scale factor $a(t)$")
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", alpha=0.25)

    ax_H.plot(sol["t"], sol["H"], color=COLOR_NGFP, lw=2.0)
    ax_H.set_ylabel(r"Hubble rate $H(t)$")
    ax_H.set_yscale("log")
    ax_H.grid(True, which="both", alpha=0.25)

    G_norm = sol["G"] / max(float(np.max(sol["G"])), 1e-30)
    ax_couplings.plot(sol["t"], G_norm, color=COLOR_RELEVANT, lw=2.0,
                      label=r"$G(t)/G_{\max}$")
    if np.any(sol["Lambda"] != 0.0):
        Lam_norm = sol["Lambda"] / max(
            float(np.max(np.abs(sol["Lambda"]))), 1e-30
        )
        ax_couplings.plot(sol["t"], Lam_norm,
                          color=COLOR_IRRELEVANT, lw=2.0,
                          label=r"$\Lambda(t)/|\Lambda|_{\max}$")
    ax_couplings.set_ylabel("running couplings")
    ax_couplings.set_xlabel(r"cosmic time $t$")
    ax_couplings.set_xscale("log")
    ax_couplings.grid(True, which="both", alpha=0.25)
    ax_couplings.legend(loc="upper right", framealpha=0.85)

    fig.suptitle(
        r"RG-improved FLRW evolution ($w = "
        f"{flrw.equation_of_state:.3g}$, $k_{{\\mathrm{{curv}}}} = "
        f"{flrw.curvature:.0f}$)",
        fontsize=14, y=0.995,
    )
    add_reference_box(
        ax_a,
        [format_arxiv("Bonanno & Reuter (2002)", "hep-th/0106133")],
        loc="upper left",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Compatibility re-exports for the legend builder
# ---------------------------------------------------------------------------

__all__ = [
    "plot_running_newton_constant",
    "plot_lapse_with_horizons",
    "plot_classical_vs_rg_lapse",
    "plot_hawking_temperature",
    "plot_scale_identification",
    "plot_flrw_evolution",
]


# add_legend_panel imported for downstream callers that want to extend
# these plots with custom proxy-artist legends; reference here keeps the
# import non-elidable for static analysers.
_ = add_legend_panel
