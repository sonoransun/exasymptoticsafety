"""Fixed point and critical exponent visualization.

Public API
----------
- :func:`plot_critical_exponents` — two-panel ``Re(theta_i)`` /
  ``Im(theta_i)`` plot along a continuation parameter, with numeric
  values shown inline in the legend.
- :func:`plot_fixed_point_locations` — fixed-point coordinates
  ``g_i^*`` along a continuation parameter (each coupling on its own
  axes; shared x).
- :func:`plot_matter_content_continuation` — specialised plot for
  matter-content sweeps (``N_s``, ``N_v``, ``N_D``) overlaying the
  bounds from :mod:`asymsafety.validation.korver_2024`.

References
----------
- Codello et al. (2009), Ann. Phys. 324, 414 [0812.0785]
- Dona et al. (2014), Phys. Rev. D 89, 084035 [1311.2898]
- Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 [2402.01260]
- See ``docs/LITERATURE.md``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from asymsafety.analysis.continuation import ContinuationResult
from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.visualization.style import (
    COLOR_RELEVANT,
    COLOR_IRRELEVANT,
    add_reference_box,
    coupling_label,
    format_arxiv,
)


def plot_critical_exponents(
    continuation: ContinuationResult,
    figsize: tuple[float, float] = (10, 6),
    *,
    show_numeric_legend: bool = True,
    show_references: bool = False,
) -> Figure:
    r"""Plot critical exponents along a continuation parameter.

    Physics
    -------
    The critical exponents ``theta_i = -eig(M)`` (with
    ``M_ij = d beta_i / d g_j`` evaluated at the fixed point) determine
    UV relevance: ``Re(theta_i) > 0`` indicates a relevant direction,
    ``Re(theta_i) < 0`` an irrelevant one. Tracking ``theta_i`` against
    a continuation parameter (gauge, dimension, matter content)
    diagnoses when the asymptotic-safety scenario gains or loses
    relevant directions, and exposes the imaginary parts of complex-
    conjugate pairs that signal spiral flow at the NGFP.

    Parameters
    ----------
    continuation : :class:`~asymsafety.analysis.continuation.ContinuationResult`
        Result of a parameter sweep.
    figsize : tuple, default ``(10, 6)``
    show_numeric_legend : bool, default ``True``
        Annotate the legend entries with the mean value of
        ``Re(theta_i)`` across the continuation.
    show_references : bool, default ``False``
        Embed an inline citation box (Codello 2009, Dona 2014).

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure: top ``Re(theta_i)``, bottom ``Im(theta_i)``.

    References
    ----------
    - Codello et al. (2009), Ann. Phys. 324, 414 [0812.0785]
    - Dona et al. (2014), Phys. Rev. D 89, 084035 [1311.2898]
    - See ``docs/LITERATURE.md``.

    See Also
    --------
    :func:`plot_fixed_point_locations`
        Coordinates ``g_i^*`` along the same continuation.
    :func:`plot_matter_content_continuation`
        Specialised plot for matter-content sweeps with reference
        bounds from Korver et al. (2024).
    :func:`asymsafety.gui.visualization_3d.fixed_point_stability_3d`
        3D zoom on a single fixed point.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    thetas = continuation.critical_exponents_array
    param = continuation.parameter_values

    n_exponents = thetas.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_exponents))

    for i in range(n_exponents):
        re = thetas[:, i].real
        im = thetas[:, i].imag

        if show_numeric_legend:
            mean_re = float(np.nanmean(re))
            label_top = (
                rf"$\theta_{i+1}$ "
                rf"($\langle\mathrm{{Re}}\rangle={mean_re:+.2f}$)"
            )
        else:
            label_top = rf"$\theta_{i+1}$"

        ax1.plot(param, re, '-o', color=colors[i],
                 markersize=3, label=label_top)
        ax2.plot(param, im, '-o', color=colors[i], markersize=3)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_ylabel(r"Re($\theta_i$)", fontsize=12)
    ax2.set_ylabel(r"Im($\theta_i$)", fontsize=12)
    ax2.set_xlabel(continuation.parameter_name, fontsize=12)
    ax1.set_title("Critical Exponents", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Shaded bands for relevant / irrelevant regions
    y_lo, y_hi = ax1.get_ylim()
    ax1.axhspan(0, y_hi, color=COLOR_RELEVANT, alpha=0.06)
    ax1.axhspan(y_lo, 0, color=COLOR_IRRELEVANT, alpha=0.06)
    ax1.text(
        0.02, 0.95, "relevant", transform=ax1.transAxes,
        fontsize=9, color=COLOR_RELEVANT, va="top", alpha=0.7,
    )
    ax1.text(
        0.02, 0.05, "irrelevant", transform=ax1.transAxes,
        fontsize=9, color=COLOR_IRRELEVANT, va="bottom", alpha=0.7,
    )
    ax1.set_ylim(y_lo, y_hi)

    if show_references:
        add_reference_box(
            ax2,
            [
                format_arxiv("Codello et al. (2009)", "0812.0785"),
                format_arxiv("Dona et al. (2014)", "1311.2898"),
            ],
            loc="lower right",
        )

    fig.tight_layout()
    return fig


def plot_fixed_point_locations(
    continuation: ContinuationResult,
    figsize: tuple[float, float] = (10, 4),
    *,
    reference_values: dict[str, float] | None = None,
) -> Figure:
    r"""Plot fixed-point coordinates as a continuation parameter varies.

    Physics
    -------
    The location ``g_i^*(p)`` of the fixed point depends on a control
    parameter ``p`` (matter content, gauge, dimension). Sudden jumps
    or NaN segments indicate the NGFP merging with another fixed point
    or disappearing — bounding the region of theory space compatible
    with asymptotic safety.

    Parameters
    ----------
    continuation : :class:`~asymsafety.analysis.continuation.ContinuationResult`
    figsize : tuple, default ``(10, 4)``
    reference_values : dict, optional
        ``{coupling_name: reference_value}``. When supplied, draws a
        dotted horizontal line at ``reference_value`` on each panel —
        useful for overlaying e.g. the Reuter (1998) ``g* ≈ 0.707``
        coordinate when the continuation is around the canonical EH
        truncation.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per coupling; shared x axis.

    References
    ----------
    - Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    - Manrique et al. (2011), Phys. Rev. Lett. 106, 251302 [1003.5129]
    - See ``docs/LITERATURE.md``.

    See Also
    --------
    :func:`plot_critical_exponents`
        Critical exponents along the same continuation.
    """
    fig, axes = plt.subplots(1, len(continuation.locations),
                             figsize=figsize, squeeze=False, sharex=True)

    for i, (name, values) in enumerate(continuation.locations.items()):
        ax = axes[0, i]
        ax.plot(continuation.parameter_values, values, 'b-o', markersize=3)
        ax.set_xlabel(continuation.parameter_name, fontsize=12)
        ax.set_ylabel(coupling_label(name).rstrip("$") + "^*$", fontsize=12)
        ax.grid(True, alpha=0.3)
        if reference_values and name in reference_values:
            ref = reference_values[name]
            ax.axhline(
                ref, color="0.45", linestyle=":", lw=1.2,
                label=f"reference: {ref:.3f}",
            )
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Fixed Point Location", fontsize=14)
    fig.tight_layout()
    return fig


def plot_matter_content_continuation(
    continuation: ContinuationResult,
    figsize: tuple[float, float] = (12, 7),
    *,
    bounds: dict[str, int] | None = None,
    bounds_label: str = "Korver et al. (2024)",
    bounds_arxiv: str = "2402.01260",
) -> Figure:
    r"""NGFP location and critical exponents vs matter content.

    Physics
    -------
    Adding matter fields shifts the gravitational beta functions
    through the matter-graviton interaction terms and changes the NGFP
    coordinates. The NGFP ceases to exist beyond certain matter
    content — Korver, Saueressig & Wang (2024) bound the foliated
    truncation at roughly ``N_s <= 12`` minimally coupled scalars and
    ``N_v <= 6`` gauge vectors. This figure plots ``g^*``, ``lambda^*``
    and ``Re(theta_i)`` against the continuation parameter (typically
    ``N_s``) and shades the region beyond the published bound.

    Parameters
    ----------
    continuation : :class:`~asymsafety.analysis.continuation.ContinuationResult`
        Result of a sweep against a matter-counting parameter.
    figsize : tuple, default ``(12, 7)``
    bounds : dict, optional
        Per-parameter integer bound, e.g. ``{"N_s": 12}``. Defaults to
        :data:`asymsafety.validation.korver_2024.FOLIATED_MATTER_BOUNDS`
        for the parameter named in the continuation.
    bounds_label : str
        Citation label drawn near the bound region.
    bounds_arxiv : str
        arXiv ID for the citation.

    Returns
    -------
    matplotlib.figure.Figure
        Combined ``g^*``, ``lambda^*``, ``Re(theta_i)`` panels with
        the matter bound shaded.

    References
    ----------
    - Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035 [1311.2898]
    - Eichhorn & Schiffer (2022) [2212.07456]
    - Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 [2402.01260]

    See Also
    --------
    :mod:`asymsafety.validation.korver_2024`
        Foliated and covariant matter bounds.
    :func:`asymsafety.beta.matter.build_eh_matter_beta_system`
        Builder for the matter-extended beta-function system.
    """
    if bounds is None:
        try:
            from asymsafety.validation.korver_2024 import FOLIATED_MATTER_BOUNDS
            bounds = {
                "N_s": int(FOLIATED_MATTER_BOUNDS.get("max_N_s", 12)),
                "N_v": int(FOLIATED_MATTER_BOUNDS.get("max_N_v", 6)),
            }
        except Exception:
            bounds = {}

    locations = continuation.locations
    thetas = continuation.critical_exponents_array
    param = continuation.parameter_values

    n_couplings = len(locations)
    n_panels = n_couplings + 1  # one for theta

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, squeeze=False, sharex=True,
    )
    axes = axes[0]

    bound_value = bounds.get(continuation.parameter_name)

    for i, (name, values) in enumerate(locations.items()):
        ax = axes[i]
        ax.plot(
            param, values, 'o-', color=COLOR_RELEVANT,
            markersize=4, lw=1.4,
        )
        ax.set_xlabel(continuation.parameter_name, fontsize=11)
        ax.set_ylabel(coupling_label(name).rstrip("$") + "^*$", fontsize=12)
        ax.grid(True, alpha=0.3)
        if bound_value is not None:
            ax.axvspan(
                bound_value, max(np.nanmax(param), bound_value + 0.5),
                color=COLOR_IRRELEVANT, alpha=0.12,
                label=f"{bounds_label} bound",
            )
            ax.axvline(bound_value, color=COLOR_IRRELEVANT, ls="--", lw=1.2)
            ax.legend(fontsize=8, loc="best")

    # Theta panel
    ax_t = axes[-1]
    n_exponents = thetas.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_exponents, 1)))
    for j in range(n_exponents):
        ax_t.plot(
            param, thetas[:, j].real, 'o-', color=colors[j],
            markersize=3, lw=1.2, label=rf"$\theta_{j+1}$",
        )
    ax_t.axhline(0, color="gray", lw=0.8, ls="--")
    ax_t.set_xlabel(continuation.parameter_name, fontsize=11)
    ax_t.set_ylabel(r"Re($\theta_i$)", fontsize=12)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)
    if bound_value is not None:
        ax_t.axvspan(
            bound_value, max(np.nanmax(param), bound_value + 0.5),
            color=COLOR_IRRELEVANT, alpha=0.12,
        )
        ax_t.axvline(bound_value, color=COLOR_IRRELEVANT, ls="--", lw=1.2)

    add_reference_box(
        axes[-1],
        [format_arxiv(bounds_label, bounds_arxiv)],
        loc="lower right",
    )

    fig.suptitle(
        rf"Matter-content continuation: NGFP vs. ${continuation.parameter_name}$",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def nu_vs_Nf(
    figsize: tuple[float, float] = (9, 6),
    *,
    Nf_min: int = 10,
    Nf_max: int = 120,
    show_references: bool = True,
) -> Figure:
    r"""Correlation-length exponent ``ν`` at the charged FP vs. ``N_f``.

    Overlays three curves:

    * the toolkit's one-loop ``4-ε`` prediction (computed via
      :func:`asymsafety.transforms.bridge.gauge_higgs.correlation_length_exponent`,
      using the simplified Abelian case with ``Nc = 1``);
    * the large-``N_f`` field-theory asymptote
      ``ν = 1 - 9.727 / N_f`` (Bonati et al. 2025, SU(2));
    * the three lattice Monte-Carlo points at
      ``N_f ∈ {30, 40, 60}`` with their published error bars.

    The visual purpose is to make explicit that the perturbative
    one-loop scheme used inside the toolkit reaches the *Wilson-Fisher*
    limit ``ν → 1/2`` from below, while the d=3 lattice numbers
    instead approach ``ν → 1`` — the two pictures share the same
    qualitative existence of a charged fixed point but disagree
    quantitatively, exactly the regulator/scheme story discussed in
    the gravity-side asymptotic-safety literature.

    Parameters
    ----------
    figsize : tuple, default ``(9, 6)``
    Nf_min, Nf_max : int
        Range of ``N_f`` swept for the smooth curves.
    show_references : bool, default ``True``

    Returns
    -------
    matplotlib.figure.Figure
    """
    from asymsafety.transforms.bridge.gauge_higgs import (
        correlation_length_exponent,
    )
    from asymsafety.validation.bonati_2025 import (
        BONATI_LARGE_NF,
        BONATI_SU2_MC,
    )

    Nfs = np.arange(Nf_min, Nf_max + 1)

    # Toolkit one-loop prediction (Nc = 1, ε = 1). Below the perturbative
    # threshold N* (where the (α, u) quadratic's discriminant goes
    # negative), the charged FP merges with the Gaussian branch and ν
    # is ill-defined in this scheme — so we restrict the curve to the
    # perturbatively-stable regime and annotate N* on the figure.
    nu_toolkit_all = np.array(
        [correlation_length_exponent(int(n), epsilon=1.0) for n in Nfs]
    )
    # Detect the perturbative threshold: find the smallest Nf for which
    # the closed-form ν is monotonically decreasing toward 1/2 from above
    # (above N* the gauge correction dominates and ν drops).
    diffs = np.diff(nu_toolkit_all)
    n_star_idx = int(np.argmax(diffs < -1e-3)) if any(diffs < -1e-3) else 0
    n_star = int(Nfs[n_star_idx + 1]) if n_star_idx + 1 < len(Nfs) else Nf_min
    mask = Nfs >= n_star
    nu_toolkit = np.where(mask, nu_toolkit_all, np.nan)

    # Large-Nf asymptote from the paper
    C = BONATI_LARGE_NF["nu_coeff"]
    nu_large_nf = 1.0 - C / Nfs

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(
        Nfs, nu_toolkit, color=COLOR_IRRELEVANT, lw=2.2,
        label=rf"toolkit one-loop 4-$\varepsilon$ (Nc=1, $N_f \geq {n_star}$)",
    )
    ax.plot(
        Nfs, nu_large_nf, color="0.25", lw=2.0, ls="--",
        label=r"large-$N_f$:  $\nu = 1 - 9.727 / N_f$",
    )
    ax.axvline(n_star, color=COLOR_IRRELEVANT, ls=":", lw=1.0, alpha=0.7)
    ax.annotate(
        rf"perturbative $N^* \approx {n_star}$",
        xy=(n_star, 0.55), xytext=(n_star + 8, 0.43),
        fontsize=8.5, color=COLOR_IRRELEVANT,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_IRRELEVANT, lw=0.8),
    )

    # Bonati MC points with error bars
    mc_Nf = sorted(BONATI_SU2_MC)
    mc_nu = [BONATI_SU2_MC[n]["nu"] for n in mc_Nf]
    mc_err = [BONATI_SU2_MC[n]["nu_err"] for n in mc_Nf]
    ax.errorbar(
        mc_Nf, mc_nu, yerr=mc_err,
        fmt="o", color=COLOR_RELEVANT, ms=9, mec="black", mew=1.0,
        elinewidth=1.2, capsize=4, zorder=5,
        label="Bonati 2025 lattice MC (SU(2))",
    )

    # Asymptotes (1/2 and 1) for orientation
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.text(Nf_max - 2, 0.51, "Wilson-Fisher (1/2)",
            fontsize=8, color="0.5", ha="right")
    ax.text(Nf_max - 2, 1.01, "large-$N_f$ asymptote (1)",
            fontsize=8, color="0.5", ha="right")

    ax.set_xlabel(r"Flavor count $N_f$", fontsize=12)
    ax.set_ylabel(r"Correlation-length exponent $\nu$", fontsize=12)
    ax.set_title(
        r"$\nu(N_f)$ at the charged fixed point: "
        r"perturbative vs lattice",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(Nf_min, Nf_max)
    ax.set_ylim(0.4, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9.5, framealpha=0.92)

    if show_references:
        add_reference_box(
            ax,
            [format_arxiv("Bonati, Pelissetto & Vicari 2025", "2410.05823")],
            loc="lower right",
        )

    return fig
