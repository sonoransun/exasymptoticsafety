"""Thermal-domain plotting (partition function, Seeley-DeWitt, Gibbs purity)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from asymsafety.visualization.style import (
    COLOR_IRRELEVANT,
    COLOR_NGFP,
    COLOR_RELEVANT,
    COLOR_TRAJECTORY,
    add_reference_box,
    apply_style,
    format_arxiv,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from asymsafety.quantum.thermal.gibbs import GibbsStatePreparer
    from asymsafety.quantum.thermal.partition import (
        PartitionFunctionEstimator,
    )


def plot_partition_function(
    estimator: PartitionFunctionEstimator,
    *,
    beta_range: tuple[float, float] = (0.01, 2.0),
    n_points: int = 80,
) -> Figure:
    """Two-panel ``Z(beta)`` and ``F(beta) = -log Z / beta`` plot.

    Overlays the high-temperature heat-kernel asymptote
    ``(4 pi beta)^(-d/2) * b_0`` (extracted via
    :meth:`estimate_coefficients`) on the partition-function panel.

    Args:
        estimator: A :class:`PartitionFunctionEstimator`.
        beta_range: ``(beta_min, beta_max)`` log-space sweep.
        n_points: Number of beta values.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    fig, (ax_z, ax_f) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    betas = np.geomspace(beta_range[0], beta_range[1], n_points)
    z_vals = np.array([estimator.gibbs.partition_function(b) for b in betas])

    ax_z.loglog(betas, z_vals, color=COLOR_RELEVANT, lw=2.0, label=r"$Z(\beta)$")

    try:
        coeffs = estimator.estimate_coefficients()
        b0 = coeffs.get("b0", float("nan"))
        d = estimator.d
        prefactor = (4.0 * np.pi * betas) ** (-d / 2.0)
        ax_z.loglog(
            betas, prefactor * b0, color=COLOR_TRAJECTORY,
            lw=1.4, ls="--",
            label=rf"free-field limit $(4\pi\beta)^{{-d/2}}\,b_0$",
        )
    except Exception:
        pass

    ax_z.set_xlabel(r"inverse temperature $\beta$")
    ax_z.set_ylabel(r"partition function $Z(\beta)$")
    ax_z.set_title("Partition function")
    ax_z.grid(True, which="both", alpha=0.25)
    ax_z.legend(loc="upper right", framealpha=0.85)

    safe_z = np.maximum(z_vals, 1e-30)
    f_vals = -np.log(safe_z) / betas
    ax_f.semilogx(betas, f_vals, color=COLOR_NGFP, lw=2.0)
    ax_f.set_xlabel(r"$\beta$")
    ax_f.set_ylabel(r"free energy $F(\beta) = -\log Z / \beta$")
    ax_f.set_title("Free energy")
    ax_f.grid(True, which="both", alpha=0.25)

    add_reference_box(
        ax_z,
        [format_arxiv("Vassilevich (2003)", "hep-th/0306138")],
        loc="lower left",
    )
    fig.tight_layout()
    return fig


def plot_seeley_dewitt_extraction(
    estimator: PartitionFunctionEstimator,
    *,
    sdw: Any | None = None,
    field_type: str = "scalar",
) -> Figure:
    """Bar chart comparing quantum-extracted Seeley-DeWitt coefficients with classical values.

    When ``sdw`` (a ``SeeleyDeWittCoefficients`` instance) is provided,
    side-by-side bars show the classical reference; deviations are
    annotated above each pair.

    Args:
        estimator: A :class:`PartitionFunctionEstimator`.
        sdw: Optional classical reference object exposing
            ``b0(field_type)``, ``b2(field_type)``,
            ``b4_on_sphere(field_type)``.
        field_type: Field type passed to ``sdw`` when comparing.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    quantum = estimator.estimate_coefficients()
    labels = list(quantum.keys())
    q_vals = np.array([quantum[k] for k in labels], dtype=float)

    x = np.arange(len(labels))
    width = 0.35

    if sdw is None:
        ax.bar(x, q_vals, width=width * 1.6,
               color=COLOR_RELEVANT, edgecolor="black", linewidth=0.6,
               label="quantum")
    else:
        c_vals = []
        for k in labels:
            try:
                if k == "b0":
                    c_vals.append(float(sdw.b0(field_type)))
                elif k == "b2":
                    c_vals.append(float(sdw.b2(field_type)))
                elif k == "b4":
                    c_vals.append(float(sdw.b4_on_sphere(field_type)))
                else:
                    c_vals.append(float("nan"))
            except Exception:
                c_vals.append(float("nan"))
        c_arr = np.array(c_vals, dtype=float)

        ax.bar(x - width / 2, q_vals, width=width,
               color=COLOR_RELEVANT, edgecolor="black", linewidth=0.5,
               label="quantum")
        ax.bar(x + width / 2, c_arr, width=width,
               color=COLOR_IRRELEVANT, edgecolor="black", linewidth=0.5,
               label="classical")

        for i, (q, c) in enumerate(zip(q_vals, c_arr)):
            if np.isfinite(c):
                ax.text(
                    i, max(q, c) * 1.05 if max(q, c) > 0 else max(q, c) * 0.95,
                    rf"$\Delta={abs(q - c):.2e}$",
                    ha="center", va="bottom", fontsize=8, color="0.3",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("coefficient value")
    ax.set_title(rf"Seeley-DeWitt extraction ({field_type})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Vassilevich (2003)", "hep-th/0306138")],
        loc="lower right",
    )
    return fig


def plot_gibbs_state_purity(
    gibbs: GibbsStatePreparer,
    *,
    beta_range: tuple[float, float] = (0.01, 5.0),
    n_points: int = 80,
) -> Figure:
    """Curve of Gibbs-state purity ``Tr(rho^2)`` vs inverse temperature.

    At ``beta -> 0`` the state is maximally mixed (purity = ``1/d_eff``);
    at ``beta -> infinity`` the state collapses onto the ground state
    (purity = 1). Crossover scale is ``beta ~ 1/Delta E``.

    Args:
        gibbs: A :class:`GibbsStatePreparer`.
        beta_range: ``(beta_min, beta_max)`` log-space sweep.
        n_points: Number of beta samples.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    betas = np.geomspace(beta_range[0], beta_range[1], n_points)
    purities = np.array([
        float(np.sum(gibbs.prepare_exact(b) ** 2)) for b in betas
    ])

    ax.semilogx(betas, purities, color=COLOR_RELEVANT, lw=2.0,
                label=r"$\mathrm{Tr}\,\rho^2$")

    H_diag = np.diag(gibbs._hamiltonian.to_matrix())  # noqa: SLF001
    d_eff = max(1, len(H_diag))
    ax.axhline(1.0 / d_eff, color=COLOR_TRAJECTORY, lw=1.0, ls="--",
               label=rf"$1/d_{{\mathrm{{eff}}}} = {1.0 / d_eff:.3g}$")
    ax.axhline(1.0, color=COLOR_NGFP, lw=1.0, ls=":",
               label="ground-state purity")

    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel("purity")
    ax.set_title("Gibbs-state purity")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.85)
    add_reference_box(
        ax,
        [format_arxiv("Vassilevich (2003)", "hep-th/0306138")],
        loc="upper left",
    )
    return fig


__all__ = [
    "plot_partition_function",
    "plot_seeley_dewitt_extraction",
    "plot_gibbs_state_purity",
]
