"""VQRG-domain plotting (cost landscape, optimisation trajectory, QFI)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

from asymsafety.visualization.style import (
    CMAP_PRESSURE,
    COLOR_NGFP,
    COLOR_RELEVANT,
    add_colorbar,
    add_reference_box,
    apply_style,
    coupling_label,
    format_arxiv,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from asymsafety.quantum.vqrg.cost import VQRGCostFunction


def _evaluate_cost_grid_2d(
    cost: VQRGCostFunction,
    base_params: np.ndarray,
    dims: tuple[int, int],
    p_range: tuple[float, float],
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the VQRG cost on a regular 2D parameter slice.

    Returns ``(P0, P1, C)`` arrays of shape ``(n_grid, n_grid)``.
    """
    p0_vals = np.linspace(p_range[0], p_range[1], n_grid)
    p1_vals = np.linspace(p_range[0], p_range[1], n_grid)
    P0, P1 = np.meshgrid(p0_vals, p1_vals, indexing="xy")
    C = np.empty_like(P0)
    params = base_params.astype(float).copy()
    i0, i1 = dims
    for ii in range(n_grid):
        for jj in range(n_grid):
            params[i0] = P0[ii, jj]
            params[i1] = P1[ii, jj]
            C[ii, jj] = float(cost.evaluate(params))
    return P0, P1, C


def plot_vqrg_cost_landscape(
    cost: VQRGCostFunction,
    base_params: np.ndarray,
    *,
    dims: tuple[int, int] = (0, 1),
    p_range: tuple[float, float] = (-np.pi, np.pi),
    n_grid: int = 32,
    ax: Axes | None = None,
) -> Figure:
    """Filled-contour plot of the VQRG cost over a 2D parameter slice.

    Holds all but two circuit parameters fixed at ``base_params`` and
    sweeps the chosen indices over ``[p_range[0], p_range[1]]``. The
    log-cost ``log10(C + epsilon)`` is plotted, so global minima at
    ``C = 0`` show up as deep wells.

    Args:
        cost: A :class:`VQRGCostFunction` instance.
        base_params: Reference parameter vector; copies are made before
            mutation, so this argument is not modified.
        dims: Pair of parameter indices to sweep.
        p_range: ``(p_lo, p_hi)`` sweep range for both axes.
        n_grid: Number of grid points per axis.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.get_figure()

    P0, P1, C = _evaluate_cost_grid_2d(cost, base_params, dims, p_range, n_grid)

    log_c = np.log10(C + 1e-12)
    contour = ax.contourf(P0, P1, log_c, levels=20, cmap=CMAP_PRESSURE)
    add_colorbar(fig, ax, contour, label=r"$\log_{10} C(\theta)$")

    # Mark the minimum in the slice
    flat_idx = int(np.argmin(C))
    ii, jj = np.unravel_index(flat_idx, C.shape)
    ax.plot(P0[ii, jj], P1[ii, jj], marker="*", color=COLOR_NGFP,
            markersize=14, markeredgecolor="black",
            zorder=5, label="slice minimum")

    ax.set_xlabel(rf"$\theta_{{{dims[0]}}}$")
    ax.set_ylabel(rf"$\theta_{{{dims[1]}}}$")
    ax.set_title(rf"VQRG cost slice on $(\theta_{{{dims[0]}}}, \theta_{{{dims[1]}}})$")
    ax.legend(loc="upper right", framealpha=0.85)
    add_reference_box(
        ax,
        [
            "Cerezo et al. (2021) [2012.09265]",
            "Stokes et al. (2020) [1909.02108]",
        ],
        loc="lower right",
    )
    return fig


def plot_vqrg_optimization_trajectory(
    history: Sequence[float] | np.ndarray,
    *,
    parameters_history: Sequence[np.ndarray] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Semi-log cost-vs-iteration curve, with optional parameter inset.

    When ``parameters_history`` is supplied (e.g. from
    :class:`OptimizationResult.parameters_history`), the first two
    parameters are plotted as a (theta_0, theta_1) trajectory in an
    inset axis.

    Args:
        history: Cost values per iteration.
        parameters_history: Optional list of parameter snapshots per
            iteration. Each entry must have at least two elements.
        ax: Optional existing axes.

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        fig = ax.get_figure()

    history = np.asarray(history, dtype=float)
    iterations = np.arange(history.size)
    ax.semilogy(iterations, history + 1e-15,
                color=COLOR_RELEVANT, lw=2.0, marker="o",
                markersize=3, label="cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$C(\theta)$")
    ax.set_title("VQRG optimisation trajectory")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.85)

    if parameters_history is not None and len(parameters_history) > 0:
        p_arr = np.asarray(parameters_history, dtype=float)
        if p_arr.ndim == 2 and p_arr.shape[1] >= 2:
            inset = ax.inset_axes((0.55, 0.55, 0.42, 0.4))
            inset.plot(p_arr[:, 0], p_arr[:, 1],
                       color=COLOR_NGFP, lw=1.2)
            inset.scatter(p_arr[0, 0], p_arr[0, 1],
                          color="black", s=18, zorder=5)
            inset.scatter(p_arr[-1, 0], p_arr[-1, 1],
                          marker="*", color=COLOR_NGFP, s=80,
                          edgecolor="black", linewidth=0.6, zorder=5)
            inset.set_xlabel(r"$\theta_0$", fontsize=8)
            inset.set_ylabel(r"$\theta_1$", fontsize=8)
            inset.set_title("parameter trajectory", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.25)

    add_reference_box(
        ax,
        ["Stokes et al. (2020) [1909.02108]"],
        loc="lower left",
    )
    return fig


def plot_quantum_fisher_eigenvalues(
    qfi_matrix: np.ndarray,
    *,
    coupling_names: Sequence[str] | None = None,
    show_matrix: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Sorted eigenvalues of the Quantum Fisher Information / Zamolodchikov metric.

    The QFI eigenvalues span the range of distinguishable directions in
    parameter space; small eigenvalues indicate flat directions
    (parameter degeneracies), large eigenvalues indicate informative
    directions.

    Args:
        qfi_matrix: A square :class:`np.ndarray` from
            :meth:`QuantumFisherInformation.compute` or
            :meth:`QuantumFisherInformation.to_coupling_space`.
        coupling_names: Optional names for the secondary heatmap labels.
        show_matrix: Add a sub-axis showing the full QFI heatmap.
        ax: Optional existing axes (only used for the spectrum panel).

    Returns:
        The matplotlib :class:`Figure`.
    """
    apply_style()
    if show_matrix and ax is None:
        fig, (ax_spec, ax_heat) = plt.subplots(
            1, 2, figsize=(11.0, 4.8),
            gridspec_kw={"width_ratios": [1.0, 1.1]},
        )
    elif ax is None:
        fig, ax_spec = plt.subplots(figsize=(7.0, 4.8))
        ax_heat = None
    else:
        fig = ax.get_figure()
        ax_spec = ax
        ax_heat = None

    eigs = np.sort(np.linalg.eigvalsh(qfi_matrix))[::-1]
    ranks = np.arange(1, eigs.size + 1)
    ax_spec.semilogy(ranks, np.maximum(eigs, 1e-30),
                     color=COLOR_RELEVANT, lw=2.0, marker="o", markersize=5)
    ax_spec.set_xlabel("eigenvalue rank")
    ax_spec.set_ylabel(r"$\lambda_i$ of QFI")
    ax_spec.set_title("Quantum Fisher / Zamolodchikov spectrum")
    ax_spec.grid(True, which="both", alpha=0.25)

    if ax_heat is not None:
        mesh = ax_heat.imshow(
            qfi_matrix, cmap=CMAP_PRESSURE, aspect="auto",
        )
        add_colorbar(fig, ax_heat, mesh, label=r"$G_{ij}$")
        ax_heat.set_title("QFI matrix")
        if coupling_names is not None and len(coupling_names) == qfi_matrix.shape[0]:
            ticks = np.arange(qfi_matrix.shape[0])
            labels = [coupling_label(n) for n in coupling_names]
            ax_heat.set_xticks(ticks)
            ax_heat.set_yticks(ticks)
            ax_heat.set_xticklabels(labels, fontsize=9)
            ax_heat.set_yticklabels(labels, fontsize=9)

    add_reference_box(
        ax_spec,
        [
            "Meyer (2021) [2103.15191]",
            format_arxiv("Stokes et al. (2020)", "1909.02108"),
        ],
        loc="upper right",
    )
    return fig


__all__ = [
    "plot_vqrg_cost_landscape",
    "plot_vqrg_optimization_trajectory",
    "plot_quantum_fisher_eigenvalues",
]
