"""Visualization of hydraulic networks and transient responses.

Provides matplotlib-based plotting of the network graph (nodes coloured
by pressure, edges scaled by pipe diameter) and time-series of the
transient pressure response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from asymsafety.visualization.style import (
    CMAP_PRESSURE,
    coupling_label,
)

if TYPE_CHECKING:
    from asymsafety.hydraulic.network import HydraulicNetwork
    from asymsafety.hydraulic.simulator import HydraulicTrajectory


def plot_hydraulic_network(
    network: HydraulicNetwork,
    pressures: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    r"""Plot the hydraulic network graph.

    Physics
    -------
    The hydraulic mapping (see
    :class:`asymsafety.hydraulic.mapping.RGToHydraulicMapper`) sends
    each coupling ``g_i`` to a node carrying pressure ``P_i``, each
    beta-function term to a pipe between nodes, and each regulator to
    a valve. Spring-embedded layout positions the nodes; pressure is
    rendered as the node colour with the shared
    :data:`asymsafety.visualization.style.CMAP_PRESSURE` colourmap.
    Reservoir nodes are drawn as grey squares; pipe widths scale with
    the pipe diameter.

    Parameters
    ----------
    network : :class:`~asymsafety.hydraulic.network.HydraulicNetwork`
        Network produced by the hydraulic mapping.
    pressures : dict, optional
        Override the per-node pressure values used for colouring. If
        ``None``, the pressures stored on each node are used.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to draw on; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    - See README section "Physical Computational Analogues".

    See Also
    --------
    :func:`asymsafety.visualization.bridge_diagram.hydraulic_analogy_diagram`
        Schematic side-by-side RG ↔ hydraulic comparison.
    :func:`plot_transient_response`
        Companion time-series plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    names = network.node_names
    n = network.n_nodes
    name_to_idx = {name: i for i, name in enumerate(names)}

    # --- Spring-layout positions ---
    rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, size=(n, 2))

    # Simple force-directed layout (Fruchterman-Reingold style)
    adj = network.adjacency_matrix()
    k_spring = 1.0
    for _ in range(200):
        # Repulsive forces between all pairs
        forces = np.zeros_like(pos)
        for i in range(n):
            diff = pos[i] - pos  # (n, 2)
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            dist = np.where(dist < 1e-4, 1e-4, dist)
            repulsion = diff / dist[:, np.newaxis] ** 2
            repulsion[i] = 0.0
            forces[i] += np.sum(repulsion, axis=0)

        # Attractive forces along edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    diff = pos[j] - pos[i]
                    dist = max(np.linalg.norm(diff), 1e-4)
                    attraction = k_spring * diff * dist
                    forces[i] += attraction
                    forces[j] -= attraction

        # Damped update
        step = 0.01
        pos += step * forces

    # --- Collect pressure values ---
    if pressures is None:
        p_vals = np.array([network.nodes[name].pressure for name in names])
    else:
        p_vals = np.array([pressures.get(name, 0.0) for name in names])

    # --- Draw edges (pipes) ---
    for pipe in network.pipes:
        i = name_to_idx.get(pipe.node_from)
        j = name_to_idx.get(pipe.node_to)
        if i is None or j is None:
            continue
        x_vals = [pos[i, 0], pos[j, 0]]
        y_vals = [pos[i, 1], pos[j, 1]]
        lw = max(0.5, pipe.diameter * 20.0)
        ax.annotate(
            "",
            xy=(pos[j, 0], pos[j, 1]),
            xytext=(pos[i, 0], pos[i, 1]),
            arrowprops=dict(
                arrowstyle="->",
                lw=lw,
                color="gray",
                alpha=0.6,
            ),
        )

    # --- Draw nodes ---
    is_reservoir = [network.nodes[name].is_reservoir for name in names]

    # Regular (coupling) nodes
    regular_idx = [i for i in range(n) if not is_reservoir[i]]
    if regular_idx:
        reg_pos = pos[regular_idx]
        reg_p = p_vals[regular_idx]
        sc = ax.scatter(
            reg_pos[:, 0], reg_pos[:, 1],
            c=reg_p, cmap=CMAP_PRESSURE, s=300, zorder=5,
            edgecolors="black", linewidths=1.5,
        )
        fig.colorbar(sc, ax=ax, label="Pressure (Pa)", shrink=0.8)
        for idx in regular_idx:
            node_name = names[idx]
            display = coupling_label(node_name).strip("$")
            ax.annotate(
                f"${display}$ (pressure)",
                (pos[idx, 0], pos[idx, 1]),
                ha="center", va="center", fontsize=9, fontweight="bold",
                zorder=6,
            )

    # Reservoir nodes (squares, light gray fill)
    res_idx = [i for i in range(n) if is_reservoir[i]]
    if res_idx:
        res_pos = pos[res_idx]
        res_p = p_vals[res_idx]
        ax.scatter(
            res_pos[:, 0], res_pos[:, 1],
            c="lightgray", s=200, zorder=5, marker="s",
            edgecolors="black", linewidths=1.5,
        )

    ax.set_title("Hydraulic Network", fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_transient_response(
    trajectory: HydraulicTrajectory,
    ax: plt.Axes | None = None,
) -> Figure:
    r"""Plot pressure vs time at each node of the hydraulic network.

    Physics
    -------
    Maps RG-time evolution of couplings onto the transient response of
    pressures in the analogue hydraulic system. Steady-state pressure
    levels correspond to the fixed-point coordinates ``g_i^*``; the
    relaxation rate of each pressure curve mirrors the (modulus of
    the) corresponding critical exponent.

    Parameters
    ----------
    trajectory : :class:`~asymsafety.hydraulic.simulator.HydraulicTrajectory`
    ax : :class:`matplotlib.axes.Axes`, optional

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :func:`plot_hydraulic_network`
        Spatial network view of the same system.
    :func:`asymsafety.visualization.phase_portrait.flow_diagram`
        Direct RG-flow analogue: running couplings vs scale.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for name in trajectory.node_names:
        if name in trajectory.pressure_values:
            ax.plot(
                trajectory.t_values,
                trajectory.pressure_values[name],
                label=coupling_label(name),
            )

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Pressure (Pa)", fontsize=14)
    ax.set_title("Transient Pressure Response", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
