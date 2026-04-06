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
    """Plot the hydraulic network graph.

    Nodes are drawn as circles coloured by pressure (using the
    ``coolwarm`` colourmap).  Pipes are drawn as arrows whose width
    is proportional to the pipe diameter.  Reservoir nodes are shown
    as squares.

    Layout is computed using a simple spring-embedding algorithm
    implemented with pure numpy (no networkx dependency).

    Args:
        network: The network to visualise.
        pressures: Optional pressure values for colouring nodes.
            If ``None``, the pressures stored in the nodes are used.
        ax: Matplotlib axes to draw on.  A new figure is created if
            ``None``.

    Returns:
        The matplotlib :class:`Figure` containing the plot.
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
    """Plot pressure vs time at each node.

    One line per node with a legend showing node names.

    Args:
        trajectory: The :class:`HydraulicTrajectory` to plot.
        ax: Matplotlib axes to draw on.  A new figure is created if
            ``None``.

    Returns:
        The matplotlib :class:`Figure` containing the plot.
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
