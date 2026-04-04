"""Hydraulic network data structures.

Defines the pipe-network graph: nodes (pressure points analogous to
couplings), pipes (flow paths analogous to off-diagonal Jacobian entries),
and the assembled network with incidence and adjacency matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HydraulicNode:
    """A node in the hydraulic network (analogue of a coupling).

    Attributes:
        name: Coupling name (e.g., "g", "lambda").
        pressure: Current pressure at this node (= coupling value).
        elevation: Static head contribution from gravity.
        is_reservoir: If True, this node has a fixed pressure (boundary).
    """

    name: str
    pressure: float = 0.0
    elevation: float = 0.0
    is_reservoir: bool = False


@dataclass
class HydraulicPipe:
    """A pipe connecting two nodes (analogue of off-diagonal Jacobian entry).

    Attributes:
        node_from: Name of the source node.
        node_to: Name of the destination node.
        length: Pipe length in metres.
        diameter: Pipe inner diameter in metres.
        friction_factor: Darcy-Weisbach friction factor.
    """

    node_from: str
    node_to: str
    length: float = 1.0
    diameter: float = 0.1
    friction_factor: float = 0.02


@dataclass
class HydraulicNetwork:
    """Complete hydraulic pipe network.

    The network graph connects :class:`HydraulicNode` instances via
    :class:`HydraulicPipe` elements and optional :class:`Valve` objects.
    The density and viscosity characterise the working fluid.

    Attributes:
        nodes: Mapping from node name to :class:`HydraulicNode`.
        pipes: List of pipes in the network.
        valves: List of :class:`Valve` objects controlling flow.
        density: Fluid density in kg/m^3.
        viscosity: Dynamic viscosity in Pa.s.
    """

    nodes: dict[str, HydraulicNode]
    pipes: list[HydraulicPipe]
    valves: list[Any] = field(default_factory=list)
    density: float = 1000.0
    viscosity: float = 1e-3

    @property
    def node_names(self) -> list[str]:
        """Ordered list of node names."""
        return list(self.nodes.keys())

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)

    def incidence_matrix(self) -> np.ndarray:
        """Build the incidence matrix A (n_pipes x n_nodes).

        ``A[p, i] = +1`` if pipe *p* leaves node *i*,
        ``A[p, i] = -1`` if pipe *p* enters node *i*.

        Returns:
            Incidence matrix of shape ``(n_pipes, n_nodes)``.
        """
        names = self.node_names
        name_to_idx = {name: i for i, name in enumerate(names)}
        n_pipes = len(self.pipes)
        n_nodes = self.n_nodes

        A = np.zeros((n_pipes, n_nodes), dtype=float)
        for p, pipe in enumerate(self.pipes):
            if pipe.node_from in name_to_idx:
                A[p, name_to_idx[pipe.node_from]] = +1.0
            if pipe.node_to in name_to_idx:
                A[p, name_to_idx[pipe.node_to]] = -1.0
        return A

    def adjacency_matrix(self) -> np.ndarray:
        """Build the adjacency matrix of the network graph.

        ``adj[i, j] = 1`` if there is a pipe from node *i* to node *j*
        (or from *j* to *i*).

        Returns:
            Symmetric adjacency matrix of shape ``(n_nodes, n_nodes)``.
        """
        names = self.node_names
        name_to_idx = {name: i for i, name in enumerate(names)}
        n = self.n_nodes

        adj = np.zeros((n, n), dtype=float)
        for pipe in self.pipes:
            i = name_to_idx.get(pipe.node_from)
            j = name_to_idx.get(pipe.node_to)
            if i is not None and j is not None:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj
