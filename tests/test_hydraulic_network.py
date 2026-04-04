"""Tests for hydraulic network data structures."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.hydraulic.network import (
    HydraulicNetwork,
    HydraulicNode,
    HydraulicPipe,
)
from asymsafety.hydraulic.valves import ExponentialValve, LitimValve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_network() -> HydraulicNetwork:
    """Two-node, one-pipe network for basic tests."""
    nodes = {
        "g": HydraulicNode(name="g", pressure=1.0),
        "lambda": HydraulicNode(name="lambda", pressure=0.5),
    }
    pipes = [HydraulicPipe(node_from="g", node_to="lambda")]
    return HydraulicNetwork(nodes=nodes, pipes=pipes)


# ---------------------------------------------------------------------------
# Node / pipe / network creation
# ---------------------------------------------------------------------------

def test_hydraulic_node_creation():
    node = HydraulicNode(name="g", pressure=1.0, elevation=0.5, is_reservoir=True)
    assert node.name == "g"
    assert node.pressure == 1.0
    assert node.elevation == 0.5
    assert node.is_reservoir is True


def test_hydraulic_pipe_creation():
    pipe = HydraulicPipe(
        node_from="g",
        node_to="lambda",
        length=2.0,
        diameter=0.2,
        friction_factor=0.03,
    )
    assert pipe.node_from == "g"
    assert pipe.node_to == "lambda"
    assert pipe.length == 2.0
    assert pipe.diameter == 0.2
    assert pipe.friction_factor == 0.03


def test_hydraulic_network_creation():
    net = _simple_network()
    assert net.n_nodes == 2
    assert set(net.node_names) == {"g", "lambda"}
    assert len(net.pipes) == 1


# ---------------------------------------------------------------------------
# Incidence and adjacency matrices
# ---------------------------------------------------------------------------

def test_incidence_matrix():
    net = _simple_network()
    A = net.incidence_matrix()
    # Shape: (n_pipes=1, n_nodes=2)
    assert A.shape == (1, 2)
    # Pipe goes from "g" to "lambda"
    idx_g = net.node_names.index("g")
    idx_lam = net.node_names.index("lambda")
    assert A[0, idx_g] == 1.0
    assert A[0, idx_lam] == -1.0


def test_adjacency_matrix():
    net = _simple_network()
    adj = net.adjacency_matrix()
    # Shape: (n_nodes, n_nodes)
    assert adj.shape == (2, 2)
    # Symmetric
    assert_allclose(adj, adj.T)
    # Off-diagonal entries should be 1
    idx_g = net.node_names.index("g")
    idx_lam = net.node_names.index("lambda")
    assert adj[idx_g, idx_lam] == 1.0
    assert adj[idx_lam, idx_g] == 1.0
    # Diagonal should be 0 (no self-loops)
    assert adj[idx_g, idx_g] == 0.0


# ---------------------------------------------------------------------------
# Valve tests
# ---------------------------------------------------------------------------

def test_litim_valve_flow_rate():
    valve = LitimValve(cv=2.0, threshold=0.5)
    # Above threshold -> flow
    q = valve.flow_rate(delta_p=4.0, opening=0.8)
    assert q > 0.0
    assert_allclose(q, 2.0 * np.sqrt(4.0), rtol=1e-10)
    # Below threshold -> no flow
    q_closed = valve.flow_rate(delta_p=4.0, opening=0.3)
    assert q_closed == 0.0


def test_litim_valve_impedance():
    valve = LitimValve(cv=1.0, threshold=0.5)
    imp = valve.impedance(delta_p=1.0, opening=1.0)
    # dQ/d(dp) = cv / (2*sqrt(|dp|)) = 1 / 2 = 0.5
    assert_allclose(imp, 0.5, rtol=1e-10)
    # Closed valve
    imp_closed = valve.impedance(delta_p=1.0, opening=0.3)
    assert imp_closed == 0.0


def test_exponential_valve_flow_rate():
    valve = ExponentialValve(cv=1.0, p_scale=1.0)
    # L'Hopital limit at dp=0: Q -> cv * p_scale * opening
    q_zero = valve.flow_rate(delta_p=0.0, opening=1.0)
    assert_allclose(q_zero, 1.0, rtol=1e-10)
    # Positive dp gives positive flow (check smoothness)
    q_pos = valve.flow_rate(delta_p=1.0, opening=1.0)
    assert np.isfinite(q_pos)


def test_exponential_valve_impedance():
    valve = ExponentialValve(cv=1.0, p_scale=1.0)
    imp = valve.impedance(delta_p=0.0, opening=1.0)
    # Near zero: dQ/d(dp) -> cv * opening * 0.5
    assert_allclose(imp, 0.5, rtol=1e-10)
    # Non-zero dp
    imp2 = valve.impedance(delta_p=1.0, opening=1.0)
    assert np.isfinite(imp2)


def test_valve_names():
    litim = LitimValve()
    exp = ExponentialValve()
    assert litim.name == "Litim"
    assert exp.name == "Exponential"
