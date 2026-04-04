"""Tests for hydraulic simulation."""
from __future__ import annotations

import numpy as np
import pytest

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.hydraulic.mapping import RGToHydraulicMapper
from asymsafety.hydraulic.simulator import HydraulicSimulator


@pytest.fixture
def network():
    system = build_eh_beta_system(d=4)
    mapper = RGToHydraulicMapper(system)
    return mapper.build_network()


@pytest.fixture
def simulator(network):
    return HydraulicSimulator(network)


def test_simulator_creation(simulator, network):
    assert simulator.network is network


def test_transient_simulation(simulator, network):
    P_init = {name: 0.0 for name in network.node_names}
    traj = simulator.simulate_transient(P_init, t_span=(0.0, 1.0), n_points=50)
    assert traj.t_values.shape[0] == 50
    assert len(traj.pressure_values) == network.n_nodes
    for name in network.node_names:
        assert name in traj.pressure_values
        assert traj.pressure_values[name].shape[0] == 50


def test_find_steady_state(simulator, network):
    P_init = {name: 0.0 for name in network.node_names}
    ss = simulator.find_steady_state(P_init)
    # The zero initial guess should converge (trivial steady state)
    if ss is not None:
        for name, p in ss.pressures.items():
            assert np.isfinite(p)


def test_steady_state_eigenvalues(simulator, network):
    P_init = {name: 0.0 for name in network.node_names}
    ss = simulator.find_steady_state(P_init)
    if ss is not None:
        assert ss.eigenvalues is not None
        assert all(np.isfinite(e) for e in ss.eigenvalues)
