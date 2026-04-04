"""Tests for RG -> hydraulic mapping."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.hydraulic.mapping import RGToHydraulicMapper
from asymsafety.hydraulic.valves import ExponentialValve, LitimValve


@pytest.fixture
def eh_system():
    return build_eh_beta_system(d=4)


@pytest.fixture
def mapper(eh_system):
    return RGToHydraulicMapper(eh_system)


def test_mapper_creation(mapper):
    assert mapper.beta_system is not None
    assert len(mapper._names) == 2  # g and lambda


def test_build_network_creates_nodes(mapper):
    network = mapper.build_network()
    # Should have at least one node per coupling (g, lambda)
    coupling_nodes = [n for n in network.node_names if not n.startswith("_reservoir")]
    assert len(coupling_nodes) == 2
    assert "g" in coupling_nodes
    assert "lambda" in coupling_nodes


def test_build_network_creates_pipes(mapper):
    network = mapper.build_network()
    # Should have pipes: at least 2 reservoir pipes (one per coupling diagonal)
    # plus off-diagonal pipes
    assert len(network.pipes) >= 2


def test_pressures_couplings_roundtrip(mapper):
    couplings = {"g": 0.5, "lambda": 0.1}
    pressures = mapper.couplings_to_pressures(couplings)
    recovered = mapper.pressures_to_couplings(pressures)
    for name in couplings:
        assert_allclose(recovered[name], couplings[name], rtol=1e-10)


def test_regulator_types(mapper):
    net_litim = mapper.build_network(regulator_type="litim")
    assert any(isinstance(v, LitimValve) for v in net_litim.valves)

    net_exp = mapper.build_network(regulator_type="exponential")
    assert any(isinstance(v, ExponentialValve) for v in net_exp.valves)
