"""Tests for cross-analogue bridge."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge
from asymsafety.transforms.bridge.impedance import ImpedanceBridge


@pytest.fixture
def eh_system():
    return build_eh_beta_system(d=4)


@pytest.fixture
def ngfp(eh_system):
    finder = FixedPointFinder(eh_system)
    fp = finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
    assert fp is not None, "NGFP search failed"
    return fp


@pytest.fixture
def stability(eh_system, ngfp):
    return analyze_stability(eh_system, ngfp)


@pytest.fixture
def bridge(eh_system, ngfp, stability):
    return CrossAnalogueBridge(eh_system, ngfp, stability)


def test_cross_analogue_creation(bridge):
    assert bridge.system is not None
    assert bridge.fp is not None
    assert bridge.stability is not None


def test_rg_critical_exponents(bridge):
    theta = bridge.rg_critical_exponents()
    assert len(theta) == 2
    assert all(np.isfinite(theta))


def test_matrix_exp_exponents(bridge):
    theta = bridge.matrix_exp_exponents(dt=0.1)
    assert len(theta) == 2
    assert all(np.isfinite(theta.real))


def test_resolvent_poles(bridge):
    theta = bridge.resolvent_poles()
    assert len(theta) == 2
    assert all(np.isfinite(theta.real))


def test_verify_commutativity(bridge):
    result = bridge.verify_commutativity(tol=0.1)
    assert "critical_exponents" in result
    assert "agreements" in result
    assert "all_agree" in result
    # Transfer matrix and resolvent should agree with direct RG
    assert result["agreements"]["transfer_matrix"]["agrees"]
    assert result["agreements"]["resolvent"]["agrees"]


def test_impedance_bridge(stability):
    M = stability.stability_matrix
    ib = ImpedanceBridge(M)
    bode = ib.bode_data(omega_range=(0.1, 10.0), n_points=20)
    assert "omega" in bode
    assert "magnitude" in bode
    assert "phase" in bode
    assert bode["omega"].shape == (20,)
    # magnitude and phase shape: (n, n, n_points)
    n = M.shape[0]
    assert bode["magnitude"].shape == (n, n, 20)
    assert bode["phase"].shape == (n, n, 20)
