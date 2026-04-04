"""Tests for discrete RG transfer matrix."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.linear.transfer_matrix import DiscreteRGTransferMatrix


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
def tm(eh_system, ngfp):
    return DiscreteRGTransferMatrix(eh_system, ngfp)


def test_compute_exact(tm):
    dt = 0.1
    T = tm.compute(dt, method="exact")
    # T = expm(M * dt)
    M = tm.stability_matrix
    expected = expm(M * dt)
    assert_allclose(T, expected, rtol=1e-10)


def test_compute_euler(tm):
    dt = 0.01
    T = tm.compute(dt, method="euler")
    M = tm.stability_matrix
    expected = np.eye(M.shape[0]) + dt * M
    assert_allclose(T, expected, rtol=1e-10)


def test_eigenvalues(tm):
    dt = 0.1
    eigs = tm.eigenvalues(dt)
    assert len(eigs) == 2  # 2-coupling system
    assert all(np.isfinite(eigs))


def test_critical_exponents_from_transfer(tm, stability):
    dt = 0.1
    theta_transfer = tm.critical_exponents_from_transfer(dt)
    theta_stability = stability.critical_exponents

    # Sort both by real part for comparison
    t_sorted = np.sort(theta_transfer.real)
    s_sorted = np.sort(theta_stability.real)
    assert_allclose(t_sorted, s_sorted, rtol=1e-4)


def test_power_iteration(tm):
    dt = 0.1
    n_steps = 10
    initial = np.array([0.01, 0.002])
    traj = tm.power_iteration(n_steps, initial, dt)
    assert traj.shape == (n_steps + 1, 2)
    # First row should be the initial condition
    assert_allclose(traj[0], initial, rtol=1e-10)
