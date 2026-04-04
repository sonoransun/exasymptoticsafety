"""Tests for matrix exponential flow."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.linear.matrix_exponential import MatrixExponentialFlow


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
def mat_exp(stability):
    return MatrixExponentialFlow(stability)


def test_propagate_shape(mat_exp):
    delta_g0 = np.array([0.01, 0.002])
    t_values = np.linspace(0, 1, 20)
    result = mat_exp.propagate(delta_g0, t_values)
    assert result.shape == (20, 2)


def test_propagate_at_zero(mat_exp):
    delta_g0 = np.array([0.01, 0.002])
    t_values = np.array([0.0])
    result = mat_exp.propagate(delta_g0, t_values)
    assert_allclose(result[0], delta_g0, rtol=1e-10)


def test_eigendecomposition(mat_exp):
    V, Theta = mat_exp.eigendecomposition()
    M = mat_exp.stability_matrix
    # M = V @ diag(-Theta) @ V^{-1}
    reconstructed = V @ np.diag(-Theta) @ np.linalg.inv(V)
    assert_allclose(reconstructed, M, rtol=1e-6)


def test_compare_with_flow(mat_exp, eh_system, ngfp):
    # Integrate a trajectory very close to the FP with tiny perturbation
    # and short integration time to stay in the linear regime
    integrator = FlowIntegrator(eh_system)
    ic = {
        "g": ngfp.location["g"] + 1e-6,
        "lambda": ngfp.location["lambda"] + 1e-7,
    }
    traj = integrator.integrate(ic, t_span=(0, 0.1), t_eval=np.linspace(0, 0.1, 20))
    comp = mat_exp.compare_with_flow(traj)
    # For a very small perturbation, linearised and full should agree
    assert comp.agrees
