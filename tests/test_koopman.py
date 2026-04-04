"""Tests for classical Koopman operator (transforms/linear/koopman.py)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.linear.koopman import ClassicalKoopmanOperator


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
def trajectories(eh_system, ngfp):
    integrator = FlowIntegrator(eh_system)
    trajs = []
    # Generate a few short trajectories near the FP
    for dg in [0.01, -0.01, 0.005]:
        ic = {
            "g": ngfp.location["g"] + dg,
            "lambda": ngfp.location["lambda"] + dg * 0.2,
        }
        traj = integrator.integrate(ic, t_span=(0, 1), t_eval=np.linspace(0, 1, 50))
        trajs.append(traj)
    return trajs


@pytest.fixture
def koopman(eh_system, ngfp):
    return ClassicalKoopmanOperator(eh_system, ngfp)


def test_build_dictionary(koopman):
    exponents = koopman._build_dictionary(degree=3)
    # For 2 couplings and degree 3: C(2+0,2)=1, C(2+1,2)=2, C(2+2,2)=3, C(2+3,2)=4
    # Total = 1 + 2 + 3 + 4 = 10
    assert len(exponents) == 10


def test_evaluate_dictionary(koopman):
    exponents = koopman._build_dictionary(degree=3)
    point = np.array([0.1, 0.2])
    vals = ClassicalKoopmanOperator._evaluate_dictionary(point, exponents)
    assert vals.shape == (len(exponents),)
    # First entry is degree 0: 0.1^0 * 0.2^0 = 1.0
    assert_allclose(vals[0], 1.0, rtol=1e-10)


def test_compute_edmd(koopman, trajectories):
    result = koopman.compute_edmd(trajectories, dictionary_degree=3)
    assert result.eigenvalues is not None
    assert len(result.eigenvalues) > 0
    assert result.dictionary_size == 10  # degree 3, 2 couplings


def test_compare_with_stability(koopman, stability):
    comp = koopman.compare_with_stability(stability)
    assert comp.method_a == "koopman_edmd"
    assert comp.method_b == "stability_matrix"
    # For the direct comparison method, values_a == values_b
    assert comp.agrees
