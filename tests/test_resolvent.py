"""Tests for resolvent operator."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.linear.resolvent import ResolventOperator


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
def resolvent(stability):
    return ResolventOperator(stability)


def test_evaluate_resolvent(resolvent):
    R = resolvent.evaluate(s=10.0 + 0j)
    assert R.shape == (2, 2)
    assert np.all(np.isfinite(R))


def test_resolvent_norm(resolvent):
    norm = resolvent.resolvent_norm(s=10.0 + 0j)
    assert norm > 0.0
    assert np.isfinite(norm)


def test_poles(resolvent, stability):
    poles = resolvent.poles()
    # Poles are eigenvalues of M
    assert_allclose(np.sort(poles.real), np.sort(stability.eigenvalues.real), rtol=1e-10)


def test_pseudospectrum(resolvent):
    real_grid, imag_grid, norm_grid = resolvent.pseudospectrum(
        epsilon=0.1,
        real_range=(-5.0, 5.0),
        imag_range=(-5.0, 5.0),
        n_grid=10,
    )
    assert real_grid.shape == (10,)
    assert imag_grid.shape == (10,)
    assert norm_grid.shape == (10, 10)


def test_poles_match_critical_exponents(resolvent, stability):
    poles = resolvent.poles()
    # -poles should equal critical exponents (theta_i = -eigenvalues of M)
    neg_poles = -poles
    theta = stability.critical_exponents
    assert_allclose(np.sort(neg_poles.real), np.sort(theta.real), rtol=1e-10)
