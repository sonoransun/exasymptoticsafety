"""Tests for Laplace transform of RG flows."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.transforms.integral.laplace import RGFlowLaplace


@pytest.fixture
def trajectory():
    system = build_eh_beta_system(d=4)
    integrator = FlowIntegrator(system)
    traj = integrator.integrate(
        {"g": 0.5, "lambda": 0.1},
        t_span=(0, 2),
        t_eval=np.linspace(0, 2, 100),
    )
    return traj


@pytest.fixture
def laplace(trajectory):
    return RGFlowLaplace(trajectory)


def test_laplace_transform_shape(laplace):
    s_values = np.linspace(0.5, 5.0, 20)
    result = laplace.transform(s_values)
    assert result.transform_values.shape == (2, 20)
    assert len(result.coupling_names) == 2


def test_laplace_transform_real_s(laplace):
    s_values = np.array([1.0, 2.0, 3.0])
    result = laplace.transform(s_values)
    assert np.all(np.isfinite(result.transform_values))


def test_detect_poles(laplace):
    poles = laplace.detect_poles(n_poles=5, s_range=(0.1, 10.0), n_search=200)
    assert isinstance(poles, dict)
    # Should have entries for each coupling
    assert len(poles) == 2


def test_transfer_function(laplace):
    num, den = laplace.transfer_function(n_poles=2)
    assert isinstance(num, np.ndarray)
    assert isinstance(den, np.ndarray)
    assert len(num) > 0
    assert len(den) > 0
    # Denominator should be monic (leading coefficient 1)
    assert_allclose(den[0], 1.0, rtol=1e-10)
