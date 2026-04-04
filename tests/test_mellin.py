"""Tests for Mellin transform."""
from __future__ import annotations

import numpy as np
import pytest

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.transforms.integral.mellin import RGFlowMellin


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
def mellin(trajectory):
    return RGFlowMellin(trajectory)


def test_mellin_transform_shape(mellin):
    s_values = np.linspace(1.0, 5.0, 15)
    result = mellin.transform(s_values)
    assert result.transform_values.shape == (2, 15)
    assert len(result.coupling_names) == 2


def test_mellin_transform_values(mellin):
    s_values = np.array([1.0, 2.0, 3.0])
    result = mellin.transform(s_values)
    assert np.all(np.isfinite(result.transform_values))


def test_scaling_dimensions(mellin):
    dims = mellin.scaling_dimensions(n_search=50)
    assert isinstance(dims, np.ndarray)
    # Dimensions should be sorted ascending
    if len(dims) > 1:
        assert np.all(np.diff(dims) >= 0)
