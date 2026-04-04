"""Tests for wavelet transform."""
from __future__ import annotations

import numpy as np
import pytest

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.transforms.integral.wavelet import RGFlowWavelet


@pytest.fixture
def trajectory():
    system = build_eh_beta_system(d=4)
    integrator = FlowIntegrator(system)
    traj = integrator.integrate(
        {"g": 0.5, "lambda": 0.1},
        t_span=(0, 2),
        t_eval=np.linspace(0, 2, 50),
    )
    return traj


@pytest.fixture
def wavelet(trajectory):
    return RGFlowWavelet(trajectory)


def test_wavelet_morlet(wavelet):
    scales = np.array([0.1, 0.5, 1.0])
    result = wavelet.transform(scales, wavelet_type="morlet")
    n_positions = len(result.positions)
    assert result.coefficients.shape == (2, 3, n_positions)
    assert len(result.coupling_names) == 2


def test_wavelet_mexican_hat(wavelet):
    scales = np.array([0.1, 0.5, 1.0])
    result = wavelet.transform(scales, wavelet_type="mexican_hat")
    n_positions = len(result.positions)
    assert result.coefficients.shape == (2, 3, n_positions)


def test_scalogram(wavelet):
    scales, positions, energy = wavelet.scalogram(n_scales=8)
    assert len(scales) == 8
    assert len(positions) > 0
    # energy shape: (n_couplings, n_scales, n_positions)
    assert energy.shape == (2, 8, len(positions))
    # Energy should be non-negative
    assert np.all(energy >= 0)


def test_wavelet_coefficients_finite(wavelet):
    scales = np.array([0.2, 0.5])
    result = wavelet.transform(scales, wavelet_type="morlet")
    assert np.all(np.isfinite(result.coefficients))
