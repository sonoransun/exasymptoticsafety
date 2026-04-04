"""Tests for Fourier transform."""
from __future__ import annotations

import numpy as np
import pytest

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.transforms.integral.fourier import RGFlowFourier


@pytest.fixture
def trajectory():
    system = build_eh_beta_system(d=4)
    integrator = FlowIntegrator(system)
    traj = integrator.integrate(
        {"g": 0.5, "lambda": 0.1},
        t_span=(0, 2),
        t_eval=np.linspace(0, 2, 64),
    )
    return traj


@pytest.fixture
def fourier(trajectory):
    return RGFlowFourier(trajectory)


def test_fourier_transform_shape(fourier, trajectory):
    result = fourier.transform()
    n_freq = len(trajectory.t_values)
    assert result.transform_values.shape == (2, n_freq)
    assert len(result.coupling_names) == 2


def test_power_spectrum(fourier):
    result = fourier.transform()
    # Power spectrum should be non-negative
    assert np.all(result.power_spectrum >= 0)


def test_dominant_frequencies(fourier):
    result = fourier.transform()
    dom_freq = result.dominant_frequencies
    assert isinstance(dom_freq, np.ndarray)
    # Frequencies should be sorted if non-empty
    if len(dom_freq) > 1:
        assert np.all(np.diff(dom_freq) >= 0)


def test_psd_shape(fourier, trajectory):
    frequencies, psd = fourier.power_spectral_density()
    n_freq = len(trajectory.t_values)
    assert psd.shape == (2, n_freq)
    assert frequencies.shape == (n_freq,)
