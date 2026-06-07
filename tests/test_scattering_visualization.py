"""Smoke tests for the scattering-amplitude plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.bridge import ScatteringBridge
from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.visualization import amplitude_plot as AP


@pytest.fixture
def amp(as_trajectory):
    return GravitonMediatedAmplitude(GravitonFormFactor(as_trajectory))


def _is_fig(fig):
    assert isinstance(fig, Figure)
    plt.close(fig)


class TestAmplitudePlots:
    def test_amplitude_vs_energy(self, amp):
        _is_fig(AP.plot_amplitude_vs_energy(amp, s_range=(1e-2, 1e6, 60)))

    def test_form_factor(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        _is_fig(AP.plot_form_factor(ff, p2_range=(1e-3, 1e5, 60)))

    def test_partial_wave_unitarity(self, amp):
        _is_fig(AP.plot_partial_wave_unitarity(amp, s_range=(1e-2, 1e5, 20)))

    def test_regge_trajectory(self):
        _is_fig(AP.plot_regge_trajectory(n_max=4))

    def test_as_vs_string(self, amp):
        bridge = ScatteringBridge(amp)
        _is_fig(AP.plot_as_vs_string(bridge, s_range=(5.0, 40.0, 40)))
