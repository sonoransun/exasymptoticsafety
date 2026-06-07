"""Tests for the physical-scattering consistency checks."""

import numpy as np
import pytest

from asymsafety.scattering import consistency as C
from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.form_factor import GravitonFormFactor


@pytest.fixture
def amp(as_trajectory):
    return GravitonMediatedAmplitude(GravitonFormFactor(as_trajectory))


class TestPartialWave:
    def test_partial_wave_finite(self, amp):
        a0 = C.partial_wave(amp, 0, 10.0)
        assert np.isfinite(abs(a0))

    def test_higher_waves(self, amp):
        for ell in (0, 1, 2):
            assert np.isfinite(abs(C.partial_wave(amp, ell, 10.0)))


class TestUnitarity:
    def test_as_bounded_gr_unbounded(self, amp):
        out = C.unitarity(amp, s_values=np.geomspace(1e-2, 1e6, 40))
        # AS partial wave plateaus (growth exponent ~ 0); GR grows ~ s.
        assert out["as_bounded"]
        assert out["gr_unbounded"]
        assert out["as_growth_exponent"] < 0.2
        assert out["gr_growth_exponent"] > 0.5
        assert out["passed"]

    def test_gr_dwarfs_as_at_high_energy(self, amp):
        out = C.unitarity(amp, s_values=np.geomspace(1e-2, 1e6, 40))
        assert out["gr_max"] > 1e3 * out["as_max"]


class TestOtherChecks:
    def test_uv_finiteness(self, amp):
        out = C.uv_finiteness(amp)
        assert out["finite"]
        assert out["passed"]

    def test_froissart_as_constant_gr_linear(self, amp):
        as_out = C.froissart_growth(amp, dressed=True)
        gr_out = C.froissart_growth(amp, dressed=False)
        assert as_out["uv_constant"]
        assert as_out["growth_exponent"] < 0.2
        assert gr_out["growth_exponent"] == pytest.approx(1.0, abs=0.1)

    def test_causality_no_ghosts(self, amp):
        out = C.causality_no_ghosts(amp)
        assert out["f_positive"]
        assert out["f_no_ghost_pole"]
        assert out["passed"]

    def test_crossing_exact(self, amp):
        out = C.crossing(amp, s=5.0, t=-2.0)
        assert out["residual"] < 1e-9
        assert out["passed"]

    def test_optical_theorem_tree_real(self, amp):
        out = C.optical_theorem(amp, 10.0)
        assert out["tree_level_real"]

    def test_full_report(self, amp):
        report = C.consistency_report(amp)
        assert report["all_passed"]
        for key in ("unitarity", "uv_finiteness", "froissart",
                    "causality", "crossing"):
            assert report[key]["passed"]
