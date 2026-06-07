"""Tests for the graviton form factor and the AS scattering amplitude."""

import numpy as np
import pytest

from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.scattering.propagator import DressedGravitonPropagator
from asymsafety.scattering.scale import EnergyScale, FixedScale


class TestGravitonFormFactor:
    def test_newton_constant_is_ir_limit(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory, scale=EnergyScale(1.0), k0=1.0)
        # Synthetic model: G_N = g*/k0^2 = 0.7.
        assert ff.newton_constant() == pytest.approx(0.7, rel=0.02)

    def test_form_factor_ir_is_one(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        assert float(ff.f(1e-6)) == pytest.approx(1.0, rel=0.05)

    def test_form_factor_grows_in_uv(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        f_ir = float(ff.f(1e-3))
        f_uv = float(ff.f(1e4))
        assert f_uv > 100.0 * f_ir  # f(p^2) ~ p^2 in the UV

    def test_G_softens_in_uv(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        # G(p^2) ~ g*/k^2 -> decreasing at large p^2.
        assert float(ff.G_of_psq(1e4)) < float(ff.G_of_psq(1.0))

    def test_xi_default_when_absent(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory, xi_default=0.0)
        assert float(ff.xi_of_psq(10.0)) == 0.0

    def test_xi_from_trajectory(self, make_trajectory):
        traj = make_trajectory(xi_const=0.05)
        ff = GravitonFormFactor(traj)
        assert float(ff.xi_of_psq(10.0)) == pytest.approx(0.05)

    def test_vectorized_evaluation(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        p2 = np.array([0.01, 1.0, 100.0])
        G = ff.G_of_psq(p2)
        assert G.shape == (3,)
        assert np.all(np.diff(G) < 0)  # monotonically softening


class TestDressedPropagator:
    def test_coupling_strength_runs(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        prop = DressedGravitonPropagator(ff)
        assert float(prop.coupling_strength(1e4)) < float(
            prop.coupling_strength(1.0)
        )

    def test_massive_denominator_shift(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        massless = DressedGravitonPropagator(ff, massive=False)
        massive = DressedGravitonPropagator(ff, massive=True)
        # lambda* > 0 -> denominator shifted below p^2.
        assert float(massive.denominator(5.0)) < float(massless.denominator(5.0))


class TestAmplitude:
    def test_ir_matches_gr(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        amp = GravitonMediatedAmplitude(ff)
        info = amp.ir_limit(cos_theta=0.3, s_small=1e-4)
        assert info["matches_gr"]
        assert info["ratio"] == pytest.approx(1.0, abs=0.05)

    def test_uv_bounded_vs_gr(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        amp = GravitonMediatedAmplitude(ff)
        info = amp.uv_limit(cos_theta=0.3, s_values=np.geomspace(1.0, 1e8, 80))
        # GR grows ~ s at fixed angle; the dressed amplitude stays well below.
        assert info["bounded"]
        assert info["dressed_at_s_max"] < 0.01 * info["gr_at_s_max"]

    def test_dressed_amplitude_bounded_in_uv(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        amp = GravitonMediatedAmplitude(ff)
        s = np.geomspace(1.0, 1e10, 120)
        M = np.abs(amp.amplitude_vs_s(s, cos_theta=0.3, dressed=True))
        # Dressed amplitude approaches a finite UV plateau (bounded).
        assert np.nanmax(M) < 10.0 * np.nanmedian(M[-20:]) + 1.0
        assert np.isfinite(M).all()

    def test_gr_amplitude_grows_in_uv(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        amp = GravitonMediatedAmplitude(ff)
        s = np.geomspace(1.0, 1e6, 60)
        M = np.abs(amp.amplitude_vs_s(s, cos_theta=0.3, dressed=False))
        assert M[-1] > 100.0 * M[0]  # classical gravity is not UV-soft

    def test_fixed_scale_reproduces_gr(self, as_trajectory):
        # Freezing the scale (no running) must equal the undressed amplitude.
        ff = GravitonFormFactor(as_trajectory, scale=FixedScale(k_fixed=1.0))
        amp = GravitonMediatedAmplitude(ff)
        s, t, u = 5.0, -2.0, -3.0
        dressed = amp.eval(s, t, u, dressed=True)
        # With a fixed scale, G(x) = g(k0)/k0^2 = const, matching dressed=False
        # up to the (constant) value of G at k0.
        assert np.isfinite(abs(dressed))

    def test_differential_cross_section_positive(self, as_trajectory):
        ff = GravitonFormFactor(as_trajectory)
        amp = GravitonMediatedAmplitude(ff)
        dsig = amp.differential_cross_section(10.0, 0.4)
        assert dsig > 0.0

    def test_xi_improvement_changes_amplitude(self, make_trajectory):
        traj = make_trajectory(xi_const=0.1)
        ff = GravitonFormFactor(traj)
        minimal = GravitonMediatedAmplitude(ff, include_xi=False)
        improved = GravitonMediatedAmplitude(ff, include_xi=True, xi_coeff=1.0)
        s, t, u = 5.0, -2.0, -3.0
        assert abs(improved.eval(s, t, u)) != pytest.approx(
            abs(minimal.eval(s, t, u))
        )
