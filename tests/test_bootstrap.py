"""Tests for the string-amplitude bootstrap (Strings from Almost Nothing)."""

import numpy as np
import pytest

from asymsafety.scattering import bootstrap as B


class TestReggeSpectrum:
    def test_trajectory(self):
        assert B.regge_trajectory(3.0, alpha0=1.0, alphap=0.5) == pytest.approx(2.5)

    def test_open_string_spectrum(self):
        spec = B.mass_spectrum(5, alpha0=1.0, alphap=1.0)
        assert np.allclose(spec, [-1, 0, 1, 2, 3, 4])

    def test_spectrum_formula(self):
        # x_n = (n - alpha0)/alphap
        spec = B.mass_spectrum(3, alpha0=0.5, alphap=0.25)
        assert spec[2] == pytest.approx((2 - 0.5) / 0.25)

    def test_max_spin(self):
        assert [B.max_spin_at_level(n) for n in range(4)] == [0, 1, 2, 3]


class TestVeneziano:
    def test_crossing_symmetric(self):
        assert abs(B.veneziano(3.3, -1.7) - B.veneziano(-1.7, 3.3)) < 1e-12

    def test_pole_at_integer_trajectory(self):
        # alpha(s) = 2 at s = 1 (alpha0=alphap=1) -> Gamma pole.
        near = abs(B.veneziano(1.0 - 1e-5, -0.3))
        far = abs(B.veneziano(0.5, -0.3))
        assert near > 1e3 * far

    def test_residue_is_degree_n_polynomial(self):
        t = np.linspace(-3.0, 3.0, 40)
        for n in (1, 2, 3):
            r = B.veneziano_residue(n, t)
            # exact degree-n fit; degree-(n-1) fit cannot reproduce it
            fit_n = np.polyval(np.polyfit(t, r, n), t)
            assert np.max(np.abs(r - fit_n)) < 1e-6
            if n >= 1:
                fit_lo = np.polyval(np.polyfit(t, r, n - 1), t)
                assert np.max(np.abs(r - fit_lo)) > 1e-6

    def test_residue_zeros(self):
        for n in (1, 2, 3):
            zeros = B.residue_zeros(n)
            assert len(zeros) == n
            assert np.allclose(B.veneziano_residue(n, zeros), 0.0, atol=1e-9)


class TestVirasoroShapiro:
    def test_full_crossing_symmetry(self):
        s, t, u = 2.1, -1.3, -2.8
        a = B.virasoro_shapiro(s, t, u)
        assert abs(a - B.virasoro_shapiro(t, s, u)) < 1e-12
        assert abs(a - B.virasoro_shapiro(u, t, s)) < 1e-12
        assert abs(a - B.virasoro_shapiro(t, u, s)) < 1e-12


class TestStringAmplitudeAdapter:
    def test_veneziano_adapter(self):
        sa = B.StringAmplitude(kind="veneziano", alpha0=1.0, alphap=1.0)
        assert np.isfinite(abs(sa.eval(0.3, -0.4)))

    def test_vs_adapter_vs_s(self):
        sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
        A = sa.amplitude_vs_s(np.linspace(5.0, 30.0, 10), cos_theta=0.3)
        assert np.isfinite(np.abs(A)).all()


class TestUltrasoft:
    def test_virasoro_shapiro_is_ultrasoft(self):
        sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
        out = B.ultrasoft_falloff(sa, cos_theta=0.3, s_lo=5.0, s_hi=50.0)
        assert out["super_polynomial"]
        assert out["ultrasoft"]
        assert out["slope_hi"] < out["slope_lo"]

    def test_constant_amplitude_not_ultrasoft(self):
        class Const:
            def amplitude_vs_s(self, s, c, dressed=True):
                return np.full_like(np.asarray(s, float), 3.0)

        out = B.ultrasoft_falloff(Const(), s_lo=5.0, s_hi=50.0)
        assert not out["super_polynomial"]
        assert not out["ultrasoft"]
