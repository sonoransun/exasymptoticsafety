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

    def test_residue_matches_amplitude_limit(self):
        # Res_{alpha(s)=n} A = -(1/n!) prod_{k=1..n}(alpha(t)+k): the sign
        # is uniform in n (no (-1)^n alternation).  Compare the closed form
        # against the residue extracted numerically from the amplitude.
        for n in (1, 2, 3):
            for t in (-2.7, -3.9, -0.6):
                s_n = n - 1.0  # alpha0 = alphap = 1
                r1 = 1e-5 * B.veneziano(s_n + 1e-5, t)
                r2 = 1e-6 * B.veneziano(s_n + 1e-6, t)
                numeric = float(np.real((10.0 * r2 - r1) / 9.0))
                assert B.veneziano_residue(n, t) == pytest.approx(
                    numeric, abs=1e-8
                )

    def test_residue_sign_uniform_in_n(self):
        # n=1, t=-2.7 (alpha(t)=-1.7): Res = -(alpha(t)+1) = +0.7 exactly;
        # the old -(-1)^n convention returned -0.7 at odd n.
        assert B.veneziano_residue(1, -2.7) == pytest.approx(0.7, rel=1e-12)
        assert B.veneziano_residue(3, -2.7) == pytest.approx(
            -(-1.7 + 1) * (-1.7 + 2) * (-1.7 + 3) / 6.0, rel=1e-12
        )


class TestVirasoroShapiro:
    """A(s,t,u) = Gamma(-a_s)Gamma(-a_t)Gamma(-a_u) / prod Gamma(1+a_i),
    massless closed-string convention alpha0=0 (sum a_i = 0 on s+t+u=0)."""

    def test_full_crossing_symmetry(self):
        s, t, u = 2.1, -1.3, -0.8  # massless surface s+t+u=0
        a = B.virasoro_shapiro(s, t, u)
        assert abs(a - B.virasoro_shapiro(t, s, u)) < 1e-12
        assert abs(a - B.virasoro_shapiro(u, t, s)) < 1e-12
        assert abs(a - B.virasoro_shapiro(t, u, s)) < 1e-12

    def test_reference_values(self):
        # mpmath (dps=40) references for Gamma(-a_i)/Gamma(1+a_i) with
        # alpha0=0, alphap=0.25 on u=-s-t (not identically 1, the failure
        # mode of the old Gamma(a_i+a_j) denominator form).
        a = B.virasoro_shapiro(2.3, -1.1, -1.2)
        assert complex(a) == pytest.approx(-24.404952827835192, rel=1e-12)
        b = B.virasoro_shapiro(7.7, -3.3, -4.4)
        assert complex(b) == pytest.approx(-0.07382954486058299, rel=1e-12)

    def test_regge_tower_of_poles(self):
        # Infinite tower: a pole at every level s_n = n/alphap = 4n.
        cos = 0.3
        for n in range(1, 7):
            sn = 4.0 * n
            t_n, u_n = -(sn / 2) * (1 - cos), -(sn / 2) * (1 + cos)
            near = abs(complex(B.virasoro_shapiro(sn + 1e-6, t_n, u_n)))
            s_mid = sn + 2.0
            far = abs(complex(B.virasoro_shapiro(
                s_mid, -(s_mid / 2) * (1 - cos), -(s_mid / 2) * (1 + cos)
            )))
            assert near > 1e3 * far

    def test_level_residues(self):
        # Res_{a_s=n} (in a_s, on sum a_i = 0) = [prod_{k=1..n-1}(a_t+k)]^2
        # / (n!)^2: positive, degree 2(n-1) -- the higher-spin tower grows
        # with the level.
        from math import factorial

        for n in (1, 2, 3):
            sn = 4.0 * n
            for t in (-3.7, -6.1):
                u = -sn - t
                r1 = (0.25 * 1e-5) * B.virasoro_shapiro(sn + 1e-5, t, u - 1e-5)
                r2 = (0.25 * 1e-6) * B.virasoro_shapiro(sn + 1e-6, t, u - 1e-6)
                numeric = float(np.real((10.0 * r2 - r1) / 9.0))
                a_t = 0.25 * t
                closed = 1.0
                for k in range(1, n):
                    closed *= (a_t + k)
                closed = closed**2 / factorial(n) ** 2
                assert numeric == pytest.approx(closed, rel=1e-6)
                assert numeric > 0.0


class TestStringAmplitudeAdapter:
    def test_veneziano_adapter(self):
        sa = B.StringAmplitude(kind="veneziano", alpha0=1.0, alphap=1.0)
        assert np.isfinite(abs(sa.eval(0.3, -0.4)))

    def test_per_kind_alpha0_defaults(self):
        # Massless closed string for VS (alpha0=0 so sum a_i = 0 on
        # u = -s-t); open bosonic string for Veneziano (alpha0=1).
        assert B.StringAmplitude(kind="virasoro_shapiro").alpha0 == 0.0
        assert B.StringAmplitude(kind="veneziano").alpha0 == 1.0

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

    def test_virasoro_shapiro_ultrasoft_pole_free_window(self):
        # Sample the dual ladder s = (n+1/2)/alphap, midway between the
        # Regge poles s_n = 4n: the envelope decay is super-polynomial
        # (slope keeps steepening), not a pole-spike fit artefact.
        sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
        s_mid = (np.arange(1, 26) + 0.5) / 0.25
        out = B.ultrasoft_falloff(sa, cos_theta=0.3, s_values=s_mid)
        assert out["ultrasoft"]
        assert out["slope_hi"] < out["slope_lo"] - 5.0
        assert out["slope_hi"] < -20.0

    def test_constant_amplitude_not_ultrasoft(self):
        class Const:
            def amplitude_vs_s(self, s, c, dressed=True):
                return np.full_like(np.asarray(s, float), 3.0)

        out = B.ultrasoft_falloff(Const(), s_lo=5.0, s_hi=50.0)
        assert not out["super_polynomial"]
        assert not out["ultrasoft"]
