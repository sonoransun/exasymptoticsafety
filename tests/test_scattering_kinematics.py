"""Tests for Mandelstam kinematics and momentum-scale identification."""

import numpy as np
import pytest

from asymsafety.scattering.kinematics import (
    Mandelstam,
    cm_momentum,
    cos_theta_of_t,
)
from asymsafety.scattering.scale import EnergyScale, FixedScale, TransferScale


class TestMandelstam:
    def test_closure_massless(self):
        m = Mandelstam.from_s_angle(10.0, 0.3, m=0.0)
        assert abs(m.closure_residual) < 1e-9

    def test_closure_massive(self):
        m = Mandelstam.from_s_angle(10.0, -0.4, m=1.2)
        # s + t + u = 4 m^2
        assert m.s + m.t + m.u == pytest.approx(4.0 * 1.2**2)

    def test_from_s_t_fixes_u(self):
        m = Mandelstam.from_s_t(8.0, -3.0, m=0.0)
        assert m.u == pytest.approx(-5.0)

    def test_angle_round_trip(self):
        s, cos_theta = 12.0, 0.62
        m = Mandelstam.from_s_angle(s, cos_theta, m=0.5)
        assert m.cos_theta == pytest.approx(cos_theta)
        assert cos_theta_of_t(s, m.t, m=0.5) == pytest.approx(cos_theta)

    def test_physical_region(self):
        # Backward-ish physical point: s > 0, t < 0, u < 0.
        m = Mandelstam.from_s_angle(20.0, 0.1, m=0.0)
        assert m.is_physical
        # Unphysical: t > 0
        bad = Mandelstam(s=10.0, t=5.0, u=-15.0, m=0.0)
        assert not bad.is_physical

    def test_cm_momentum(self):
        assert cm_momentum(16.0, m=0.0) == pytest.approx(2.0)
        assert cm_momentum(16.0, m=1.0) == pytest.approx(np.sqrt(3.0))

    def test_crossing_swaps_s_t(self):
        m = Mandelstam.from_s_t(7.0, -2.0, m=0.0)
        c = m.crossed_s_t()
        assert (c.s, c.t) == (m.t, m.s)


class TestMomentumScale:
    def test_energy_scale(self):
        scale = EnergyScale(xi=2.0)
        assert scale.k_of_psq(9.0) == pytest.approx(2.0 * 3.0)

    def test_energy_scale_abs_for_spacelike(self):
        scale = EnergyScale(xi=1.0)
        # negative (spacelike) invariant -> uses |p2|
        assert scale.k_of_psq(-4.0) == pytest.approx(2.0)

    def test_transfer_scale_is_energy_scale(self):
        assert TransferScale(xi=1.5).k_of_psq(4.0) == pytest.approx(3.0)

    def test_fixed_scale_constant(self):
        scale = FixedScale(k_fixed=5.0)
        out = scale.k_of_psq(np.array([1.0, 100.0, 1e6]))
        assert np.allclose(out, 5.0)

    def test_vectorized(self):
        scale = EnergyScale(xi=1.0)
        out = scale.k_of_psq(np.array([1.0, 4.0, 9.0]))
        assert np.allclose(out, [1.0, 2.0, 3.0])
