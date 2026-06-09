"""Tests for the cosmology / RG-improved gravity module."""

from __future__ import annotations

import numpy as np
import pytest

from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator, RGTrajectory
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW
from asymsafety.cosmology.scale_identification import (
    GeodesicDistanceScale,
    HubbleScale,
    InverseDistanceScale,
    ProperDistanceScale,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eh_trajectory() -> RGTrajectory:
    """An RGTrajectory flowing from the UV NGFP toward the IR."""
    sys = build_eh_beta_system(d=4)
    fp = FixedPointFinder(sys).find_fixed_point({"g": 0.7, "lambda": 0.14})
    ic_uv = {
        "g": fp.location["g"] - 0.001,
        "lambda": fp.location["lambda"] + 0.001,
    }
    flow = FlowIntegrator(sys)
    return flow.integrate(ic_uv, t_span=(10, -10), max_step=0.05)


# ---------------------------------------------------------------------------
# Scale identification
# ---------------------------------------------------------------------------

class TestScaleIdentification:
    def test_inverse_distance_scalar(self):
        s = InverseDistanceScale(xi=2.0)
        assert s.k_of_r(0.5) == pytest.approx(4.0)

    def test_inverse_distance_array(self):
        s = InverseDistanceScale(xi=1.0)
        result = s.k_of_r(np.array([1.0, 2.0, 4.0]))
        assert np.allclose(result, [1.0, 0.5, 0.25])

    def test_geodesic_softens_origin(self):
        """delta > 0 keeps k(0) finite."""
        s = GeodesicDistanceScale(xi=1.0, delta=0.1)
        k_at_origin = s.k_of_r(0.0)
        assert np.isfinite(k_at_origin)
        assert k_at_origin == pytest.approx(10.0)

    def test_geodesic_reduces_to_inverse_at_large_r(self):
        s_geo = GeodesicDistanceScale(xi=1.0, delta=0.01)
        s_inv = InverseDistanceScale(xi=1.0)
        for r in [10.0, 100.0]:
            assert s_geo.k_of_r(r) == pytest.approx(s_inv.k_of_r(r), rel=1e-3)

    def test_proper_distance_at_horizon_finite(self):
        s = ProperDistanceScale(xi=1.0, r_h=2.0, delta=0.01)
        assert np.isfinite(s.k_of_r(2.0))

    def test_proper_distance_outside_horizon(self):
        s = ProperDistanceScale(xi=1.0, r_h=1.0, delta=0.0)
        assert s.k_of_r(2.0) == pytest.approx(1.0)

    def test_hubble_scale_requires_hubble(self):
        s = HubbleScale(xi=1.0)
        with pytest.raises(ValueError, match="hubble"):
            s.k_of_t(1.0)

    def test_hubble_scale_with_hubble(self):
        s = HubbleScale(xi=2.0)
        assert s.k_of_t(1.0, hubble=3.0) == pytest.approx(6.0)

    def test_hubble_scale_no_radial(self):
        with pytest.raises(NotImplementedError):
            HubbleScale(xi=1.0).k_of_r(1.0)


# ---------------------------------------------------------------------------
# RG-improved Schwarzschild
# ---------------------------------------------------------------------------

class TestRGImprovedSchwarzschild:
    """Bonanno-Reuter regularization: G → 0 at r → 0, f → 1 at origin."""

    def test_constructor_requires_g_coupling(self, eh_trajectory):
        # Hand-build a trajectory missing 'g'
        from asymsafety.analysis.flow import RGTrajectory
        bad = RGTrajectory(
            t_values=np.array([0.0]),
            coupling_values={"foo": np.array([0.0])},
            coupling_names=["foo"],
        )
        with pytest.raises(ValueError, match="'g'"):
            RGImprovedSchwarzschild(trajectory=bad, M=1.0)

    def test_running_G_finite_at_small_r(self, eh_trajectory):
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        G_small = bh.G(1e-3)
        assert np.isfinite(G_small)
        assert G_small > 0

    def test_running_G_vanishes_at_origin(self, eh_trajectory):
        """Asymptotic safety: G(r→0) → 0."""
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        G_at = [bh.G(r) for r in [1e-1, 1e-2, 1e-3, 1e-4]]
        # Strictly decreasing toward zero
        for a, b in zip(G_at, G_at[1:]):
            assert b < a

    def test_lapse_to_unity_at_origin(self, eh_trajectory):
        """f(r) → 1 as r → 0 (softened core; curvature still ∝ 1/r — see
        the rg_improved_bh module docstring, HV-15a-b)."""
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        assert bh.lapse(1e-4) == pytest.approx(1.0, abs=0.01)

    def test_lapse_classical_far_field(self, eh_trajectory):
        """At intermediate r, the lapse departs from 1 (gravitational well)."""
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        f_close = bh.lapse(1.0)
        assert f_close < 1.0

    def test_horizon_exists_for_large_M(self, eh_trajectory):
        bh = RGImprovedSchwarzschild(eh_trajectory, M=10.0, k0=1.0)
        h = bh.horizons(1e-3, 100.0, 5000)
        assert len(h) >= 1

    def test_no_horizon_below_critical_mass(self, eh_trajectory):
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1e-4, k0=1.0)
        h = bh.horizons(1e-3, 100.0, 5000)
        assert len(h) == 0

    def test_critical_mass_finite_and_positive(self, eh_trajectory):
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        Mc = bh.critical_mass(M_search=(1e-4, 5.0))
        assert Mc is not None
        assert 0 < Mc < 1.0  # tiny remnant in our normalization

    def test_metric_components_consistent(self, eh_trajectory):
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0)
        comps = bh.metric_components(np.array([0.5, 1.0, 2.0]))
        # g_tt = -f, g_rr = 1/f → g_tt * g_rr = -1 (signature)
        assert np.allclose(comps["g_tt"] * comps["g_rr"], -1.0)

    def test_lambda_term_off_by_default(self, eh_trajectory):
        """include_lambda=False matches the no-Λ formula."""
        bh = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0,
                                       include_lambda=False)
        # Lapse = 1 - 2GM/r exactly
        r = 1.0
        expected = 1.0 - 2.0 * bh.G(r) * 1.0 / r
        assert bh.lapse(r) == pytest.approx(expected, rel=1e-9)

    def test_lambda_term_on_modifies_lapse(self, eh_trajectory):
        bh_no_lam = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0,
                                              include_lambda=False)
        bh_lam = RGImprovedSchwarzschild(eh_trajectory, M=1.0, k0=1.0,
                                          include_lambda=True)
        # With Λ, the lapse is smaller (at r where Λ r²/3 > 0)
        r = 1.0
        assert bh_lam.lapse(r) <= bh_no_lam.lapse(r)


# ---------------------------------------------------------------------------
# RG-improved FLRW
# ---------------------------------------------------------------------------

class TestRGImprovedFLRW:
    def test_constructor_requires_g(self, eh_trajectory):
        from asymsafety.analysis.flow import RGTrajectory
        bad = RGTrajectory(
            t_values=np.array([0.0]),
            coupling_values={"foo": np.array([0.0])},
            coupling_names=["foo"],
        )
        with pytest.raises(ValueError, match="'g'"):
            RGImprovedFLRW(trajectory=bad)

    def test_G_lookup_returns_positive(self, eh_trajectory):
        flrw = RGImprovedFLRW(eh_trajectory, scale=InverseDistanceScale(xi=1.0))
        G_val = flrw.G(1.0)
        assert G_val > 0
        assert np.isfinite(G_val)

    def test_integrate_returns_expected_keys(self, eh_trajectory):
        flrw = RGImprovedFLRW(eh_trajectory, scale=InverseDistanceScale(xi=1.0))
        result = flrw.integrate(a0=0.1, t_span=(0.1, 1.0), rho0=1.0)
        for key in ("t", "a", "rho", "H", "G", "Lambda"):
            assert key in result
            assert isinstance(result[key], np.ndarray)

    def test_integrate_expanding_universe(self, eh_trajectory):
        """Scale factor should grow from initial value."""
        flrw = RGImprovedFLRW(eh_trajectory, scale=InverseDistanceScale(xi=1.0))
        result = flrw.integrate(a0=0.1, t_span=(0.1, 2.0), rho0=1.0)
        assert result["a"][-1] > result["a"][0]

    def test_integrate_density_decreases(self, eh_trajectory):
        flrw = RGImprovedFLRW(eh_trajectory, scale=InverseDistanceScale(xi=1.0))
        result = flrw.integrate(a0=0.1, t_span=(0.1, 2.0), rho0=1.0)
        assert result["rho"][-1] < result["rho"][0]


# ---------------------------------------------------------------------------
# Trajectory at_scale bug regression
# ---------------------------------------------------------------------------

class TestTrajectoryDescendingT:
    """at_scale must work whether the trajectory was integrated UV→IR or IR→UV."""

    def test_descending_t_interpolation(self, eh_trajectory):
        """Trajectory built with t_span=(10, -10) returns NGFP at large t."""
        c_uv = eh_trajectory.at_scale(9.0)
        c_ir = eh_trajectory.at_scale(-9.0)
        # NGFP-like at UV
        assert c_uv["g"] > 0.5
        # Smaller g at IR
        assert c_ir["g"] < c_uv["g"]
