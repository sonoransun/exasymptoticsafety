"""Tests for the 3D Abelian-Higgs cross-analogue bridge.

Exercises :class:`asymsafety.transforms.bridge.gauge_higgs.GaugeHiggsAnalogue`
end-to-end: build a one-loop AHM β-system, find the charged fixed
point, wrap it into a :class:`CrossAnalogueBridge`, and check that
the toolkit's analogue methods (transfer matrix, resolvent) agree on
the AHM critical exponents.

The toolkit's perturbative one-loop scheme is NOT expected to
reproduce lattice MC numbers; assertions here are qualitative
(existence, sign, monotonicity, internal consistency of the bridge).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge
from asymsafety.transforms.bridge.gauge_higgs import (
    GaugeHiggsAnalogue,
    build_ahm_system,
    charged_fp_guess,
    correlation_length_exponent,
)


@pytest.fixture(scope="module")
def analogue_n60():
    """A AHM analogue at N_f = 60 (deep in the stable charged-FP region)."""
    return GaugeHiggsAnalogue(N=60, Nc=1, epsilon=1.0)


class TestAHMBetaSystem:
    """Symbolic + numeric sanity of the one-loop AHM β-system."""

    def test_system_has_three_couplings(self):
        system = build_ahm_system(N=40, Nc=1, epsilon=1.0)
        assert system.coupling_names == ["alpha", "u", "r"]

    def test_charged_fp_guess_matches_analytic(self):
        guess = charged_fp_guess(N=60, epsilon=1.0)
        # alpha* = ε/N for Nc = 1 in the perturbative scheme
        np.testing.assert_allclose(guess["alpha"], 1.0 / 60.0, rtol=1e-12)
        assert guess["r"] == 0.0
        assert 0 < guess["u"] < 1.0

    def test_evaluate_beta_at_fp_is_small(self):
        """β_i should be tiny at the analytic FP guess."""
        system = build_ahm_system(N=60, Nc=1, epsilon=1.0)
        guess = charged_fp_guess(60, epsilon=1.0)
        beta = system.evaluate(guess)
        assert all(abs(v) < 1e-6 for v in beta.values()), (
            f"|β| should vanish at the charged FP guess, got {beta}"
        )


class TestGaugeHiggsAnalogue:
    def test_construction_produces_bridge(self, analogue_n60):
        assert isinstance(analogue_n60.bridge, CrossAnalogueBridge)
        assert analogue_n60.bridge.system is analogue_n60.system
        assert analogue_n60.bridge.fp is analogue_n60.fixed_point

    def test_fp_is_not_gaussian(self, analogue_n60):
        loc = analogue_n60.fixed_point.location
        assert loc["alpha"] > 0
        assert not analogue_n60.fixed_point.is_gaussian

    def test_exactly_one_strongly_relevant_direction(self, analogue_n60):
        """Mass direction θ_r ≈ 2; the other two are < 1 or negative."""
        ce = analogue_n60.stability.critical_exponents.real
        n_strong = int(np.sum(ce > 1.0))
        assert n_strong == 1

    def test_nu_finite_and_below_unity(self, analogue_n60):
        """One-loop ν approaches the WF limit ≈ 1/2 from below at large N."""
        assert 0 < analogue_n60.nu < 1.0

    def test_below_perturbative_threshold_raises(self):
        """Very small N (and Nc=1) makes the gauge β coefficient 0 → no FP."""
        # N = 0 collapses the alpha² coefficient to 0, so the only FP
        # of β_α = −ε α + 0·α² is α = 0 (the trivial GFP), and the
        # closed-form guess returns α* = ε/0 = inf.
        with pytest.raises((RuntimeError, ZeroDivisionError)):
            GaugeHiggsAnalogue(N=0, Nc=1, epsilon=1.0)


class TestBridgeCommutativityOnAHM:
    """The toolkit's RG ↔ transfer-matrix ↔ resolvent paths agree."""

    def test_rg_and_resolvent_match(self, analogue_n60):
        rg = analogue_n60.bridge.rg_critical_exponents()
        res = analogue_n60.bridge.resolvent_poles()
        # Sort by real part descending and compare
        rg_sorted = np.sort(rg.real)[::-1]
        res_sorted = np.sort(res.real)[::-1]
        np.testing.assert_allclose(rg_sorted, res_sorted, atol=1e-6)

    def test_verify_commutativity_returns_dict(self, analogue_n60):
        """The high-level commutativity check completes without error."""
        result = analogue_n60.bridge.verify_commutativity(tol=0.2)
        assert "critical_exponents" in result
        assert "agreements" in result
        # All shipped paths (RG, transfer matrix, resolvent) should be
        # present and agree on this small 3-coupling system.
        for method, agreement in result["agreements"].items():
            assert agreement["agrees"], (
                f"Bridge path {method!r} disagrees with RG: "
                f"max deviation {agreement['max_deviation']:.4f}"
            )


class TestQualitativeBonatiTrends:
    """Sanity-check the bridge against published *trends* (not numbers)."""

    @pytest.mark.parametrize("N", [30, 40, 60, 100])
    def test_nu_finite_at_published_Nf(self, N):
        nu = correlation_length_exponent(N, epsilon=1.0)
        assert np.isfinite(nu)
        assert 0 < nu < 2.0

    def test_one_loop_nu_monotone(self):
        Nfs = [20, 30, 50, 80, 120]
        nus = [correlation_length_exponent(n) for n in Nfs]
        diffs = [nus[i + 1] - nus[i] for i in range(len(nus) - 1)]
        assert all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)
