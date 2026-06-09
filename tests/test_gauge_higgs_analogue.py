"""Tests for the 3D Abelian-Higgs cross-analogue bridge.

Exercises :class:`asymsafety.transforms.bridge.gauge_higgs.GaugeHiggsAnalogue`
end-to-end: build a one-loop AHM β-system, find the charged fixed
point, wrap it into a :class:`CrossAnalogueBridge`, and check that
the toolkit's analogue methods (transfer matrix, resolvent) agree on
the AHM critical exponents.

The toolkit's perturbative one-loop scheme is NOT expected to
reproduce lattice MC numbers exactly; at the u₊ charged FP it agrees
with the Bonati et al. (2025) large-N_f form ν = 1 − 9.727/N_f to
~1.5% and with the MC values to <10%, with ν *increasing* toward 1
with N_f. Assertions here cover existence, the exact u₊ root, the
single relevant direction, that trend, and the internal consistency
of the bridge.
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
        # u* is the WF-connected plus root u₊ = (B + √(B²−4AC))/(2A)
        # of (N+4)u² − (ε + 6α*)u + 9α*² = 0 (HLM 1974), not the
        # tricritical minus root.
        A, B, C = 64.0, 1.0 + 6.0 / 60.0, 9.0 / 60.0 ** 2
        u_plus = (B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        np.testing.assert_allclose(guess["u"], u_plus, rtol=1e-12)

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

    def test_exactly_one_relevant_direction(self, analogue_n60):
        """The charged FP has exactly one relevant direction (the mass).

        At the u₊ charged FP the triangular Jacobian gives
        θ = {θ_r, −√disc, −ε}: only the mass direction is relevant.
        (The u₋ root would be tricritical with two relevant directions.)
        """
        ce = analogue_n60.stability.critical_exponents.real
        n_relevant = int(np.sum(ce > 0.0))
        assert n_relevant == 1
        # And the relevant one is the mass exponent θ_r = 1/ν > 1.
        assert ce.max() > 1.0

    def test_nu_finite_and_below_unity(self, analogue_n60):
        """One-loop ν → 1/(2−ε) = 1 from below at large N (ε = 1)."""
        assert 0.5 < analogue_n60.nu < 1.0

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
        """The linear-algebra regression check passes at the tight default tol.

        All three paths derive from the same Jacobian eigendecomposition,
        so they must agree to ~1e-13 (default tol 1e-9), not 0.2.
        """
        result = analogue_n60.bridge.verify_commutativity()
        assert "critical_exponents" in result
        assert "agreements" in result
        # All shipped paths (RG, transfer matrix, resolvent) should be
        # present and agree on this small 3-coupling system.
        for method, agreement in result["agreements"].items():
            assert agreement["agrees"], (
                f"Bridge path {method!r} disagrees with RG: "
                f"max deviation {agreement['max_deviation']:.4e}"
            )
        # The resolvent path is definitionally identical to the RG path.
        assert result["agreements"]["resolvent"]["max_deviation"] == 0.0


class TestQualitativeBonatiTrends:
    """Sanity-check the bridge against published *trends* (not numbers)."""

    @pytest.mark.parametrize("N", [30, 40, 60, 100])
    def test_nu_finite_at_published_Nf(self, N):
        nu = correlation_length_exponent(N, epsilon=1.0)
        assert np.isfinite(nu)
        assert 0 < nu < 2.0

    def test_one_loop_nu_monotone_increasing(self):
        """ν(N_f) must *increase* toward 1 with N_f (Bonati 2025 trend).

        At the u₊ charged FP, ν → 1/(2−ε) = 1 from below — the same
        direction as the lattice MC / large-N_f form ν = 1 − 9.727/N_f
        [arXiv:2410.05823]. All N_f are above the one-loop threshold
        N* ≈ 27.9 (disc > 0), so no fallback branch is exercised.
        """
        Nfs = [30, 40, 60, 100, 200]
        nus = [correlation_length_exponent(n) for n in Nfs]
        diffs = [nus[i + 1] - nus[i] for i in range(len(nus) - 1)]
        assert all(d > 0 for d in diffs), (
            f"ν(N_f) must increase toward 1 with N_f, got {nus}"
        )
        assert all(0.5 < nu < 1.0 for nu in nus)

    def test_nu_matches_bonati_large_nf(self):
        """One-loop ν at u₊ agrees with ν = 1 − 9.727/N_f to ~1.5%."""
        from asymsafety.validation.bonati_2025 import large_nf_nu

        for Nf in (30, 40, 60):
            nu = correlation_length_exponent(Nf, epsilon=1.0)
            rel = abs(nu - large_nf_nu(Nf)) / large_nf_nu(Nf)
            assert rel < 0.015, (
                f"N_f={Nf}: ν={nu:.4f} vs large-N_f {large_nf_nu(Nf):.4f} "
                f"(rel err {rel:.3f})"
            )

    def test_validate_nu_vs_nf_passes(self):
        """The Bonati MC validation (10% rtol + monotonicity) passes."""
        from asymsafety.validation.bonati_2025 import validate_nu_vs_nf

        computed = {
            n: correlation_length_exponent(n, epsilon=1.0)
            for n in (30, 40, 60)
        }
        report = validate_nu_vs_nf(computed)
        assert report["monotonic"]
        assert report["all_passed"], f"Bonati MC validation failed: {report}"
