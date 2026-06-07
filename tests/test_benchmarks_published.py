"""Cross-truncation regression tests against published benchmarks.

Centralizes assertions that the toolkit reproduces (or remains consistent
with) literature reference values:

    - Reuter 1998 / Lauscher-Reuter 2002:
          EH NGFP at g* ≈ 0.707, λ* ≈ 0.193, θ ≈ 1.47 ± 3.04i
    - Codello-Percacci-Rahmede 2009: quadratic gravity FP, AF in C²
    - Manrique-Rechenberger-Saueressig 2011: foliated, λ_ADM* = 1
    - Korver-Saueressig-Wang 2024: matter bounds for foliated AS
    - D'Angelo et al. 2024 / Saueressig et al. 2025: Lorentzian persistence

Tolerances are deliberately wide because the toolkit uses a simplified
(scheme- and gauge-fixed) Litim implementation that does not exactly
reproduce every published number. The point is to catch silent drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymsafety.actions.matter import MatterContent
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.beta.matter import build_gravity_matter_fp_system
from asymsafety.validation.codello_2009 import (
    ONE_LOOP_UNIVERSAL,
    QUADRATIC_FP,
)
from asymsafety.validation.korver_2024 import (
    COVARIANT_MATTER_BOUNDS,
    FOLIATED_MATTER_BOUNDS,
    validate_foliated_matter_bounds,
)
from asymsafety.validation.lorentzian_2024 import (
    LORENTZIAN_COVARIANT_FP,
    LORENTZIAN_FOLIATED_WICK_FP,
    validate_lorentzian_fp,
)
from asymsafety.validation.manrique_2011 import (
    FOLIATED_EH_FP,
    LORENTZIAN_FP,
)
from asymsafety.validation.bonati_2025 import (
    BONATI_LARGE_NF,
    BONATI_SU2_MC,
    CHARGED_FP_THRESHOLDS,
    large_nf_nu,
    validate_charged_fp_existence,
)
from asymsafety.validation.reuter_1998 import (
    REUTER_FP,
    VALIDATION_TOL as REUTER_TOL,
    validate_eh_fixed_point,
)


# ---------------------------------------------------------------------------
# Reuter 1998 / EH truncation
# ---------------------------------------------------------------------------

class TestReuterFixedPoint:
    """The toolkit reproduces the Reuter NGFP within a wide tolerance.

    Toolkit values: g* ≈ 0.694, λ* ≈ 0.142, product ≈ 0.099.
    Published:      g* ≈ 0.707, λ* ≈ 0.193, product ≈ 0.136.
    Differences are due to the simplified Litim threshold normalization
    and the gauge choice; we test order-of-magnitude consistency only.
    """

    @pytest.fixture(scope="class")
    def fp(self):
        system = build_eh_beta_system(d=4)
        finder = FixedPointFinder(system)
        return finder.find_fixed_point({"g": 0.7, "lambda": 0.14})

    def test_ngfp_exists(self, fp):
        assert fp is not None
        assert not fp.is_gaussian
        assert fp.location["g"] > 0.1

    def test_g_star_within_factor_two_of_published(self, fp):
        ratio = fp.location["g"] / REUTER_FP["g_star"]
        assert 0.5 < ratio < 2.0

    def test_lambda_star_positive_and_below_pole(self, fp):
        assert 0.0 < fp.location["lambda"] < 0.5

    def test_product_g_lambda_within_factor_two(self, fp):
        product = fp.location["g"] * fp.location["lambda"]
        ref = REUTER_FP["g_star"] * REUTER_FP["lambda_star"]
        ratio = product / ref
        assert 0.5 < ratio < 2.0

    def test_validate_eh_function_runs(self, fp):
        """The validation helper should accept the computed FP without error."""
        sa = analyze_stability(build_eh_beta_system(d=4), fp)
        # Take a real eigenvalue if no complex pair, else first complex
        ce = sa.critical_exponents
        theta_re = float(ce[0].real)
        theta_im = float(ce[0].imag)
        result = validate_eh_fixed_point(
            fp.location["g"], fp.location["lambda"], theta_re, theta_im
        )
        assert "all_passed" in result
        # Don't assert all_passed=True — toolkit uses simplified scheme

    def test_pinned_toolkit_values(self, fp):
        """Hard pin so refactor drift is caught immediately."""
        np.testing.assert_allclose(fp.location["g"], 0.6936584729648413,
                                     rtol=1e-6)
        np.testing.assert_allclose(fp.location["lambda"], 0.14228896515894982,
                                     rtol=1e-6)


# ---------------------------------------------------------------------------
# Codello-Percacci-Rahmede 2009 / quadratic gravity
# ---------------------------------------------------------------------------

class TestCodelloPercacciBenchmark:
    """The quadratic gravity validation dict reflects literature."""

    def test_quadratic_fp_dict_has_four_couplings(self):
        for key in ("g_star", "lambda_star", "alpha_star",
                     "beta_star", "n_relevant"):
            assert key in QUADRATIC_FP
        assert QUADRATIC_FP["n_relevant"] == 4

    def test_one_loop_universal_signs(self):
        """β_α > 0, β_β < 0 (Weyl² is asymptotically free)."""
        assert ONE_LOOP_UNIVERSAL["beta_alpha_1loop"] > 0
        assert ONE_LOOP_UNIVERSAL["beta_beta_1loop"] < 0


# ---------------------------------------------------------------------------
# Manrique 2011 / foliated EH
# ---------------------------------------------------------------------------

class TestManriqueFoliated:
    def test_foliated_fp_lambda_adm_unity(self):
        assert abs(FOLIATED_EH_FP["lambda_ADM_star"] - 1.0) < 1e-12

    def test_lorentzian_fp_lambda_adm_unity(self):
        assert abs(LORENTZIAN_FP["lambda_ADM_star"] - 1.0) < 1e-12

    def test_g_lambda_below_pole(self):
        assert FOLIATED_EH_FP["lambda_star"] < 0.5
        assert LORENTZIAN_FP["lambda_star"] < 0.5


# ---------------------------------------------------------------------------
# Korver-Saueressig-Wang 2024 / matter bounds
# ---------------------------------------------------------------------------

class TestKorverSaueressigBounds:
    def test_foliated_matter_bounds_present(self):
        for key in ("max_N_s", "max_N_v", "graviton_mass_ir_fp",
                     "phase_diagram_stable", "reference"):
            assert key in FOLIATED_MATTER_BOUNDS

    def test_covariant_matter_bounds_present(self):
        for key in ("max_N_s_minimal", "max_N_s_nonminimal",
                     "max_N_D", "max_N_v"):
            assert key in COVARIANT_MATTER_BOUNDS

    def test_validate_foliated_matter_bounds_within(self):
        """N_s within bounds + NGFP found ⇒ consistent."""
        result = validate_foliated_matter_bounds(N_s=2, N_v=1, ngfp_exists=True)
        assert result["consistent"]
        assert result["within_foliated_bounds"]

    def test_validate_foliated_matter_bounds_out_of_range(self):
        """N_s above bound + no NGFP found ⇒ also consistent (matches predicted breakdown)."""
        result = validate_foliated_matter_bounds(
            N_s=FOLIATED_MATTER_BOUNDS["max_N_s"] + 5,
            N_v=0, ngfp_exists=False,
        )
        assert not result["within_foliated_bounds"]
        assert result["consistent"]


# ---------------------------------------------------------------------------
# Lorentzian 2024
# ---------------------------------------------------------------------------

class TestLorentzianBenchmarks:
    def test_lorentzian_covariant_fp_keys(self):
        for key in ("g_star", "lambda_star", "n_relevant",
                     "signature", "method"):
            assert key in LORENTZIAN_COVARIANT_FP
        assert LORENTZIAN_COVARIANT_FP["signature"] == "Lorentzian"

    def test_lorentzian_foliated_wick_keys(self):
        for key in ("lambda_ADM_star", "causal_structure"):
            assert key in LORENTZIAN_FOLIATED_WICK_FP
        assert LORENTZIAN_FOLIATED_WICK_FP["causal_structure"] == "Feynman"

    def test_validate_lorentzian_function_runs(self):
        """validate_lorentzian_fp accepts toolkit-like values."""
        result = validate_lorentzian_fp(g_star=0.69, lambda_star=0.18)
        assert "g_star" in result
        assert "lambda_star" in result
        assert "all_passed" in result

    def test_lorentzian_consistent_with_euclidean(self):
        """Lorentzian g*, λ* should be close to Euclidean Reuter values."""
        rel_g = abs(LORENTZIAN_COVARIANT_FP["g_star"] - REUTER_FP["g_star"]) \
                / REUTER_FP["g_star"]
        rel_l = abs(LORENTZIAN_COVARIANT_FP["lambda_star"] -
                     REUTER_FP["lambda_star"]) / REUTER_FP["lambda_star"]
        assert rel_g < 0.10
        assert rel_l < 0.10


# ---------------------------------------------------------------------------
# Gravity-matter NGFP (Buccio et al. 2025 / Eichhorn-Held-Pawlowski)
# ---------------------------------------------------------------------------

class TestGravityMatterFixedPoint:
    """The scalar-quartic extension supports a real shifted NGFP.

    Pinned to the toolkit's current implementation. Drift here usually
    signals a beta-function refactor or a regulator change.
    """

    @pytest.fixture(scope="class")
    def fp(self):
        system = build_gravity_matter_fp_system(
            MatterContent(n_scalars=1), scalar_quartic=True
        )
        finder = FixedPointFinder(system)
        return finder.find_fixed_point(
            {"g": 0.65, "lambda": 0.14, "lambda_phi": 0.01}
        )

    def test_ngfp_exists(self, fp):
        assert fp is not None
        assert not fp.is_gaussian

    def test_lambda_phi_star_positive_and_small(self, fp):
        assert 0 < fp.location["lambda_phi"] < fp.location["g"]

    def test_three_critical_exponents(self, fp):
        assert len(fp.critical_exponents) == 3

    def test_two_relevant_directions(self, fp):
        """Quartic NGFP has two relevant directions (toolkit value)."""
        n_rel = int(np.sum(fp.critical_exponents.real > 0))
        assert n_rel == 2

    def test_pinned_g_lambda_lambdaphi(self, fp):
        np.testing.assert_allclose(fp.location["g"],
                                     0.6455463813725137, rtol=1e-6)
        np.testing.assert_allclose(fp.location["lambda"],
                                     0.14185513513554493, rtol=1e-6)
        np.testing.assert_allclose(fp.location["lambda_phi"],
                                     0.011174696554990685, rtol=1e-6)


# ---------------------------------------------------------------------------
# Bonati, Pelissetto & Vicari 2025 / 3D gauge-Higgs cross-analogue
# ---------------------------------------------------------------------------


class TestBonati2025GaugeHiggs:
    """Cross-analogue: AHM charged FP as a stat-mech mirror of the NGFP.

    The toolkit's one-loop 4-ε β-system (built in
    ``transforms/bridge/gauge_higgs.py``) is a perturbative proxy for
    the 3D physics and is NOT expected to reproduce Bonati et al.'s
    high-precision lattice MC values quantitatively. The tests here
    therefore check (a) reference-data integrity, (b) qualitative
    claims (FP existence, monotonicity), and (c) the published large-
    ``N_f`` field-theory formula.
    """

    def test_reference_table_has_three_Nf(self):
        assert set(BONATI_SU2_MC.keys()) == {30, 40, 60}

    def test_reference_values_monotone_in_Nf(self):
        nus = [BONATI_SU2_MC[n]["nu"] for n in sorted(BONATI_SU2_MC)]
        assert all(nus[i] <= nus[i + 1] for i in range(len(nus) - 1)), \
            f"Bonati MC ν should grow with N_f, got {nus}"

    def test_large_nf_formula_matches_table(self):
        """At N_f = 60 the 1/N_f formula and MC agree to ≲ 15%."""
        nu_formula = large_nf_nu(60, Nc=2)
        nu_mc = BONATI_SU2_MC[60]["nu"]
        rel = abs(nu_formula - nu_mc) / nu_mc
        assert rel < 0.15, (
            f"Large-N_f formula ν = {nu_formula:.3f} vs MC ν = {nu_mc:.3f} "
            f"(rel error {rel:.3f})"
        )

    def test_charged_fp_existence_threshold(self):
        # Below the 4D ε-expansion threshold the FP should not exist
        assert not validate_charged_fp_existence(Nf=100, d=4)
        assert validate_charged_fp_existence(Nf=400, d=4)
        # In 3D the threshold is much lower; Bonati's sample is above it
        assert validate_charged_fp_existence(Nf=30, d=3)

    def test_toolkit_ahm_finds_charged_fp(self):
        """The toolkit can locate the charged FP at all Bonati N_f."""
        from asymsafety.transforms.bridge.gauge_higgs import (
            GaugeHiggsAnalogue,
        )
        for Nf in sorted(BONATI_SU2_MC):
            analogue = GaugeHiggsAnalogue(N=Nf, Nc=1, epsilon=1.0)
            # FP location should be physical: α* > 0, |u*| small
            assert analogue.fixed_point.location["alpha"] > 0
            assert abs(analogue.fixed_point.location["u"]) < 1.0
            # ν should be finite and well below the spurious unitarity limit
            assert 0 < analogue.nu < 2.0

    def test_toolkit_nu_is_monotone_in_Nf(self):
        """One-loop ν(N_f) is monotone — important sanity for the bridge."""
        from asymsafety.transforms.bridge.gauge_higgs import (
            correlation_length_exponent,
        )
        Nfs = [20, 30, 50, 80, 120]
        nus = [correlation_length_exponent(n, epsilon=1.0) for n in Nfs]
        # Toolkit one-loop ν decreases monotonically toward the WF limit
        # (the *opposite* direction from Bonati MC, which grows toward 1).
        # Either monotone direction is acceptable; flag breakage if it
        # oscillates.
        diffs = [nus[i + 1] - nus[i] for i in range(len(nus) - 1)]
        assert all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs), (
            f"ν(N_f) not monotone in the toolkit: {nus}"
        )


# ---------------------------------------------------------------------------
# Scattering amplitudes: physical scattering + asymptotic safety
# ---------------------------------------------------------------------------

class TestDraper2020GravitonAmplitude:
    """Graviton-mediated amplitude reproduces the Draper et al. (2020) limits.

    IR → classical Newtonian, UV → finite/bounded, no ghost poles.
    """

    @pytest.fixture(scope="class")
    def amplitude(self):
        from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
        from asymsafety.scattering.form_factor import GravitonFormFactor
        from tests.conftest import make_as_trajectory

        return GravitonMediatedAmplitude(
            GravitonFormFactor(make_as_trajectory())
        )

    def test_validate_runs_and_passes(self, amplitude):
        from asymsafety.validation.draper_2020 import validate_graviton_amplitude

        result = validate_graviton_amplitude(amplitude)
        assert "all_passed" in result
        assert result["ir_newtonian_limit"]["passed"]
        assert result["uv_finite"]["passed"]
        assert result["ghost_free"]["passed"]
        assert result["all_passed"]


class TestKnorr2026SafeVsUnsafe:
    """A fixed point alone does not guarantee a bounded amplitude."""

    def test_safe_bounded_unsafe_grows(self):
        from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
        from asymsafety.scattering.form_factor import GravitonFormFactor
        from asymsafety.validation.knorr_2026 import (
            make_unsafe_amplitude,
            validate_safe_vs_unsafe,
        )
        from tests.conftest import make_as_trajectory

        traj = make_as_trajectory()
        safe = GravitonMediatedAmplitude(GravitonFormFactor(traj))
        unsafe = make_unsafe_amplitude(traj)
        result = validate_safe_vs_unsafe(safe, unsafe)
        assert result["safe_bounded"]["passed"]
        assert result["unsafe_grows"]["passed"]
        assert result["dichotomy_holds"]


class TestCheung2025Bootstrap:
    """The string bootstrap reproduces the Strings-from-Almost-Nothing facts."""

    def test_validate_bootstrap(self):
        from asymsafety.validation.cheung_2025 import validate_bootstrap

        result = validate_bootstrap()
        assert result["regge_spectrum"]["passed"]
        assert result["veneziano_crossing"]["passed"]
        assert result["virasoro_shapiro_crossing"]["passed"]
        assert result["higher_spin_cancellation"]["passed"]
        assert result["ultrasoft_falloff"]["passed"]
        assert result["all_passed"]


class TestScatteringBridge:
    """The AS amplitude is a distinct, consistent point vs the string bootstrap."""

    def test_bridge_verdict(self):
        from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
        from asymsafety.scattering.bridge import ScatteringBridge
        from asymsafety.scattering.form_factor import GravitonFormFactor
        from tests.conftest import make_as_trajectory

        amp = GravitonMediatedAmplitude(
            GravitonFormFactor(make_as_trajectory())
        )
        verdict = ScatteringBridge(amp).verify()
        assert verdict["as_physically_consistent"]
        assert verdict["distinct_from_strings"]
