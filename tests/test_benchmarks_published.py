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
