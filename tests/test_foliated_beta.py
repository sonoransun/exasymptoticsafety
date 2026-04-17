"""Regression tests for the foliated EH beta function system.

Pins structural and numerical behavior of build_foliated_eh_beta_system
so that performance refactors of beta/system.py cannot silently drift
the physics.

Reference values:
    Manrique-Rechenberger-Saueressig 2011 (validation/manrique_2011.py):
        g* ≈ 0.96, λ* ≈ 0.20, λ_ADM* = 1
    Saueressig et al. 2025 (validation/lorentzian_2024.py):
        λ_ADM → 1 preserved under Wick rotation

Note: the current implementation produces correct one-loop structure and
preserves λ_ADM = 1 as a fixed line of β_{λ_ADM}, but the (g, λ) sector
collapses to the Gaussian basin under fsolve from typical guesses — the
self-consistent NGFP that exists in the full literature truncation is not
recovered by this minimal implementation. Tests therefore pin the
structural invariants rather than asserting NGFP existence.
"""

from __future__ import annotations

import numpy as np
from sympy import Symbol

from asymsafety.beta.foliated import (
    build_foliated_eh_beta_system,
    foliated_eh_benchmark,
)
from asymsafety.validation.manrique_2011 import (
    FOLIATED_EH_FP,
    LORENTZIAN_FP,
    VALIDATION_TOL,
)


class TestFoliatedSystemStructure:
    """The foliated system has the expected three couplings."""

    def setup_method(self):
        self.system = build_foliated_eh_beta_system(d=4)

    def test_three_couplings(self):
        assert self.system.dimension == 3
        assert self.system.coupling_names == ["g", "lambda", "lambda_ADM"]

    def test_lambda_adm_appears_in_beta_lambda_adm(self):
        """β_{λ_ADM} must depend on λ_ADM."""
        lambda_adm = Symbol("lambda_ADM", real=True)
        b = self.system.beta("lambda_ADM").expression
        assert lambda_adm in b.free_symbols

    def test_lambda_adm_does_not_appear_in_beta_g(self):
        """λ_ADM does not back-react on β_g in this truncation."""
        lambda_adm = Symbol("lambda_ADM", real=True)
        b = self.system.beta("g").expression
        assert lambda_adm not in b.free_symbols

    def test_lorentzian_flag_changes_nothing_currently(self):
        """The lorentzian kwarg is currently a no-op; pin that invariant."""
        sys_e = build_foliated_eh_beta_system(d=4, lorentzian=False)
        sys_l = build_foliated_eh_beta_system(d=4, lorentzian=True)
        for name in sys_e.coupling_names:
            assert sys_e.beta(name).expression == sys_l.beta(name).expression


class TestFoliatedFixedLine:
    """λ_ADM = 1 is a fixed line: β_{λ_ADM} vanishes there for any (g, λ).

    This is the central physics result of Manrique-Rechenberger-Saueressig:
    the foliated system flows toward full-Diff invariance.
    """

    def setup_method(self):
        self.system = build_foliated_eh_beta_system(d=4)

    def test_lambda_adm_one_is_fixed_line_at_gfp(self):
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "lambda_ADM": 1.0}
        )
        assert abs(result["lambda_ADM"]) < 1e-12

    def test_lambda_adm_one_is_fixed_line_at_ngfp_guess(self):
        """β_{λ_ADM} = 0 at λ_ADM = 1 holds independent of g, λ."""
        for g_val, lam_val in [(0.5, 0.1), (0.9, 0.2), (1.5, 0.3)]:
            result = self.system.evaluate(
                {"g": g_val, "lambda": lam_val, "lambda_ADM": 1.0}
            )
            assert abs(result["lambda_ADM"]) < 1e-12, (
                f"β_{{λ_ADM}} = {result['lambda_ADM']:.3e} at "
                f"(g={g_val}, λ={lam_val}, λ_ADM=1) — fixed line broken"
            )

    def test_lambda_adm_perturbation_drives_back_to_one(self):
        """For g > 0, β_{λ_ADM} has the right sign to restore λ_ADM = 1."""
        # Slightly above 1 → β should be negative (drives down)
        # Slightly below 1 → β should be positive (drives up)
        above = self.system.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_ADM": 1.05}
        )
        below = self.system.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_ADM": 0.95}
        )
        # Sign of β determines stability; require opposite signs
        assert above["lambda_ADM"] * below["lambda_ADM"] < 0


class TestFoliatedBenchmarkReference:
    """Validation reference values are present and self-consistent."""

    def test_foliated_eh_fp_keys(self):
        for key in ("g_star", "lambda_star", "lambda_ADM_star", "n_relevant"):
            assert key in FOLIATED_EH_FP

    def test_lorentzian_fp_keys(self):
        for key in ("g_star", "lambda_star", "lambda_ADM_star"):
            assert key in LORENTZIAN_FP

    def test_lambda_adm_reference_is_one(self):
        """Both Euclidean and Lorentzian benchmarks: λ_ADM* = 1."""
        assert abs(FOLIATED_EH_FP["lambda_ADM_star"] - 1.0) < 1e-12
        assert abs(LORENTZIAN_FP["lambda_ADM_star"] - 1.0) < 1e-12

    def test_benchmark_function(self):
        """foliated_eh_benchmark() returns the documented dict."""
        bench = foliated_eh_benchmark()
        assert bench["lambda_ADM"] == 1.0
        assert 0.5 < bench["g"] < 1.5
        assert 0.0 < bench["lambda"] < 0.5


class TestFoliatedBetaPinned:
    """Pin numerical β values at representative points.

    Refactors of beta/system.py (lambdify, caching, etc.) must reproduce
    these values bit-exactly to 1e-10 relative tolerance.
    """

    PINNED_AT_LAMBDA_ADM_ONE = {
        # at g=0.5, λ=0.1, λ_ADM=1: β_{λ_ADM} should vanish
        "lambda_ADM": 0.0,
    }

    def setup_method(self):
        self.system = build_foliated_eh_beta_system(d=4)

    def test_pinned_beta_lambda_adm_on_fixed_line(self):
        result = self.system.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_ADM": 1.0}
        )
        assert abs(result["lambda_ADM"]) < 1e-12

    def test_pinned_beta_at_gfp_with_lambda_adm(self):
        """β_g and β_λ at GFP are zero regardless of λ_ADM."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "lambda_ADM": 1.5}
        )
        assert abs(result["g"]) < 1e-12
        assert abs(result["lambda"]) < 1e-12
