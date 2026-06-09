"""Regression tests for the foliated EH beta function system.

Pins structural and numerical behavior of build_foliated_eh_beta_system
so that performance refactors of beta/system.py cannot silently drift
the physics.

Reference values:
    Manrique-Rechenberger-Saueressig 2011, PRL 106, 251302 [1102.5012],
    Eq. (10) (validation/manrique_2011.py):
        Euclidean g* ≈ 0.19, λ* ≈ 0.31, θ = 1.07 ± 3.31 i;
        λ_ADM = 1 imposed by the Diff-invariant ansatz (not a running
        coupling in MRS).

Note: the implemented foliated system is schematic. λ_ADM = 1 is a
fixed plane of β_{λ_ADM} by construction, with eigenvalue
∂β_{λ_ADM}/∂λ_ADM = gλ/(π(1-2λ)) > 0 at physical couplings — the plane
is UV-repulsive (IR-attractive). The (g, λ) sector admits no NGFP at
physical λ (η_N = -2 is unreachable for λ > -1/2), so the MRS NGFP is
not a root of this system. Tests therefore pin structural invariants
and the corrected plane orientation rather than asserting NGFP
existence.
"""

from __future__ import annotations

import numpy as np
import pytest
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
    """λ_ADM = 1 is a fixed plane: β_{λ_ADM} vanishes there for any (g, λ).

    The plane is built into the schematic β_{λ_ADM} ∝ g(λ_ADM - 1); in
    Manrique-Rechenberger-Saueressig (2011) λ_ADM is not a running
    coupling at all — λ_ADM = 1 is imposed by their Diff-invariant
    ansatz. At physical couplings the plane is UV-repulsive (see the
    sign-aware tests below).
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

    def test_lambda_adm_plane_is_uv_repulsive_at_physical_lambda(self):
        """β_{λ_ADM} signs: λ_ADM = 1 is UV-repulsive (IR-attractive).

        ∂β_{λ_ADM}/∂λ_ADM = gλ/(π(1-2λ)) > 0 for g > 0, 0 < λ < 1/2,
        so perturbations grow toward the UV (∂_t δ = +|·|δ): β > 0
        above the plane and β < 0 below it. The previous assertion only
        required opposite signs and was blind to this orientation
        (HV-15c-d); the comments even demanded the opposite
        ("restoring") signs, which the system does not produce.
        """
        above = self.system.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_ADM": 1.05}
        )
        below = self.system.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_ADM": 0.95}
        )
        assert above["lambda_ADM"] > 0, (
            f"β_{{λ_ADM}} = {above['lambda_ADM']:.3e} above the plane — "
            "expected > 0 (UV-repulsive)"
        )
        assert below["lambda_ADM"] < 0, (
            f"β_{{λ_ADM}} = {below['lambda_ADM']:.3e} below the plane — "
            "expected < 0 (UV-repulsive)"
        )

    def test_lambda_adm_eigenvalue_is_g_lambda_over_pi_one_minus_2lambda(self):
        """∂β_{λ_ADM}/∂λ_ADM at (g, λ, 1) equals gλ/(π(1-2λ)) exactly."""
        g_val, lam_val = 0.5, 0.1
        M = self.system.jacobian_numerical(
            {"g": g_val, "lambda": lam_val, "lambda_ADM": 1.0}
        )
        expected = g_val * lam_val / (np.pi * (1 - 2 * lam_val))
        assert M[2, 2] == pytest.approx(expected, rel=1e-10)
        assert M[2, 2] > 0  # UV-repulsive plane


class TestFoliatedBenchmarkReference:
    """Validation reference values match MRS PRL 106, 251302, Eq. (10)."""

    def test_foliated_eh_fp_keys(self):
        for key in ("g_star", "lambda_star", "lambda_ADM_star", "n_relevant"):
            assert key in FOLIATED_EH_FP

    def test_lorentzian_fp_keys(self):
        for key in ("g_star", "lambda_star", "lambda_ADM_star"):
            assert key in LORENTZIAN_FP

    def test_euclidean_fp_is_mrs_eq10(self):
        """Euclidean MRS Eq. (10): g* = 0.19, λ* = 0.31, θ = 1.07 ± 3.31i."""
        assert FOLIATED_EH_FP["g_star"] == pytest.approx(0.19)
        assert FOLIATED_EH_FP["lambda_star"] == pytest.approx(0.31)
        assert FOLIATED_EH_FP["theta_real"] == pytest.approx(1.07)
        assert FOLIATED_EH_FP["theta_imag"] == pytest.approx(3.31)

    def test_lorentzian_fp_is_mrs_eq10(self):
        """Lorentzian MRS Eq. (10): g* = 0.21, λ* = 0.30."""
        assert LORENTZIAN_FP["g_star"] == pytest.approx(0.21)
        assert LORENTZIAN_FP["lambda_star"] == pytest.approx(0.30)

    def test_lambda_adm_reference_is_one(self):
        """λ_ADM = 1 in both dicts (imposed by the MRS ansatz, not run)."""
        assert abs(FOLIATED_EH_FP["lambda_ADM_star"] - 1.0) < 1e-12
        assert abs(LORENTZIAN_FP["lambda_ADM_star"] - 1.0) < 1e-12

    def test_benchmark_function(self):
        """foliated_eh_benchmark() returns the MRS Eq. (10) values."""
        bench = foliated_eh_benchmark()
        assert bench["lambda_ADM"] == 1.0
        assert bench["g"] == pytest.approx(FOLIATED_EH_FP["g_star"])
        assert bench["lambda"] == pytest.approx(FOLIATED_EH_FP["lambda_star"])

    def test_benchmark_is_not_a_root_of_the_schematic_system(self):
        """The MRS NGFP is a literature value, not a root of this system.

        β_g = (2 + η_N)g = 0 with g > 0 needs η_N = -2, unreachable for
        λ > -1/2 with the schematic coefficients — pin that the residual
        at the literature point is O(1), so nobody re-advertises it as a
        toolkit fixed point (HV-15c-b).
        """
        system = build_foliated_eh_beta_system(d=4)
        res = system.evaluate(
            {"g": FOLIATED_EH_FP["g_star"],
             "lambda": FOLIATED_EH_FP["lambda_star"],
             "lambda_ADM": 1.0}
        )
        assert abs(res["g"]) > 0.1
        assert abs(res["lambda_ADM"]) < 1e-12  # still on the fixed plane


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
