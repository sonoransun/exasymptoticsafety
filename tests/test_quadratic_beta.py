"""Regression tests for the quadratic gravity beta function system.

Pins the structural and numerical behavior of build_quadratic_beta_system
so that performance refactors of beta/system.py cannot silently drift
the physics. Reference values are drawn from
src/asymsafety/validation/codello_2009.py (Codello, Percacci & Rahmede 2009).

Note: the current truncation implements only the one-loop α/β running with
λ-dependent threshold corrections; β_α and β_β are independent of α and β
themselves, so the polynomial system has no isolated NGFP in (α, β). The
universal one-loop coefficients are recovered up to scheme-dependent
threshold contributions.
"""

from __future__ import annotations

import numpy as np
from sympy import Rational, Symbol, pi

from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.validation.codello_2009 import (
    ONE_LOOP_UNIVERSAL,
    QUADRATIC_FP,
)


class TestQuadraticSystemStructure:
    """The quadratic system has the expected couplings and shape."""

    def setup_method(self):
        self.system = build_quadratic_beta_system(d=4)

    def test_four_couplings(self):
        assert self.system.dimension == 4
        assert self.system.coupling_names == ["g", "lambda", "alpha", "beta"]

    def test_alpha_beta_one_loop_independence(self):
        """β_α and β_β do not depend on α or β at one loop."""
        alpha = Symbol("alpha", real=True)
        beta_ = Symbol("beta", real=True)
        b_alpha = self.system.beta("alpha").expression
        b_beta = self.system.beta("beta").expression
        assert alpha not in b_alpha.free_symbols
        assert beta_ not in b_alpha.free_symbols
        assert alpha not in b_beta.free_symbols
        assert beta_ not in b_beta.free_symbols

    def test_gaussian_fp_has_running_higher_derivative_couplings(self):
        """At g=λ=α=β=0, β_α and β_β are nonzero (one-loop universal)."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        assert abs(result["g"]) < 1e-12
        assert abs(result["lambda"]) < 1e-12
        assert abs(result["alpha"]) > 1e-6
        assert abs(result["beta"]) > 1e-6


class TestQuadraticOneLoopCoefficients:
    """Universal one-loop coefficients for β_α, β_β.

    Codello-Percacci-Rahmede (2009): the C² coupling β is asymptotically
    free with universal one-loop coefficient -196/45 (in 1/16π² units).
    The R² coupling α has a positive one-loop running.
    """

    def setup_method(self):
        self.system = build_quadratic_beta_system(d=4)

    def test_beta_beta_negative_at_lambda_zero(self):
        """C² coupling is asymptotically free: β_β < 0 at λ=0."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        assert result["beta"] < 0

    def test_beta_alpha_positive_at_lambda_zero(self):
        """R² coupling β_α > 0 at λ=0 in this scheme."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        assert result["alpha"] > 0

    def test_beta_beta_universal_order_of_magnitude(self):
        """β_β at λ=0 within order of magnitude of -196/(45·16π²)."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        ref = float(Rational(-196, 45) / (16 * pi**2))
        ratio = result["beta"] / ref
        assert 0.3 < ratio < 3.0, (
            f"β_β = {result['beta']:.4e} not within factor 3 of "
            f"published universal {ref:.4e}"
        )


class TestQuadraticPinnedValues:
    """Pin the exact toolkit values so refactors trigger CI failures.

    These values are what the current implementation produces; literature
    cross-checks live in the assertion docstrings, not in the asserts.
    """

    PINNED_AT_GFP = {
        "alpha": 2.357124758346052e-3,   # at g=λ=α=β=0
        "beta": -2.3149075984950782e-2,  # at g=λ=α=β=0
    }
    PINNED_AT_LAMBDA_NONZERO = {
        # at g=0, λ=0.193, α=β=0 (Reuter-like λ)
        "alpha": None,  # filled below by computation tolerance
    }

    def setup_method(self):
        self.system = build_quadratic_beta_system(d=4)

    def test_pinned_beta_alpha_at_gfp(self):
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        np.testing.assert_allclose(
            result["alpha"], self.PINNED_AT_GFP["alpha"], rtol=1e-10
        )

    def test_pinned_beta_beta_at_gfp(self):
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        np.testing.assert_allclose(
            result["beta"], self.PINNED_AT_GFP["beta"], rtol=1e-10
        )

    def test_validation_dict_keys_present(self):
        """Sanity-check that the QUADRATIC_FP reference dict still has its keys."""
        for key in ("g_star", "lambda_star", "alpha_star",
                     "beta_star", "n_relevant"):
            assert key in QUADRATIC_FP

    def test_one_loop_universal_dict_keys(self):
        for key in ("beta_alpha_1loop", "beta_beta_1loop"):
            assert key in ONE_LOOP_UNIVERSAL
        assert ONE_LOOP_UNIVERSAL["beta_beta_1loop"] < 0  # AF
