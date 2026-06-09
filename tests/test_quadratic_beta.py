"""Regression tests for the quadratic gravity beta function system.

Pins the structural and numerical behavior of build_quadratic_beta_system
so that performance refactors of beta/system.py cannot silently drift
the physics. Reference values are drawn from
src/asymsafety/validation/codello_2009.py (one-loop universals:
Fradkin-Tseytlin 1982; Avramidi-Barvinsky 1985; Codello-Percacci
hep-th/0607128).

Note: at one loop the four-derivative coefficient runnings are exact,
scheme- and gauge-independent constants in the coefficient basis
(alpha*R^2 + beta*C^2):

    beta_alpha = +(1/16 pi^2) * 5/36   (R^2 coefficient, at omega=0)
    beta_beta  = +(1/16 pi^2) * 133/20 (C^2 coefficient; positive = AF)

Because both are nonzero constants, the (g, lambda, alpha, beta) system
has NO interior NGFP; the literature QUADRATIC_FP values (Codello-
Percacci-Rahmede 2009, full nonperturbative calculation) are external
reference values only.
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

    def test_alpha_beta_one_loop_constants(self):
        """β_α and β_β are coupling-independent constants at one loop.

        The one-loop runnings of the four-derivative coefficients are
        scheme- and gauge-independent pure numbers — they depend on no
        coupling at all (in particular not on λ via spurious threshold
        functions, and not on α, β).
        """
        b_alpha = self.system.beta("alpha").expression
        b_beta = self.system.beta("beta").expression
        assert b_alpha.free_symbols == set()
        assert b_beta.free_symbols == set()

    def test_no_interior_fixed_point(self):
        """The one-loop system admits NO complete fixed point.

        β_α and β_β are nonzero constants, so no point in
        (g, λ, α, β) zeroes all four betas; and the α/β columns of the
        Jacobian vanish identically (the literature n_relevant=4 is not
        realizable in this truncation).
        """
        b_alpha = self.system.beta("alpha").expression
        b_beta = self.system.beta("beta").expression
        assert b_alpha != 0
        assert b_beta != 0
        jac = self.system.jacobian_symbolic()
        names = self.system.coupling_names
        a_col = names.index("alpha")
        b_col = names.index("beta")
        for row in range(len(names)):
            assert jac[row, a_col] == 0
            assert jac[row, b_col] == 0

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

    Coefficient basis (action ⊃ α R² + β C²):
        β_α = +(1/16π²)·5/36 and β_β = +(1/16π²)·133/20, exact
    (Fradkin-Tseytlin 1982; Avramidi-Barvinsky 1985; Codello-Percacci
    hep-th/0607128). β_β > 0 means the C² coefficient grows toward the
    UV — asymptotic freedom (the inverse coupling 1/(2β) → 0⁺).
    """

    def setup_method(self):
        self.system = build_quadratic_beta_system(d=4)

    def test_beta_beta_positive_at_lambda_zero(self):
        """C² coefficient is asymptotically free: β_β > 0 (coefficient basis)."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        assert result["beta"] > 0

    def test_beta_alpha_positive_at_lambda_zero(self):
        """R² coefficient running β_α = +5/(36·16π²) > 0."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        assert result["alpha"] > 0

    def test_beta_alpha_exact_universal(self):
        """16π²·β_α equals the exact rational 5/36, symbolically."""
        b_alpha = self.system.beta("alpha").expression
        assert (b_alpha * 16 * pi**2 - Rational(5, 36)).simplify() == 0

    def test_beta_beta_exact_universal(self):
        """16π²·β_β equals the exact rational 133/20, symbolically."""
        b_beta = self.system.beta("beta").expression
        assert (b_beta * 16 * pi**2 - Rational(133, 20)).simplify() == 0

    def test_universals_match_validation_dict(self):
        """The codello_2009 ONE_LOOP_UNIVERSAL dict matches the system."""
        result = self.system.evaluate(
            {"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0}
        )
        ref_alpha = ONE_LOOP_UNIVERSAL["beta_alpha_1loop"] / float(16 * pi**2)
        ref_beta = ONE_LOOP_UNIVERSAL["beta_beta_1loop"] / float(16 * pi**2)
        np.testing.assert_allclose(result["alpha"], ref_alpha, rtol=1e-12)
        np.testing.assert_allclose(result["beta"], ref_beta, rtol=1e-12)


class TestQuadraticPinnedValues:
    """Pin the exact toolkit values so refactors trigger CI failures.

    These are the exact one-loop universals: 5/(36·16π²) and
    133/(20·16π²) — coefficient basis, see module docstring.
    """

    PINNED_AT_GFP = {
        "alpha": 8.795241635619598e-4,  # 5/(36·16π²), at any point
        "beta": 4.211161695134664e-2,   # 133/(20·16π²), at any point
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
        """QUADRATIC_FP keeps its keys (literature reference values only —
        not a fixed point of this one-loop truncation)."""
        for key in ("g_star", "lambda_star", "alpha_star",
                     "beta_star", "n_relevant"):
            assert key in QUADRATIC_FP

    def test_one_loop_universal_dict_keys(self):
        for key in ("beta_alpha_1loop", "beta_beta_1loop"):
            assert key in ONE_LOOP_UNIVERSAL
        # Exact coefficient-basis universals
        np.testing.assert_allclose(
            ONE_LOOP_UNIVERSAL["beta_alpha_1loop"], 5 / 36, rtol=1e-15
        )
        np.testing.assert_allclose(
            ONE_LOOP_UNIVERSAL["beta_beta_1loop"], 133 / 20, rtol=1e-15
        )
        # Positive = asymptotic freedom in the coefficient basis
        assert ONE_LOOP_UNIVERSAL["beta_beta_1loop"] > 0
