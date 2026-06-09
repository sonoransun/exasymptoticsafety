"""Tests for Seeley-DeWitt heat kernel coefficients."""

import pytest
import sympy
from sympy import Rational, Symbol, simplify

from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients


class TestSeeleyDeWitt:
    """Test heat kernel coefficients against known results."""

    def setup_method(self):
        self.R = Symbol("R_bar", positive=True)
        self.sd = SeeleyDeWittCoefficients(d=4, R_bar=self.R)

    def test_b0_scalar(self):
        """b_0 for a scalar: tr(I) = 1."""
        assert self.sd.b0("scalar") == 1

    def test_b0_vector(self):
        """b_0 for a vector: tr(I) = d = 4."""
        assert self.sd.b0("vector") == 4

    def test_b0_TT_tensor(self):
        """b_0 for TT tensor: tr(I) = (d+1)(d-2)/2 = 5."""
        assert self.sd.b0("TT_tensor") == 5

    def test_b0_ghost(self):
        """b_0 for ghost vector: same as vector = 4."""
        assert self.sd.b0("ghost_vector") == 4

    def test_b2_scalar_minimal(self):
        """b_2 for minimally coupled scalar: R̄/6."""
        b2 = self.sd.b2("scalar")
        assert simplify(b2 - self.R / 6) == 0

    def test_b2_vector(self):
        """b_2 for a vector field: includes endomorphism -R̄_μν."""
        b2 = self.sd.b2("vector")
        # b_2 = tr(I) R̄/6 - tr(E) = 4R̄/6 - (-R̄) = 4R̄/6 + R̄ = 5R̄/3
        expected = Rational(4, 6) * self.R + self.R  # = 5R/3
        assert simplify(b2 - expected) == 0

    def test_b4_returns_dict(self):
        """b_4 should return a dictionary with R2, Ric2, Riem2 keys."""
        b4 = self.sd.b4("scalar")
        assert "R2" in b4
        assert "Ric2" in b4
        assert "Riem2" in b4

    def test_b4_scalar_universal(self):
        """b_4 universal part for scalar: 1/180 Riem², -1/180 Ric², 1/72 R²."""
        b4 = self.sd.b4("scalar")
        # Universal part (no endomorphism, no bundle curvature):
        assert b4["R2"] == Rational(1, 72)
        assert b4["Ric2"] == Rational(-1, 180)
        assert b4["Riem2"] == Rational(1, 180)

    def test_b4_scalar_on_sphere(self):
        """Scalar b_4 on S^4: 29 R̄²/2160 (= 29/15 at R̄ = 12).

        Standard minimal-scalar value; verified against the exact mode
        sum over the Rubin-Ordóñez S^4 spectrum.
        """
        b4 = self.sd.b4_on_sphere("scalar")
        assert simplify(b4 - Rational(29, 2160) * self.R**2) == 0

    def test_b4_vector_riem2_coefficient(self):
        """Spin-1 Riem² coefficient: 4/180 - 1/12 = -11/180.

        tr(Ω²) = -R²_μνρσ for the vector bundle (antisymmetry of Ω),
        so the (1/12)tr(Ω²) term *subtracts*; Christensen & Duff (1979),
        Vassilevich (2003).
        """
        b4 = self.sd.b4("vector")
        assert b4["Riem2"] == Rational(-11, 180)

    def test_b4_vector_on_sphere(self):
        """Vector (E = -Ric) b_4 on S^4: 179 R̄²/540 (= 716/15 at R̄ = 12).

        Verified against both the Vassilevich master formula and the
        exact mode sum for -D² - Ric on S^4.
        """
        b4 = self.sd.b4_on_sphere("vector")
        assert simplify(b4 - Rational(179, 540) * self.R**2) == 0

    def test_b2_TT_constrained(self):
        """b_2 for -D² on the constrained TT bundle: -(5/6) R̄.

        Constrained spectral route (Lauscher & Reuter PRD 65, 025013):
        exact mode sum over eigenvalues l(l+3)-2 with degeneracies
        5(2l+3)(l-1)(l+4)/6, l ≥ 2.
        """
        b2 = self.sd.b2("TT_tensor")
        assert simplify(b2 + Rational(5, 6) * self.R) == 0

    def test_b2_TT_lichnerowicz(self):
        """b_2 for the Lichnerowicz operator on TT (tr E = 10R̄/3): -(25/6) R̄."""
        b2 = self.sd.b2("TT_tensor", endomorphism=Rational(10, 3) * self.R)
        assert simplify(b2 + Rational(25, 6) * self.R) == 0

    def test_b4_TT_on_sphere(self):
        """b_4 for -D² on the constrained TT bundle on S^4: -R̄²/432."""
        b4 = self.sd.b4_on_sphere("TT_tensor")
        assert simplify(b4 + self.R**2 / 432) == 0

    def test_b4_TT_lichnerowicz_on_sphere(self):
        """b_4 for the Lichnerowicz operator on TT on S^4: 719 R̄²/432."""
        b4 = self.sd.b4_on_sphere(
            "TT_tensor", endomorphism=Rational(10, 3) * self.R,
        )
        assert simplify(b4 - Rational(719, 432) * self.R**2) == 0

    def test_ghost_vector_matches_vector(self):
        """Ghost vector shares the vector bundle coefficients."""
        assert simplify(
            self.sd.b4_on_sphere("ghost_vector")
            - self.sd.b4_on_sphere("vector")
        ) == 0
