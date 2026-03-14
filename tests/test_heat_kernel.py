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
