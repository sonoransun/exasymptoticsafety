"""Tests for curvature invariants on maximally symmetric backgrounds."""

import pytest
import sympy
from sympy import Rational, Symbol, simplify

from asymsafety.geometry.curvature import (
    MaxSymBackground,
    CurvatureInvariantBasis,
)


class TestMaxSymBackground:
    """Test curvature invariants on S^d."""

    def setup_method(self):
        self.R = Symbol("R_bar", positive=True)

    def test_riemann_squared_d4(self):
        """R²_μνρσ = 2R̄²/(4·3) = R̄²/6 on S⁴."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        assert simplify(bg.riemann_squared - self.R**2 / 6) == 0

    def test_ricci_tensor_squared_d4(self):
        """R²_μν = R̄²/4 on S⁴."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        assert simplify(bg.ricci_tensor_squared - self.R**2 / 4) == 0

    def test_weyl_vanishes(self):
        """C² = 0 on any maximally symmetric space."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        assert bg.weyl_squared == 0

    def test_gauss_bonnet_d4(self):
        """E = R²_μνρσ - 4R²_μν + R² = R̄²/6 - R̄² + R̄² = R̄²/6 on S⁴."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        expected = (Rational(2, 12) - Rational(4, 4) + 1) * self.R**2
        # = (1/6 - 1 + 1) R² = R²/6
        assert simplify(bg.gauss_bonnet - expected) == 0

    def test_weyl_squared_explicit_d4(self):
        """C² = R²_μνρσ - 2R²_μν + (1/3)R² = 0 on S⁴."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        C2 = (bg.riemann_squared
              - 2 * bg.ricci_tensor_squared
              + Rational(1, 3) * bg.ricci_scalar_squared)
        assert simplify(C2) == 0

    def test_sphere_radius_d4(self):
        """a² = d(d-1)/R̄ = 12/R̄ for S⁴."""
        bg = MaxSymBackground(d=4, R_bar=self.R)
        assert simplify(bg.sphere_radius_squared - 12 / self.R) == 0


class TestCurvatureInvariantBasis:
    """Test decomposition into {R², C², E} basis."""

    def test_pure_R2(self):
        """Pure R² decomposes as R² in the {R², C², E} basis."""
        basis = CurvatureInvariantBasis(d=4)
        result = basis.decompose(c_R2=1, c_Ric2=0, c_Riem2=0)
        assert result["R2"] == 1
        assert result["C2"] == 0
        assert result["E"] == 0

    def test_pure_Ric2(self):
        """Pure R²_μν: R²_μν = (1/3)R² + (1/2)C² - (1/2)E."""
        basis = CurvatureInvariantBasis(d=4)
        result = basis.decompose(c_R2=0, c_Ric2=1, c_Riem2=0)
        assert result["R2"] == Rational(1, 3)
        assert result["C2"] == Rational(1, 2)
        assert result["E"] == Rational(-1, 2)

    def test_pure_Riem2(self):
        """Pure R²_μνρσ: R²_μνρσ = (1/3)R² + 2C² - E."""
        basis = CurvatureInvariantBasis(d=4)
        result = basis.decompose(c_R2=0, c_Ric2=0, c_Riem2=1)
        assert result["R2"] == Rational(1, 3)
        assert result["C2"] == 2
        assert result["E"] == -1

    def test_gauss_bonnet_is_topological(self):
        """E = R²_μνρσ - 4R²_μν + R² decomposes as pure E."""
        basis = CurvatureInvariantBasis(d=4)
        # c_R2=1, c_Ric2=-4, c_Riem2=1
        result = basis.decompose(c_R2=1, c_Ric2=-4, c_Riem2=1)
        assert result["R2"] == 0
        assert result["C2"] == 0
        assert result["E"] == 1

    def test_weyl_squared_decomposition(self):
        """C² = R²_μνρσ - 2R²_μν + (1/3)R² decomposes as pure C²."""
        basis = CurvatureInvariantBasis(d=4)
        result = basis.decompose(
            c_R2=Rational(1, 3), c_Ric2=-2, c_Riem2=1
        )
        assert result["R2"] == 0
        assert result["C2"] == 1
        assert result["E"] == 0
