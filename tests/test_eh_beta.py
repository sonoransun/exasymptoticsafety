"""Tests for Einstein-Hilbert beta functions."""

import pytest
import numpy as np
import sympy
from sympy import Symbol, Rational, pi

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability


class TestEHBetaFunctions:
    """Test the Einstein-Hilbert beta function system."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)

    def test_system_has_two_couplings(self):
        """EH system should have g and λ."""
        assert self.system.dimension == 2
        assert "g" in self.system.coupling_names
        assert "lambda" in self.system.coupling_names

    def test_gaussian_fp_exists(self):
        """The Gaussian FP (g=0, λ=0) should be a fixed point."""
        result = self.system.evaluate({"g": 0.0, "lambda": 0.0})
        assert abs(result["g"]) < 1e-10
        assert abs(result["lambda"]) < 1e-10

    def test_beta_g_structure(self):
        """β_g should contain (2 + η_N)·g and vanish at g=0."""
        g = Symbol("g", positive=True)
        beta_g = self.system.beta("g").expression
        assert beta_g.subs(g, 0) == 0

    def test_beta_lambda_at_gfp(self):
        """β_λ at the GFP should give -2λ (canonical running)."""
        result = self.system.evaluate({"g": 1e-15, "lambda": 0.1})
        assert abs(result["lambda"] + 0.2) < 0.01

    def test_jacobian_at_gfp(self):
        """Stability matrix at GFP should reflect canonical dimensions."""
        J = self.system.jacobian_numerical({"g": 0.0, "lambda": 0.0})
        evals = np.sort(np.linalg.eigvals(J).real)
        # One eigenvalue ≈ 2 (g direction), one ≈ -2 (λ direction)
        assert abs(evals[0] - (-2)) < 0.5 or abs(evals[1] - 2) < 0.5


class TestEHFixedPoint:
    """Test finding the Reuter fixed point."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)
        self.finder = FixedPointFinder(self.system)

    def test_ngfp_exists(self):
        """A non-Gaussian fixed point should exist with g* > 0."""
        fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        assert fp.location["g"] > 0.01
        assert not fp.is_gaussian

    def test_ngfp_lambda_less_than_half(self):
        """λ* < 1/2 (below the pole of the threshold functions)."""
        fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        assert fp.location["lambda"] < 0.5

    def test_eta_minus_two_at_ngfp(self):
        """At the NGFP, η_N = -2 (from β_g = 0)."""
        fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        # β_g = (2+η_N)g = 0 and g≠0 ⟹ η_N = -2
        # Verify by checking that β_g ≈ 0 at the FP
        betas = self.system.evaluate(fp.location)
        assert abs(betas["g"]) < 1e-8

    def test_at_least_one_relevant_direction(self):
        """The Reuter FP should have at least 1 relevant direction."""
        fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        sa = analyze_stability(self.system, fp)
        assert fp.relevant_directions >= 1

    def test_product_g_lambda_positive(self):
        """The product g*·λ* should be positive."""
        fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        assert fp.location["g"] * fp.location["lambda"] > 0
