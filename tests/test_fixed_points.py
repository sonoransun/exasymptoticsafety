"""Tests for fixed point finding and stability analysis."""

import pytest
import numpy as np
import sympy
from sympy import Symbol

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.analysis.fixed_points import FixedPoint, FixedPointFinder
from asymsafety.analysis.stability import analyze_stability


def make_simple_system():
    """Create a simple 2D system with known fixed points.

    β_x = x(1 - x)
    β_y = y(1 - y)

    Fixed points: (0,0), (0,1), (1,0), (1,1)
    """
    x = Symbol("x", real=True)
    y = Symbol("y", real=True)

    system = BetaFunctionSystem()
    system.add(BetaFunction("x", x, x * (1 - x)))
    system.add(BetaFunction("y", y, y * (1 - y)))
    return system


class TestFixedPointFinder:
    """Test the fixed point finding algorithm."""

    def setup_method(self):
        self.system = make_simple_system()
        self.finder = FixedPointFinder(self.system)

    def test_gaussian_fp(self):
        """Gaussian FP at (0, 0)."""
        gfp = self.finder.gaussian_fixed_point()
        assert abs(gfp.location["x"]) < 1e-10
        assert abs(gfp.location["y"]) < 1e-10

    def test_find_known_fp(self):
        """Find the FP at (1, 1) from initial guess."""
        fp = self.finder.find_fixed_point({"x": 0.9, "y": 0.9})
        assert fp is not None
        assert abs(fp.location["x"] - 1.0) < 1e-6
        assert abs(fp.location["y"] - 1.0) < 1e-6

    def test_find_all_fps(self):
        """Should find all 4 fixed points."""
        fps = self.finder.find_all_fixed_points(
            bounds={"x": (-0.5, 1.5), "y": (-0.5, 1.5)},
            n_grid=5,
            n_random=20,
        )
        # Should find at least 3 (GFP + 2 non-trivial)
        assert len(fps) >= 3

    def test_stability_at_origin(self):
        """At (0,0): eigenvalues of [[1,0],[0,1]] = {1, 1}."""
        gfp = self.finder.gaussian_fixed_point()
        sa = analyze_stability(self.system, gfp)
        # Jacobian at (0,0): ∂β_x/∂x = 1-2x = 1, etc.
        assert abs(sa.eigenvalues[0] - 1.0) < 1e-6
        assert abs(sa.eigenvalues[1] - 1.0) < 1e-6

    def test_stability_at_one_one(self):
        """At (1,1): eigenvalues of [[-1,0],[0,-1]] = {-1, -1}."""
        fp = self.finder.find_fixed_point({"x": 0.9, "y": 0.9})
        assert fp is not None
        sa = analyze_stability(self.system, fp)
        # Both eigenvalues should be -1
        assert all(abs(e + 1.0) < 1e-6 for e in sa.eigenvalues.real)


class TestFixedPointClassification:
    """Test fixed point classification."""

    def test_relevant_directions(self):
        """Test counting of relevant directions."""
        fp = FixedPoint(
            location={"x": 0, "y": 0},
            critical_exponents=np.array([2.0, -1.0]),
        )
        assert fp.relevant_directions == 1
        assert fp.irrelevant_directions == 1

    def test_uv_attractive(self):
        """Test UV-attractive check."""
        fp = FixedPoint(
            location={"x": 0, "y": 0},
            critical_exponents=np.array([2.0, 1.0]),
        )
        assert fp.is_uv_attractive

    def test_not_uv_attractive(self):
        fp = FixedPoint(
            location={"x": 0, "y": 0},
            critical_exponents=np.array([2.0, -1.0]),
        )
        assert not fp.is_uv_attractive

    def test_gaussian_check(self):
        fp = FixedPoint(location={"x": 0, "y": 0})
        assert fp.is_gaussian
        fp2 = FixedPoint(location={"x": 1.0, "y": 0})
        assert not fp2.is_gaussian
