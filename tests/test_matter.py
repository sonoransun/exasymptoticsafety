"""Tests for gravity-matter fixed points with scalar self-interactions."""

from __future__ import annotations

import numpy as np
import pytest
from sympy import Symbol

from asymsafety.actions.matter import MatterContent
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.beta.matter import (
    build_eh_matter_beta_system,
    build_gravity_matter_fp_system,
    scalar_anomalous_dimension,
    scan_gravity_matter_fps,
)
from asymsafety.utils.conventions import conformal_coupling


class TestScalarAnomalousDimension:
    """Test the scalar field anomalous dimension."""

    def setup_method(self):
        self.g = Symbol("g", positive=True)
        self.lam = Symbol("lambda", real=True)
        self.eta = scalar_anomalous_dimension(self.g, self.lam)

    def test_eta_phi_vanishes_at_g_zero(self):
        """eta_phi should vanish when g = 0."""
        assert self.eta.subs(self.g, 0) == 0

    def test_eta_phi_sign(self):
        """eta_phi should be negative for g > 0, lambda < 1/2."""
        val = float(self.eta.subs([(self.g, 0.5), (self.lam, 0.1)]))
        assert val < 0

    def test_eta_phi_diverges_at_lambda_half(self):
        """eta_phi should diverge as lambda -> 1/2 (graviton pole)."""
        val_close = float(
            self.eta.subs([(self.g, 1.0), (self.lam, 0.499)])
        )
        val_far = float(
            self.eta.subs([(self.g, 1.0), (self.lam, 0.0)])
        )
        assert abs(val_close) > 10 * abs(val_far)


class TestBuildGravityMatterFPSystem:
    """Test the system builder for various extension combinations."""

    def test_base_system_matches_eh_matter(self):
        """With all extensions off, should have same couplings as EH+matter."""
        matter = MatterContent(n_scalars=1)
        sys_base = build_gravity_matter_fp_system(matter, scalar_quartic=False)
        sys_eh = build_eh_matter_beta_system(matter)
        assert sys_base.dimension == 2
        assert sys_base.coupling_names == sys_eh.coupling_names

    def test_quartic_adds_lambda_phi(self):
        matter = MatterContent(n_scalars=1)
        sys = build_gravity_matter_fp_system(matter, scalar_quartic=True)
        assert sys.dimension == 3
        assert "lambda_phi" in sys.coupling_names

    def test_yukawa_adds_y(self):
        matter = MatterContent(n_scalars=1, n_dirac=1)
        sys = build_gravity_matter_fp_system(matter, yukawa=True)
        assert "y" in sys.coupling_names

    def test_running_xi_adds_xi(self):
        matter = MatterContent(n_scalars=1)
        sys = build_gravity_matter_fp_system(
            matter, scalar_quartic=True, running_xi=True
        )
        assert "xi" in sys.coupling_names

    def test_all_extensions(self):
        """All three extensions should give 5-coupling system."""
        matter = MatterContent(n_scalars=1, n_dirac=1)
        sys = build_gravity_matter_fp_system(
            matter, scalar_quartic=True, yukawa=True, running_xi=True
        )
        assert sys.dimension == 5
        assert sys.coupling_names == [
            "g", "lambda", "lambda_phi", "y", "xi"
        ]

    def test_yukawa_requires_fermions(self):
        matter = MatterContent(n_scalars=1, n_dirac=0)
        with pytest.raises(ValueError, match="n_dirac"):
            build_gravity_matter_fp_system(matter, yukawa=True)

    def test_running_xi_requires_quartic(self):
        matter = MatterContent(n_scalars=1)
        with pytest.raises(ValueError, match="scalar_quartic"):
            build_gravity_matter_fp_system(matter, running_xi=True)

    def test_gaussian_fp_vanishes(self):
        """All betas should vanish at the Gaussian fixed point."""
        matter = MatterContent(n_scalars=1)
        sys = build_gravity_matter_fp_system(matter, scalar_quartic=True)
        result = sys.evaluate(
            {"g": 0.0, "lambda": 0.0, "lambda_phi": 0.0}
        )
        for name, val in result.items():
            assert abs(val) < 1e-10, f"beta_{name} = {val} at GFP"

    def test_evaluate_returns_all_couplings(self):
        matter = MatterContent(n_scalars=1, n_dirac=1)
        sys = build_gravity_matter_fp_system(
            matter, scalar_quartic=True, yukawa=True
        )
        result = sys.evaluate(
            {"g": 0.5, "lambda": 0.1, "lambda_phi": 0.01, "y": 0.1}
        )
        assert set(result.keys()) == {"g", "lambda", "lambda_phi", "y"}


class TestQuarticFixedPoint:
    """Test the shifted NGFP with non-zero lambda_phi*."""

    def setup_method(self):
        self.matter = MatterContent(n_scalars=1)
        self.system = build_gravity_matter_fp_system(
            self.matter, scalar_quartic=True
        )
        self.finder = FixedPointFinder(self.system)
        self.fp = self.finder.find_fixed_point(
            {"g": 0.65, "lambda": 0.14, "lambda_phi": 0.01}
        )

    def test_ngfp_exists(self):
        assert self.fp is not None
        assert self.fp.location["g"] > 0.01
        assert not self.fp.is_gaussian

    def test_lambda_phi_star_positive(self):
        """Gravity induces a positive quartic coupling at the NGFP."""
        assert self.fp is not None
        assert self.fp.location["lambda_phi"] > 0

    def test_lambda_phi_star_small(self):
        """lambda_phi* should be O(g*^2), i.e., smaller than g*."""
        assert self.fp is not None
        assert self.fp.location["lambda_phi"] < self.fp.location["g"]

    def test_lambda_below_half(self):
        """lambda* should remain below the graviton pole at 1/2."""
        assert self.fp is not None
        assert self.fp.location["lambda"] < 0.5

    def test_three_critical_exponents(self):
        assert self.fp is not None
        assert len(self.fp.critical_exponents) == 3

    def test_betas_vanish_at_fp(self):
        """Beta functions should vanish at the fixed point."""
        assert self.fp is not None
        betas = self.system.evaluate(self.fp.location)
        for name, val in betas.items():
            assert abs(val) < 1e-6, f"beta_{name} = {val} at FP"

    def test_gravity_couplings_near_pure_eh(self):
        """g* should be in the same ballpark as the pure EH NGFP (~0.69)."""
        assert self.fp is not None
        assert 0.1 < self.fp.location["g"] < 2.0


class TestYukawaFixedPoint:
    """Test the NGFP with Yukawa coupling."""

    def setup_method(self):
        self.matter = MatterContent(n_scalars=1, n_dirac=1)
        self.system = build_gravity_matter_fp_system(
            self.matter, scalar_quartic=True, yukawa=True
        )
        self.finder = FixedPointFinder(self.system)
        # The Dirac fermion shifts the gravitational NGFP to ~(0.70, 0.148)
        self.guess = {"g": 0.70, "lambda": 0.148, "lambda_phi": 0.01, "y": 0.0}

    def test_ngfp_exists_with_yukawa(self):
        fp = self.finder.find_fixed_point(self.guess)
        assert fp is not None
        assert not fp.is_gaussian

    def test_y_star_vanishes_or_small(self):
        """y* should be zero or small: gravity cannot generate Yukawa."""
        fp = self.finder.find_fixed_point(self.guess)
        assert fp is not None
        assert abs(fp.location["y"]) < 0.1


class TestRunningXiFixedPoint:
    """Test the NGFP with running non-minimal coupling."""

    def setup_method(self):
        self.matter = MatterContent(n_scalars=1)
        self.system = build_gravity_matter_fp_system(
            self.matter, scalar_quartic=True, running_xi=True
        )
        self.finder = FixedPointFinder(self.system)

    def test_ngfp_exists_with_xi(self):
        fp = self.finder.find_fixed_point(
            {"g": 0.65, "lambda": 0.14, "lambda_phi": 0.01, "xi": 1.0 / 6}
        )
        assert fp is not None
        assert not fp.is_gaussian

    def test_xi_star_near_conformal(self):
        """xi* should be in the vicinity of conformal coupling 1/6."""
        fp = self.finder.find_fixed_point(
            {"g": 0.65, "lambda": 0.14, "lambda_phi": 0.01, "xi": 1.0 / 6}
        )
        assert fp is not None
        xi_conf = float(conformal_coupling(4))
        assert abs(fp.location["xi"] - xi_conf) < 0.5


class TestScanGravityMatterFPs:
    """Test the scan function for the extended system."""

    def test_scan_returns_results(self):
        results = scan_gravity_matter_fps(
            n_scalars_range=range(1, 4),
            scalar_quartic=True,
        )
        assert len(results) == 3

    def test_scan_first_entry_exists(self):
        results = scan_gravity_matter_fps(
            n_scalars_range=range(1, 2),
            scalar_quartic=True,
        )
        assert results[0]["fp_exists"]

    def test_scan_keys(self):
        results = scan_gravity_matter_fps(
            n_scalars_range=range(1, 2),
            scalar_quartic=True,
        )
        assert "n_scalars" in results[0]
        assert "g" in results[0]
        assert "lambda_phi" in results[0]
