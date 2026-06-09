"""Tests for gravity-matter fixed points with scalar self-interactions."""

from __future__ import annotations

import numpy as np
import pytest
from sympy import Rational, Symbol, pi, simplify

from asymsafety.actions.matter import MatterContent, matter_eta_N_correction
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.beta.matter import (
    build_eh_matter_beta_system,
    build_gravity_matter_fp_system,
    scalar_anomalous_dimension,
    scan_gravity_matter_fps,
    scan_matter_content,
)
from asymsafety.utils.conventions import conformal_coupling


class TestZeroMatterUnification:
    """The zero-matter limit must reduce exactly to pure gravity (HV-12)."""

    def test_eh_matter_zero_matter_identical_to_eh(self):
        """build_eh_matter_beta_system(MatterContent()) ≡ build_eh_beta_system."""
        eh = build_eh_beta_system(d=4)
        ehm = build_eh_matter_beta_system(MatterContent(), d=4)
        for name in ("g", "lambda"):
            diff = simplify(
                eh.beta(name).expression - ehm.beta(name).expression
            )
            assert diff == 0, f"beta_{name} differs in zero-matter limit"

    def test_gravity_matter_zero_matter_identical_to_eh(self):
        """build_gravity_matter_fp_system at zero matter ≡ build_eh_beta_system."""
        eh = build_eh_beta_system(d=4)
        gm = build_gravity_matter_fp_system(MatterContent(), d=4)
        for name in ("g", "lambda"):
            diff = simplify(
                eh.beta(name).expression - gm.beta(name).expression
            )
            assert diff == 0, f"beta_{name} differs in zero-matter limit"

    def test_zero_matter_ngfp_exists(self):
        """scan_matter_content must find the Reuter FP at N_s = 0."""
        results = scan_matter_content(range(0, 1))
        assert results[0]["fp_exists"]
        assert results[0]["g_star"] == pytest.approx(
            0.7073208809868445, rel=1e-8
        )
        assert results[0]["lambda_star"] == pytest.approx(
            0.19320050715078566, rel=1e-8
        )


class TestDEPMatterWeights:
    """Per-field η_N weights from Dona-Eichhorn-Percacci [1311.2898]."""

    def test_scalar_weight(self):
        """A_scalar = +1/(6π) per field (DEP Eq. (35))."""
        A, B = matter_eta_N_correction(MatterContent(n_scalars=1), 4)
        assert simplify(A - Rational(1, 6) / pi) == 0
        assert B == 0

    def test_dirac_weight(self):
        """A_dirac = +1/(3π) per field (DEP Eq. (38))."""
        A, B = matter_eta_N_correction(MatterContent(n_dirac=1), 4)
        assert simplify(A - Rational(1, 3) / pi) == 0
        assert B == 0

    def test_vector_weight(self):
        """A_vector = -2/(3π) per field in d=4 (DEP Eq. (38))."""
        A, B = matter_eta_N_correction(MatterContent(n_vectors=1), 4)
        assert simplify(A + Rational(2, 3) / pi) == 0
        assert B == 0


def _fp_for(matter: MatterContent, guess: dict) -> dict | None:
    system = build_eh_matter_beta_system(matter, 4)
    fp = FixedPointFinder(system).find_fixed_point(guess)
    if fp is None or fp.location.get("g", 0) <= 1e-6:
        return None
    return fp.location


class TestMatterTrends:
    """Full-system NGFP trends vs DEP [1311.2898] Sec. IV."""

    def test_scalars_push_lambda_up(self):
        """Scalars push λ* towards larger values (towards the λ=1/2
        pole - the DEP destabilization mechanism)."""
        guess = {"g": 0.7, "lambda": 0.19}
        lams = []
        for n in range(0, 5):
            loc = _fp_for(MatterContent(n_scalars=n), guess)
            assert loc is not None
            lams.append(loc["lambda"])
            guess = loc
        assert all(b > a for a, b in zip(lams, lams[1:]))

    def test_dirac_increases_g_decreases_lambda(self):
        """Fermions shift the FP towards larger g* and smaller λ*
        (DEP: 'considerable shift towards larger G and more
        negative Λ')."""
        loc0 = _fp_for(MatterContent(), {"g": 0.7, "lambda": 0.19})
        loc1 = _fp_for(MatterContent(n_dirac=1), {"g": 1.0, "lambda": 0.13})
        assert loc0 is not None and loc1 is not None
        assert loc1["g"] > loc0["g"]
        assert loc1["lambda"] < loc0["lambda"]

    def test_vectors_decrease_g(self):
        """Vectors decrease g* (DEP: 'the effect of vector degrees of
        freedom is always to decrease G*'); no bound on N_v."""
        guess = {"g": 0.7, "lambda": 0.19}
        gs = []
        for n in range(0, 4):
            loc = _fp_for(MatterContent(n_vectors=n), guess)
            assert loc is not None
            gs.append(loc["g"])
            guess = loc
        assert all(b < a for a, b in zip(gs, gs[1:]))


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
        # One scalar shifts the gravitational NGFP to ~(0.660, 0.206)
        self.fp = self.finder.find_fixed_point(
            {"g": 0.66, "lambda": 0.21, "lambda_phi": 0.01}
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
        """g* should be in the same ballpark as the pure EH NGFP (~0.707)."""
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
        # The Dirac fermion shifts the gravitational NGFP to ~(0.93, 0.158)
        self.guess = {"g": 0.93, "lambda": 0.158, "lambda_phi": 0.015, "y": 0.0}

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
            {"g": 0.66, "lambda": 0.21, "lambda_phi": 0.01, "xi": 1.0 / 6}
        )
        assert fp is not None
        assert not fp.is_gaussian

    def test_xi_star_near_conformal(self):
        """xi* should be in the vicinity of conformal coupling 1/6."""
        fp = self.finder.find_fixed_point(
            {"g": 0.66, "lambda": 0.21, "lambda_phi": 0.01, "xi": 1.0 / 6}
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
