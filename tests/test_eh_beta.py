"""Tests for Einstein-Hilbert beta functions."""

import pytest
import numpy as np
import sympy
from sympy import Symbol, Rational, pi, simplify

from asymsafety.beta.einstein_hilbert import (
    build_eh_beta_system,
    eh_fixed_point_litim_d4,
)
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability


# Toolkit NGFP for EH + Litim (Type Ia, de Donder) in d=4, consistent
# with Litim PRL 92, 201301 [hep-th/0312114] and CPR [0805.2909].
G_STAR = 0.7073208809868445
LAMBDA_STAR = 0.19320050715078566
THETA_REAL = 1.475302425763855
THETA_IMAG = 3.043205846411925


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

    def test_beta_lambda_g0_identity(self):
        """β_λ(g=0, λ) = -2λ identically (all traces ∝ g)."""
        g = Symbol("g", positive=True)
        lam = Symbol("lambda", real=True)
        beta_lam = self.system.beta("lambda").expression
        assert simplify(beta_lam.subs(g, 0) + 2 * lam) == 0

    def test_jacobian_at_gfp(self):
        """Stability matrix at GFP should reflect canonical dimensions."""
        J = self.system.jacobian_numerical({"g": 0.0, "lambda": 0.0})
        evals = np.sort(np.linalg.eigvals(J).real)
        assert abs(evals[0] - (-2)) < 1e-8
        assert abs(evals[1] - 2) < 1e-8

    def test_gfp_vacuum_energy_slope(self):
        """∂β_λ/∂g at the GFP is +1/(2π): graviton loops generate
        positive vacuum-energy flow (10-mode graviton trace vs -8
        ghost modes; Reuter 1998 [hep-th/9605030])."""
        g = Symbol("g", positive=True)
        lam = Symbol("lambda", real=True)
        beta_lam = self.system.beta("lambda").expression
        slope = simplify(
            sympy.diff(beta_lam, g).subs({g: 0, lam: 0})
        )
        assert simplify(slope - 1 / (2 * pi)) == 0


class TestEHDimensionDependence:
    """Test the d-dependence of the EH builder (Reuter d-dim forms)."""

    def test_d3_differs_from_d4(self):
        """d=3 must not silently return the d=4 expressions."""
        g = Symbol("g", positive=True)
        sys3 = build_eh_beta_system(d=3)
        sys4 = build_eh_beta_system(d=4)
        diff = simplify(
            sys3.beta("g").expression - sys4.beta("g").expression
        )
        assert diff != 0

    def test_canonical_dimension_term(self):
        """β_g/g → (d-2) as g → 0 in any dimension."""
        g = Symbol("g", positive=True)
        for d in (3, 4, 5):
            sys_d = build_eh_beta_system(d=d)
            beta_g = sys_d.beta("g").expression
            assert sympy.limit(beta_g / g, g, 0) == d - 2

    def test_d_below_three_raises(self):
        """d < 3 is outside the validity of the closed forms."""
        with pytest.raises(NotImplementedError):
            build_eh_beta_system(d=2)

    def test_non_harmonic_gauge_raises(self):
        """Only the de Donder (harmonic) gauge is implemented."""
        with pytest.raises(NotImplementedError):
            build_eh_beta_system(d=4, gauge="landau")


class TestEHFixedPoint:
    """Test finding the Reuter fixed point."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)
        self.finder = FixedPointFinder(self.system)
        self.fp = self.finder.find_fixed_point({"g": 0.7, "lambda": 0.19})

    def test_ngfp_exists(self):
        """A non-Gaussian fixed point should exist with g* > 0."""
        assert self.fp is not None
        assert self.fp.location["g"] > 0.01
        assert not self.fp.is_gaussian

    def test_ngfp_location(self):
        """NGFP at g* ≈ 0.707, λ* ≈ 0.193 (Litim PRL 92, 201301;
        CPR 0805.2909)."""
        assert self.fp is not None
        assert self.fp.location["g"] == pytest.approx(G_STAR, rel=1e-8)
        assert self.fp.location["lambda"] == pytest.approx(
            LAMBDA_STAR, rel=1e-8
        )

    def test_ngfp_lambda_less_than_half(self):
        """λ* < 1/2 (below the pole of the threshold functions)."""
        assert self.fp is not None
        assert self.fp.location["lambda"] < 0.5

    def test_eta_minus_two_at_ngfp(self):
        """At the NGFP, η_N = -2 (from β_g = 0)."""
        assert self.fp is not None
        # β_g = (2+η_N)g = 0 and g≠0 ⟹ η_N = -2
        # Verify by checking that β_g ≈ 0 at the FP
        betas = self.system.evaluate(self.fp.location)
        assert abs(betas["g"]) < 1e-8

    def test_complex_critical_exponents(self):
        """θ = 1.475 ± 3.043i, the scheme-robust complex pair
        (Litim PRL 92, 201301 [hep-th/0312114])."""
        assert self.fp is not None
        analyze_stability(self.system, self.fp)
        thetas = np.sort_complex(np.asarray(
            self.fp.critical_exponents, dtype=complex
        ))
        assert thetas[0].real == pytest.approx(THETA_REAL, rel=1e-8)
        assert abs(thetas[0].imag) == pytest.approx(THETA_IMAG, rel=1e-8)
        assert thetas[1] == pytest.approx(np.conj(thetas[0]), rel=1e-8)

    def test_two_relevant_directions(self):
        """The Reuter FP has exactly 2 relevant directions
        (Re θ > 0 for the complex-conjugate pair)."""
        assert self.fp is not None
        sa = analyze_stability(self.system, self.fp)
        assert self.fp.relevant_directions == 2

    def test_product_g_lambda(self):
        """The (approximately) universal product g*·λ* ≈ 0.1367."""
        assert self.fp is not None
        product = self.fp.location["g"] * self.fp.location["lambda"]
        assert product == pytest.approx(G_STAR * LAMBDA_STAR, rel=1e-8)


class TestEHFixedPointDict:
    """eh_fixed_point_litim_d4 must report the toolkit's own NGFP."""

    def test_dict_matches_computed_fp(self):
        ref = eh_fixed_point_litim_d4()
        assert ref["g"] == pytest.approx(G_STAR, rel=1e-12)
        assert ref["lambda"] == pytest.approx(LAMBDA_STAR, rel=1e-12)
        assert ref["theta_real"] == pytest.approx(THETA_REAL, rel=1e-12)
        assert ref["theta_imag"] == pytest.approx(THETA_IMAG, rel=1e-12)
