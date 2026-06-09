"""Tests for threshold functions."""

import math

import pytest
import sympy
from sympy import Rational, Symbol, gamma, pi

from asymsafety.frg.threshold import ThresholdFunctions, QFunctional
from asymsafety.frg.regulator import ExponentialRegulator, LitimRegulator


class TestLitimThresholdFunctions:
    """Test threshold functions with the Litim regulator."""

    def setup_method(self):
        self.tf = ThresholdFunctions(LitimRegulator())
        self.w = Symbol("w", real=True)

    def test_Phi_1_1_zero(self):
        """Φ^1_1(0) = 1/Γ(2) = 1."""
        result = self.tf.Phi(1, 1, 0)
        assert sympy.simplify(result - 1) == 0

    def test_Phi_1_2_zero(self):
        """Φ^1_2(0) = 1/Γ(3) = 1/2."""
        result = self.tf.Phi(1, 2, 0)
        assert sympy.simplify(result - Rational(1, 2)) == 0

    def test_Phi_2_1_zero(self):
        """Φ^2_1(0) = 1/Γ(2) = 1."""
        result = self.tf.Phi(2, 1, 0)
        assert sympy.simplify(result - 1) == 0

    def test_Phi_1_1_w(self):
        """Φ^1_1(w) = 1/(1+w)."""
        result = self.tf.Phi(1, 1, self.w)
        expected = 1 / (1 + self.w)
        assert sympy.simplify(result - expected) == 0

    def test_Phi_2_1_w(self):
        """Φ^2_1(w) = 1/(1+w)²."""
        result = self.tf.Phi(2, 1, self.w)
        expected = 1 / (1 + self.w)**2
        assert sympy.simplify(result - expected) == 0

    def test_Phi_1_2_w(self):
        """Φ^1_2(w) = 1/(2(1+w))."""
        result = self.tf.Phi(1, 2, self.w)
        expected = 1 / (2 * (1 + self.w))
        assert sympy.simplify(result - expected) == 0

    def test_Phi_tilde_1_1_zero(self):
        """Φ̃^1_1(0) = 1/Γ(3) = 1/2."""
        result = self.tf.Phi_tilde(1, 1, 0)
        assert sympy.simplify(result - Rational(1, 2)) == 0

    def test_Phi_tilde_1_2_zero(self):
        """Φ̃^1_2(0) = 1/Γ(4) = 1/6."""
        result = self.tf.Phi_tilde(1, 2, 0)
        assert sympy.simplify(result - Rational(1, 6)) == 0

    def test_Phi_adjusted_reduces(self):
        """Φ̂ reduces to Φ when η = 0."""
        result = self.tf.Phi_adjusted(1, 1, self.w, 0)
        expected = self.tf.Phi(1, 1, self.w)
        assert sympy.simplify(result - expected) == 0

    def test_Phi_adjusted_with_eta(self):
        """Φ̂^1_1(w; η) = Φ^1_1(w) - η/4 Φ̃^1_1(w)."""
        eta = Symbol("eta")
        result = self.tf.Phi_adjusted(1, 1, self.w, eta)
        expected = self.tf.Phi(1, 1, self.w) - eta / 4 * self.tf.Phi_tilde(1, 1, self.w)
        assert sympy.simplify(result - expected) == 0


class TestNumericalThreshold:
    """Test numerical evaluation of threshold functions."""

    def test_litim_numerical_Phi(self):
        """Numerical Φ^1_1(0) = 1."""
        tf = ThresholdFunctions(LitimRegulator())
        val = tf.evaluate_numerical("Phi", 1, 1, 0.0)
        assert abs(val - 1.0) < 1e-10

    def test_litim_numerical_Phi_nonzero(self):
        """Numerical Φ^1_1(0.5) = 1/1.5 = 2/3."""
        tf = ThresholdFunctions(LitimRegulator())
        val = tf.evaluate_numerical("Phi", 1, 1, 0.5)
        assert abs(val - 2.0/3.0) < 1e-10

    def test_litim_numerical_Phi_tilde(self):
        """Numerical Φ̃^1_1(0) = 1/2."""
        tf = ThresholdFunctions(LitimRegulator())
        val = tf.evaluate_numerical("Phi_tilde", 1, 1, 0.0)
        assert abs(val - 0.5) < 1e-10


class TestExponentialNumericalThreshold:
    """Numerical threshold functions for the exponential regulator.

    With the Reuter-convention numerator R_k - z R_k' (= ∂_t R_k / 2k²)
    and shape r(y) = 1/(e^y - 1), the w = 0 values reduce to exact Bose
    integrals:
        Φ^1_1(0) = π²/6,   Φ^2_1(0) = 1,   Φ^1_2(0) = 2ζ(3),
        Φ̃^1_1(0) = 1.
    """

    def setup_method(self):
        self.tf = ThresholdFunctions(ExponentialRegulator())

    def test_Phi_1_1_zero_is_pi_sq_over_6(self):
        """Φ^1_1(0) = π²/6 (exact Bose integral ∫ y/(e^y-1) dy)."""
        val = self.tf.evaluate_numerical("Phi", 1, 1, 0.0)
        assert abs(val - math.pi**2 / 6) < 1e-8

    def test_Phi_2_1_zero_is_one(self):
        """Φ^2_1(0) = 1 exactly."""
        val = self.tf.evaluate_numerical("Phi", 2, 1, 0.0)
        assert abs(val - 1.0) < 1e-8

    def test_Phi_1_2_zero_is_two_zeta3(self):
        """Φ^1_2(0) = 2ζ(3) (exact Bose integral ∫ y²/(e^y-1) dy / Γ(2))."""
        val = self.tf.evaluate_numerical("Phi", 1, 2, 0.0)
        assert abs(val - 2 * float(sympy.zeta(3))) < 1e-8

    def test_Phi_tilde_1_1_zero_is_one(self):
        """Φ̃^1_1(0) = ∫ e^{-y} dy = 1 exactly."""
        val = self.tf.evaluate_numerical("Phi_tilde", 1, 1, 0.0)
        assert abs(val - 1.0) < 1e-8

    def test_Phi_1_1_half(self):
        """Φ^1_1(1/2) against high-precision quadrature (mpmath, dps=40)."""
        val = self.tf.evaluate_numerical("Phi", 1, 1, 0.5)
        assert abs(val - 1.2713210370208803) < 1e-8

    def test_Phi_2_1_half(self):
        """Φ^2_1(1/2) against high-precision quadrature (mpmath, dps=40)."""
        val = self.tf.evaluate_numerical("Phi", 2, 1, 0.5)
        assert abs(val - 0.56195881707831066) < 1e-8

    def test_Phi_tilde_1_1_half(self):
        """Φ̃^1_1(1/2) against high-precision quadrature (mpmath, dps=40)."""
        val = self.tf.evaluate_numerical("Phi_tilde", 1, 1, 0.5)
        assert abs(val - 0.74722605965469218) < 1e-8

    def test_Phi_negative_w(self):
        """Φ values must be finite (no exp-overflow NaN) for w < 0 too."""
        val = self.tf.evaluate_numerical("Phi", 1, 1, -0.3)
        assert math.isfinite(val)
        assert abs(val - 2.0259674698) < 1e-8


class TestSymbolicPhiBranch:
    """Symbolic (non-Litim) Phi branch normalization (HV-2b).

    _phi_symbolic must carry the module-convention numerator
    [R_k - z R_k'] = ∂_t R_k / 2 — the same normalization as the Litim
    closed form and evaluate_numerical — and return an unevaluated
    sympy.Integral (not a bare integrand).
    """

    def setup_method(self):
        self.tf = ThresholdFunctions(ExponentialRegulator())

    def test_returns_integral(self):
        """Phi for a non-Litim regulator is an unevaluated Integral."""
        res = self.tf.Phi(1, 1, 0)
        assert isinstance(res, sympy.Expr)
        assert res.has(sympy.Integral)

    def test_normalization_matches_reuter_convention(self):
        """Φ^1_1(0) = π²/6 — 1×, not 2× (old dtR_k numerator gave 2Φ+2Φ̃)."""
        val = float(self.tf.Phi(1, 1, 0).evalf())
        assert abs(val - math.pi**2 / 6) < 1e-8

    def test_symbolic_equals_numerical_branch(self):
        """Symbolic and numerical branches agree on Φ^2_1(1/2)."""
        sym = float(self.tf.Phi(2, 1, sympy.Rational(1, 2)).evalf())
        num = self.tf.evaluate_numerical("Phi", 2, 1, 0.5)
        assert abs(sym - num) < 1e-8


class TestQFunctional:
    """Test Q-functional evaluation."""

    def test_Q_litim_basic(self):
        """Q_{d/2}[∂_t R_k / (P_k + m²)] for d=4, p=1."""
        q = QFunctional(LitimRegulator())
        k = Symbol("k", positive=True)
        m_sq = Symbol("m_sq", positive=True)
        result = q.evaluate(2, 1, m_sq, k)
        expected = 2 * k**6 / (2 * (k**2 + m_sq))
        assert sympy.simplify(result - expected) == 0
