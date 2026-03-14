"""Tests for threshold functions."""

import pytest
import sympy
from sympy import Rational, Symbol, gamma, pi

from asymsafety.frg.threshold import ThresholdFunctions, QFunctional
from asymsafety.frg.regulator import LitimRegulator


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
