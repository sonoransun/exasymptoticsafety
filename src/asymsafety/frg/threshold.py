"""Threshold functions for the Wetterich equation.

Threshold functions encode the regulator dependence of beta functions.
They appear universally in the FRG flow equations for gravity.

Standard definitions:
    Φ^p_n(w) = (1/Γ(n)) ∫_0^∞ dz z^{n-1} [R_k - z R_k'] / (z + R_k + k^2 w)^p
    Φ̃^p_n(w) = (1/Γ(n)) ∫_0^∞ dz z^{n-1} R_k / (z + R_k + k^2 w)^p

For the Litim regulator (closed-form results):
    Φ^p_n(w) = 1 / [Γ(n+1) (1+w)^p]
    Φ̃^p_n(w) = 1 / [Γ(n+2) (1+w)^p]
"""

from __future__ import annotations

from functools import lru_cache

import sympy
from sympy import Expr, Function, Rational, Symbol, gamma, oo, integrate

from asymsafety.frg.regulator import ExponentialRegulator, LitimRegulator, Regulator


class ThresholdFunctions:
    """Compute threshold functions Φ^p_n(w) and Φ̃^p_n(w).

    For the Litim regulator, returns exact closed-form expressions.
    For other regulators, returns symbolic integral expressions or
    provides numerical evaluation.
    """

    def __init__(self, regulator: Regulator | None = None):
        self.regulator = regulator or LitimRegulator()

    def Phi(self, p: int, n: int | Rational,
            w: Expr) -> Expr:
        """Threshold function Φ^p_n(w).

        Args:
            p: Power of the denominator.
            n: Order (related to dimension via Q-functional).
            w: Dimensionless mass argument (e.g., w = -2λ for graviton).

        Returns:
            Symbolic expression for Φ^p_n(w).
        """
        if isinstance(self.regulator, LitimRegulator):
            return self._phi_litim(p, n, w)
        return self._phi_symbolic(p, n, w)

    def Phi_tilde(self, p: int, n: int | Rational,
                  w: Expr) -> Expr:
        """Threshold function Φ̃^p_n(w).

        These appear in η_N corrections to the flow:
        the modified threshold function with one extra power of z
        in the numerator from R_k (without the ∂_t piece).
        """
        if isinstance(self.regulator, LitimRegulator):
            return self._phi_tilde_litim(p, n, w)
        return self._phi_tilde_symbolic(p, n, w)

    def Phi_adjusted(self, p: int, n: int | Rational,
                     w: Expr, eta: Expr) -> Expr:
        """Threshold function with anomalous dimension correction.

        Φ̂^p_n(w; η) = Φ^p_n(w) - η/(2(n+1)) Φ̃^p_n(w)

        This combination arises when the regulator includes the
        wave-function renormalization: R_k ∝ Z_k (k^2 - z)θ(k^2-z).
        """
        return self.Phi(p, n, w) - eta / (2 * (n + 1)) * self.Phi_tilde(p, n, w)

    # --- Litim regulator (closed form) ---

    def _phi_litim(self, p: int, n: int | Rational, w: Expr) -> Expr:
        """Φ^p_n(w) = 1 / [Γ(n+1) (1+w)^p] for the Litim regulator."""
        return 1 / (gamma(n + 1) * (1 + w)**p)

    def _phi_tilde_litim(self, p: int, n: int | Rational, w: Expr) -> Expr:
        """Φ̃^p_n(w) = 1 / [Γ(n+2) (1+w)^p] for the Litim regulator."""
        return 1 / (gamma(n + 2) * (1 + w)**p)

    # --- General regulator (symbolic integrals) ---

    def _phi_symbolic(self, p: int, n: int | Rational, w: Expr) -> Expr:
        """Symbolic integral form Φ^p_n(w) for a general regulator.

        Uses the module-docstring numerator [R_k - z R_k'] = ∂_t R_k / 2
        (Reuter 1998, hep-th/9605030) — *not* the full ∂_t R_k, whose
        factor 2 belongs to :class:`QFunctional` — so this branch
        carries the same normalization as the Litim closed form
        ``_phi_litim`` and as ``evaluate_numerical`` (e.g. it reproduces
        1/[Γ(n+1)(1+w)^p] for the Litim shape and Φ^1_1(0) = π²/6 for
        the exponential shape). Evaluated at k = 1: the threshold
        functions are k-independent in this dimensionless form.

        Returns:
            Unevaluated :class:`sympy.Integral` over z ∈ (0, ∞),
            divided by Γ(n).
        """
        z = Symbol("z_int", positive=True)
        k = sympy.Integer(1)  # dimensionless form: k-dependence drops out
        R = self.regulator.R_k(z, k)
        numerator = R - z * sympy.diff(R, z)
        integrand = z**(n - 1) * numerator / (z + R + w)**p
        return sympy.Integral(integrand, (z, 0, oo)) / gamma(n)

    def _phi_tilde_symbolic(self, p: int, n: int | Rational, w: Expr) -> Expr:
        """Symbolic integral form for Φ̃."""
        z = Symbol("z_int", positive=True)
        k = Symbol("k_int", positive=True)
        R = self.regulator.R_k(z, k)
        integrand = z**(n - 1) * R / (z + R + k**2 * w)**p
        return integrand / gamma(n)

    def evaluate_numerical(self, func_type: str, p: int, n: int,
                           w_val: float) -> float:
        """Numerical evaluation of threshold functions.

        Args:
            func_type: "Phi" or "Phi_tilde".
            p: Denominator power.
            n: Order.
            w_val: Numerical value of the mass argument.

        Returns:
            Numerical value of the threshold function.
        """
        if isinstance(self.regulator, LitimRegulator):
            from math import gamma as mgamma
            if func_type == "Phi":
                return 1.0 / (mgamma(n + 1) * (1 + w_val)**p)
            else:
                return 1.0 / (mgamma(n + 2) * (1 + w_val)**p)

        # For other regulators, use numerical quadrature
        from scipy.integrate import quad
        from math import gamma as mgamma
        import numpy as np

        k_val = 1.0  # Set k=1, results are k-independent in dimensionless form

        if func_type == "Phi":
            def integrand(z):
                # Numerator per the module convention (Reuter 1998,
                # hep-th/9605030): R_k - z R_k' = -z y r'(y), which for
                # the exponential shape r(y) = 1/(e^y - 1) equals
                # k^2 y^2 e^y / (e^y - 1)^2.  This reproduces the exact
                # Bose-integral anchors Φ^1_1(0) = π²/6, Φ^2_1(0) = 1,
                # Φ^1_2(0) = 2ζ(3) and the Litim closed form
                # 1/[Γ(n+1)(1+w)^p] in the Litim limit.
                y = z / k_val**2
                r = self._shape_numerical(y)
                dr = self._shape_derivative_numerical(y)
                R_val = z * r
                num_val = -z * y * dr
                denom = (z + R_val + k_val**2 * w_val)**p
                if abs(denom) < 1e-300:
                    return 0.0
                return z**(n - 1) * num_val / denom
        else:
            def integrand(z):
                y = z / k_val**2
                r = self._shape_numerical(y)
                R_val = z * r
                denom = (z + R_val + k_val**2 * w_val)**p
                if abs(denom) < 1e-300:
                    return 0.0
                return z**(n - 1) * R_val / denom

        result, _ = quad(integrand, 0, np.inf, limit=200)
        return result / mgamma(n)

    def _shape_numerical(self, y: float) -> float:
        """Numerical shape function r(y) = 1/(e^y - 1) for the
        exponential regulator.

        For large y, e^y (and a fortiori (e^y - 1)²) overflows double
        precision; use the exact asymptotic r(y) ≈ e^{-y} instead
        (relative error < e^{-350}).
        """
        import numpy as np
        if isinstance(self.regulator, ExponentialRegulator):
            if y < 1e-10:
                return 1.0 / y if y > 0 else 1e10
            if y > 350.0:
                return float(np.exp(-y))
            return 1.0 / (np.exp(y) - 1)
        raise NotImplementedError(f"Numerical shape for {self.regulator.name}")

    def _shape_derivative_numerical(self, y: float) -> float:
        """Numerical dr/dy = -e^y/(e^y - 1)² for the exponential regulator.

        For large y, (e^y - 1)² overflows double precision (and for
        y > 709 the naive ratio evaluates to inf/inf = NaN); use the
        exact asymptotic dr/dy ≈ -e^{-y} instead.
        """
        import numpy as np
        if isinstance(self.regulator, ExponentialRegulator):
            if y < 1e-10:
                return -1.0 / y**2 if y > 0 else -1e20
            if y > 350.0:
                return -float(np.exp(-y))
            ey = np.exp(y)
            return -ey / (ey - 1)**2
        raise NotImplementedError


# --- Convenience: Q-functionals ---

class QFunctional:
    """Q-functional Q_n[W] used in the heat kernel trace evaluation.

    Q_n[W] = (1/Γ(n)) ∫_0^∞ dz z^{n-1} W(z)       for n > 0
    Q_0[W] = W(0)                                     for n = 0
    Q_{-m}[W] = (-1)^m W^{(m)}(0)                     for n = -m < 0

    For the Wetterich equation with Litim regulator:
        Q_n[∂_t R_k / (P_k + m^2)^p] = 2k^{2n+2} / [Γ(n+1) (k^2 + m^2)^p]
    """

    def __init__(self, regulator: Regulator | None = None):
        self.regulator = regulator or LitimRegulator()

    def evaluate(self, n: int | Rational, p: int,
                 mass_sq: Expr, k: Symbol) -> Expr:
        """Evaluate Q_n[∂_t R_k / (P_k + m^2)^p].

        For the Litim regulator, this is:
            2 k^{2n+2} / [Γ(n+1) (k^2 + m^2)^p]
        """
        if isinstance(self.regulator, LitimRegulator):
            return 2 * k**(2*n + 2) / (gamma(n + 1) * (k**2 + mass_sq)**p)
        raise NotImplementedError(
            f"Q-functional not implemented for {self.regulator.name}"
        )
