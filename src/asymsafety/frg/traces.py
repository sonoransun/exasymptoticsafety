"""Master trace evaluator for the Wetterich equation.

Evaluates the functional trace on the RHS of the Wetterich equation:

    (1/2) Tr[∂_t R_k · (Γ^(2)_k + R_k)^{-1}]

using the heat kernel expansion or spectral sums. The result is
decomposed into curvature invariants {1, R, R², C², ...} whose
coefficients give the running of the respective couplings.

References:
    Saueressig (2023), in Handbook of Quantum Gravity [2302.14152]
        (most current comprehensive review of FRG methods for gravity)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi, gamma

from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients, FieldType
from asymsafety.frg.regulator import LitimRegulator, Regulator
from asymsafety.frg.threshold import QFunctional, ThresholdFunctions
from asymsafety.geometry.curvature import CurvatureInvariantBasis


class TraceEvaluator(ABC):
    """Abstract base for trace evaluation methods."""

    @abstractmethod
    def evaluate(self, field_type: FieldType, endomorphism: Expr,
                 mass_sq: Expr, d: int,
                 max_order: int = 2) -> dict[str, Expr]:
        """Evaluate the trace contribution from one field sector.

        Args:
            field_type: Type of field running in the loop.
            endomorphism: The endomorphism E in Δ = -D² + E.
            mass_sq: Effective mass squared (k^2-normalized).
            d: Spacetime dimension.
            max_order: Maximum curvature order (0: Λ, 1: R, 2: R²+C²).

        Returns:
            Dictionary mapping invariant names to their coefficients:
            {"V": ..., "R": ..., "R2": ..., "C2": ...}
            where V is the volume (cosmological constant) term.
        """


class HeatKernelTraceEvaluator(TraceEvaluator):
    """Evaluate traces using the Seeley-DeWitt heat kernel expansion.

    For a Laplace-type operator Δ = -D² + E with regulator R_k:

    (1/2) Tr[∂_t R_k / (Δ + R_k)] =
        (1/2) · (4π)^{-d/2} · Σ_n b_{2n} · Q_{d/2-n}[∂_t R_k/(P_k+E)]

    For the Litim regulator, Q_n reduces to threshold functions with
    closed-form expressions.
    """

    def __init__(self, regulator: Regulator | None = None):
        self.regulator = regulator or LitimRegulator()
        self.threshold = ThresholdFunctions(self.regulator)
        self.q_func = QFunctional(self.regulator)

    def evaluate(self, field_type: FieldType, endomorphism: Expr,
                 mass_sq: Expr, d: int,
                 max_order: int = 2) -> dict[str, Expr]:
        """Evaluate trace via heat kernel expansion.

        The result for each curvature order n is:
            (1/2) · (4π)^{-d/2} · b_{2n} · Q_{d/2-n}[W]

        where W(z) = ∂_t R_k(z) / (P_k(z) + mass_sq)
        and Q_{d/2-n}[W] is related to threshold functions Φ^1_{d/2-n}(w).
        """
        R_bar = Symbol("R_bar")
        k = Symbol("k", positive=True)
        sd = SeeleyDeWittCoefficients(d=d, R_bar=R_bar)

        # Dimensionless mass: w = mass_sq / k^2
        w = mass_sq

        # Prefactor: (1/2) · (4π)^{-d/2}
        prefactor = Rational(1, 2) * (4 * pi)**(-Rational(d, 2))

        result: dict[str, Expr] = {}

        # --- Order 0: volume term (b_0 × Q_{d/2}) ---
        b0 = sd.b0(field_type)
        # Q_{d/2}[∂_t R_k / (P_k + m²)] = 2k^{d+2} Φ^1_{d/2}(w) / k^2
        # = 2 k^d Φ^1_{d/2}(w) ... but with proper normalization:
        # Using Φ^1_n(w) = 1/[Γ(n+1)(1+w)] for Litim
        phi_d2 = self.threshold.Phi(1, Rational(d, 2), w)
        result["V"] = prefactor * b0 * 2 * phi_d2

        if max_order < 1:
            return result

        # --- Order 1: R term (b_2 × Q_{d/2-1}) ---
        b2 = sd.b2(field_type, endomorphism)
        phi_d2_m1 = self.threshold.Phi(1, Rational(d, 2) - 1, w)
        # b2 is proportional to R_bar, extract the coefficient
        b2_coeff = b2.coeff(R_bar) if b2 != sympy.S.Zero else sympy.S.Zero
        b2_const = b2 - b2_coeff * R_bar
        result["V"] = result["V"] + prefactor * b2_const * 2 * phi_d2_m1
        result["R"] = prefactor * b2_coeff * 2 * phi_d2_m1

        if max_order < 2:
            return result

        # --- Order 2: R² terms (b_4 × Q_{d/2-2}) ---
        b4_coeffs = sd.b4(field_type, endomorphism)
        phi_d2_m2 = self.threshold.Phi(1, Rational(d, 2) - 2, w)

        # Convert to {R², C²} basis
        basis = CurvatureInvariantBasis(d=d)
        if d == 4:
            decomposed = basis.decompose(
                b4_coeffs["R2"], b4_coeffs["Ric2"], b4_coeffs["Riem2"]
            )
            result["R2"] = prefactor * decomposed["R2"] * 2 * phi_d2_m2
            result["C2"] = prefactor * decomposed["C2"] * 2 * phi_d2_m2
        else:
            # For d ≠ 4, keep in the {R², Ric², Riem²} basis
            result["R2"] = prefactor * b4_coeffs["R2"] * 2 * phi_d2_m2
            result["Ric2"] = prefactor * b4_coeffs["Ric2"] * 2 * phi_d2_m2
            result["Riem2"] = prefactor * b4_coeffs["Riem2"] * 2 * phi_d2_m2

        return result

    def evaluate_with_eta(self, field_type: FieldType, endomorphism: Expr,
                          mass_sq: Expr, eta: Symbol,
                          d: int, max_order: int = 1) -> dict[str, Expr]:
        """Evaluate trace including anomalous dimension corrections.

        When the regulator includes wave-function renormalization
        R_k = Z_k r_k, the ∂_t R_k picks up an η contribution:
            ∂_t R_k = (2 - η) k^2 θ(k^2-z)  [for Litim]

        This modifies the threshold functions:
            Φ^p_n → Φ^p_n - η/(2(n+1)) Φ̃^p_n
        """
        R_bar = Symbol("R_bar")
        sd = SeeleyDeWittCoefficients(d=d, R_bar=R_bar)

        w = mass_sq
        prefactor = Rational(1, 2) * (4 * pi)**(-Rational(d, 2))

        result: dict[str, Expr] = {}

        # Order 0: volume
        b0 = sd.b0(field_type)
        phi_adj = self.threshold.Phi_adjusted(1, Rational(d, 2), w, eta)
        result["V"] = prefactor * b0 * 2 * phi_adj

        if max_order >= 1:
            # Order 1: R
            b2 = sd.b2(field_type, endomorphism)
            phi_adj_1 = self.threshold.Phi_adjusted(
                1, Rational(d, 2) - 1, w, eta
            )
            b2_coeff = b2.coeff(R_bar) if b2 != sympy.S.Zero else sympy.S.Zero
            b2_const = b2 - b2_coeff * R_bar
            result["V"] = result["V"] + prefactor * b2_const * 2 * phi_adj_1
            result["R"] = prefactor * b2_coeff * 2 * phi_adj_1

        return result
