"""Projection of the flow equation onto the truncation basis.

After evaluating the functional trace, we obtain the RHS of the
Wetterich equation decomposed into curvature invariants:

    ∂_t Γ_k = ∫√g [a_0 + a_1 R + a_2 R² + a_3 C² + ...]

Comparing with the truncation ansatz:

    Γ_k = ∫√g [Λ/(8πG) - R/(16πG) + α R² + β C²]

we extract the beta functions:

    ∂_t [Λ/(8πG)] = a_0     → gives ∂_t Λ and ∂_t G
    ∂_t [-1/(16πG)] = a_1   → gives ∂_t G
    ∂_t α = a_2              → gives β_α
    ∂_t β = a_3              → gives β_β
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi, solve


@dataclass
class ProjectionResult:
    """Result of projecting the flow onto the truncation basis.

    Stores the coefficients of each curvature invariant in the
    trace evaluation, and provides methods to extract beta functions.
    """

    # Coefficients from trace evaluation (RHS of Wetterich eq)
    volume_coeff: Expr = sympy.S.Zero   # coefficient of ∫√g (1)
    R_coeff: Expr = sympy.S.Zero        # coefficient of ∫√g R
    R2_coeff: Expr = sympy.S.Zero       # coefficient of ∫√g R²
    C2_coeff: Expr = sympy.S.Zero       # coefficient of ∫√g C²

    def extract_eh_betas(self, g: Symbol, lam: Symbol,
                         eta_N: Expr, d: int = 4) -> dict[str, Expr]:
        """Extract Einstein-Hilbert beta functions from projection.

        The truncation is:
            Γ_k = (k^d / 16πg) ∫√̃g (R̃ - 2λ)

        where tildes denote dimensionless quantities.

        ∂_t Γ_k = ∂_t[k^d/(16πg)] ∫(R̃-2λ) + [k^d/(16πg)] ∫∂_t(-2λ)

        The volume term gives: ∂_t[-2λ k^d/(16πg)] = volume_coeff
        The R term gives: ∂_t[k^d/(16πg)] = R_coeff

        From R_coeff:
            ∂_t[1/(16πg)] = R_coeff / k^d
            -(1/(16πg²)) ∂_t g = R_coeff / k^d
            β_g = ∂_t g = -16πg² k^{-d} R_coeff

        But we also know β_g = (d-2+η_N) g from dimensional analysis.
        The self-consistent approach uses both.

        Returns:
            {"beta_g": ..., "beta_lambda": ..., "eta_N": ...}
        """
        # The R coefficient gives the running of 1/(16πG):
        # ∂_t[1/(16πg)] = (d-2+η_N)/(16πg) [from dim analysis + anomalous dim]
        # This is already incorporated in the standard approach.

        # Standard EH beta functions (dimensionless, d=4):
        # β_g = (2 + η_N) g
        # β_λ = (η_N - 2)λ + g/(2π) [trace contributions]
        beta_g = (d - 2 + eta_N) * g

        # The cosmological constant beta function:
        # β_λ = -(2 - η_N)λ + (16πg) × volume_coeff / (something)
        # The trace contributions modify the naive dimensional running.
        beta_lambda = -(2 - eta_N) * lam + 8 * pi * g * self.volume_coeff

        return {
            "beta_g": beta_g,
            "beta_lambda": beta_lambda,
        }

    def extract_quadratic_betas(self) -> dict[str, Expr]:
        """Extract quadratic gravity beta functions.

        β_α = R2_coeff (running of R² coupling)
        β_β = C2_coeff (running of C² coupling)
        """
        return {
            "beta_alpha": self.R2_coeff,
            "beta_beta": self.C2_coeff,
        }
