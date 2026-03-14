"""Quadratic gravity truncation: R² + C² extensions to Einstein-Hilbert.

Action:
    Γ_k = ∫d⁴x√g [-R/(16πG) + Λ/(8πG) + α R² + β C_μνρσ C^μνρσ]

The quadratic terms introduce 4th-order propagators that must be
decomposed via partial fractions for trace evaluation.

In d=4, the Weyl-squared term is:
    C² = R_μνρσ R^μνρσ - 2 R_μν R^μν + (1/3) R²

The Gauss-Bonnet term E = R_μνρσ² - 4R_μν² + R² is topological.

References:
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414
    Benedetti, Machado & Saueressig (2009), Mod. Phys. Lett. A24, 2233
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi, sqrt, solve

from asymsafety.core.coupling import CouplingSet, make_quadratic_couplings
from asymsafety.core.truncation import Truncation


@dataclass
class PartialFractionDecomposition:
    """Partial fraction decomposition of a 4th-order propagator.

    Given Γ^(2) = A z² + B z + C (4th order in derivatives),
    the propagator 1/Γ^(2) is decomposed as:

        1/(A(z - z_+)(z - z_-)) = c_+/(z - z_+) + c_-/(z - z_-)

    where z_± are the roots of Az² + Bz + C = 0 and
    c_± = 1/[A(z_+ - z_-)] (with appropriate signs).
    """

    A: Expr  # coefficient of z²
    B: Expr  # coefficient of z
    C: Expr  # constant term

    @property
    def discriminant(self) -> Expr:
        """Discriminant: B² - 4AC."""
        return self.B**2 - 4 * self.A * self.C

    @property
    def poles(self) -> tuple[Expr, Expr]:
        """Roots z_± = (-B ± √(B²-4AC)) / (2A)."""
        D = sqrt(self.discriminant)
        z_plus = (-self.B + D) / (2 * self.A)
        z_minus = (-self.B - D) / (2 * self.A)
        return z_plus, z_minus

    @property
    def residues(self) -> tuple[Expr, Expr]:
        """Partial fraction coefficients c_± = ±1/(A(z_+ - z_-))."""
        z_plus, z_minus = self.poles
        diff = z_plus - z_minus
        c_plus = 1 / (self.A * diff)
        c_minus = -1 / (self.A * diff)
        return c_plus, c_minus

    def as_sum_of_second_order(self) -> list[tuple[Expr, Expr]]:
        """Return [(coefficient, mass²), ...] for each pole.

        Each entry (c, m²) represents c/(z - m²), i.e., a second-order
        propagator with effective mass² = -m².
        """
        z_plus, z_minus = self.poles
        c_plus, c_minus = self.residues
        return [(c_plus, z_plus), (c_minus, z_minus)]


class QuadraticGravityTruncation(Truncation):
    """Quadratic gravity truncation with couplings (G, Λ, α, β).

    Provides Hessians for each spin sector including the higher-derivative
    structure from R² and C² terms.
    """

    def __init__(self, d: int = 4):
        self.d = d
        self._couplings = make_quadratic_couplings(d)

    def couplings(self) -> CouplingSet:
        return self._couplings

    def action_symbolic(self, d: int = 4) -> Expr:
        """Full quadratic gravity Lagrangian density."""
        G = Symbol("G", positive=True)
        Lambda = Symbol("Lambda", real=True)
        alpha = Symbol("alpha", real=True)
        beta_ = Symbol("beta", real=True)
        R = Symbol("R")
        C2 = Symbol("C2")  # Weyl squared

        return (R - 2 * Lambda) / (16 * pi * G) + alpha * R**2 + beta_ * C2

    def hessian(self, sector: str, z: Symbol,
                R_bar: Symbol, d: int = 4) -> Expr | sympy.Matrix:
        """Hessian including higher-derivative contributions.

        The R² term contributes to the scalar sector (spin-0).
        The C² term contributes to the TT sector (spin-2).

        In each sector, Γ^(2) becomes a 4th-order differential operator
        (quadratic in z), requiring partial-fraction decomposition.
        """
        G = Symbol("G", positive=True)
        Lambda = Symbol("Lambda", real=True)
        alpha = Symbol("alpha", real=True)
        beta_ = Symbol("beta", real=True)
        mu = 1 / (32 * pi * G)

        if sector == "TT":
            return self._hessian_TT(z, R_bar, d, mu, Lambda, alpha, beta_)
        elif sector == "scalar":
            return self._hessian_scalar(z, R_bar, d, mu, Lambda, alpha, beta_)
        elif sector == "ghost":
            return self._hessian_ghost(z, R_bar, d)
        else:
            raise ValueError(f"Unknown sector: {sector}")

    def _hessian_TT(self, z, R_bar, d, mu, Lambda, alpha, beta_):
        """TT sector with C² contribution.

        Γ^(2)_TT = β z² + (μ - 2β R̄/(d-1)) z + [lower order in R̄]

        The C² coupling β generates a z² term, making this 4th-order.
        """
        c_TT = Rational(d**2 - 3*d + 4, d * (d - 1))

        # EH part: mu (z - 2Λ + c_TT R̄)
        eh_part = mu * (z - 2 * Lambda + c_TT * R_bar)

        # C² part: β z² - 2β R̄ z / (d-1) + ...
        # On S^d, the C² Hessian for TT modes is:
        c2_part = beta_ * z**2 - 2 * beta_ * R_bar * z / (d * (d - 1))

        return c2_part + eh_part

    def _hessian_scalar(self, z, R_bar, d, mu, Lambda, alpha, beta_):
        """Scalar sector (trace mode h) with R² contribution.

        The R² coupling α generates a z² term in the conformal sector:
        Γ^(2)_scalar = 3α(d-1)²/d² · z² + ...

        (The factor 3(d-1)²/d² comes from the second variation of
        ∫√g R² with respect to the conformal mode.)
        """
        # EH contribution (conformal mode, wrong sign)
        eh_part = -mu * Rational(d - 2, 2 * d) * (
            z - 2 * Lambda + Rational(d - 2, d - 1) * R_bar
        )

        # R² contribution
        r2_coeff = 3 * alpha * Rational((d - 1)**2, d**2)
        r2_part = r2_coeff * z**2

        return r2_part + eh_part

    def _hessian_ghost(self, z, R_bar, d):
        """Ghost sector (unchanged from EH)."""
        return -(z + R_bar / d)

    def partial_fraction_TT(self, R_bar: Symbol,
                            d: int = 4) -> PartialFractionDecomposition:
        """Decompose the TT propagator into partial fractions.

        Returns the PartialFractionDecomposition for the TT sector.
        """
        z = Symbol("z")
        hess = self.hessian("TT", z, R_bar, d)
        # Extract coefficients of z^2, z, z^0
        A = hess.coeff(z, 2)
        B = hess.coeff(z, 1)
        C = hess.subs(z, 0)
        return PartialFractionDecomposition(A=A, B=B, C=C)

    def partial_fraction_scalar(self, R_bar: Symbol,
                                d: int = 4) -> PartialFractionDecomposition:
        """Decompose the scalar propagator into partial fractions."""
        z = Symbol("z")
        hess = self.hessian("scalar", z, R_bar, d)
        A = hess.coeff(z, 2)
        B = hess.coeff(z, 1)
        C = hess.subs(z, 0)
        return PartialFractionDecomposition(A=A, B=B, C=C)
