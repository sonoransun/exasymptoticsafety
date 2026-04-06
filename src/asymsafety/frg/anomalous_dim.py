"""Self-consistent anomalous dimension solver.

The anomalous dimension η_N = -∂_t ln(G) appears on both sides of the
flow equation because the regulator contains the wave-function
renormalization: R_k ∝ Z_k.

The structure is always linear in η_N:
    η_N = g · A(λ, α, β) + g · η_N · B(λ, α, β)

which gives the algebraic solution:
    η_N = g · A / (1 - g · B)

References:
    Reuter & Saueressig (2002), Phys. Rev. D 65, 065016
    Pawlowski & Reichert (2023), [2309.10785]
        (systematic vertex expansion for dynamical graviton propagator)
"""

import sympy
from sympy import Expr, Rational, Symbol, pi, simplify

from asymsafety.frg.threshold import ThresholdFunctions


class AnomalousDimensionSolver:
    """Solve for the graviton anomalous dimension self-consistently."""

    def __init__(self, threshold: ThresholdFunctions | None = None):
        self.threshold = threshold or ThresholdFunctions()

    def solve(self, g: Symbol, A: Expr, B: Expr) -> Expr:
        """Solve η_N = g·A + g·η_N·B for η_N.

        Returns:
            η_N = g·A / (1 - g·B)
        """
        return g * A / (1 - g * B)

    def compute_AB_einstein_hilbert(self, lam: Symbol,
                                    d: int = 4) -> tuple[Expr, Expr]:
        """Compute A and B coefficients for the EH truncation.

        In d=4 with the standard decomposition (Reuter & Saueressig 2002):

        The anomalous dimension comes from the R-projection of the flow.
        The structure involves threshold functions with argument w = -2λ.

        η_N enters through the modified threshold functions:
            Φ̂^p_n = Φ^p_n - η_N/(2(n+1)) Φ̃^p_n

        After separating the η_N-dependent and independent parts:

        A(λ) contains the η_N-independent threshold function contributions
        B(λ) contains the η_N-dependent threshold function contributions

        For the Litim regulator in d=4:
            A(λ) = (1/3π) [5/(1-2λ) - 4 + 6/(1-2λ)²]
            B(λ) = -(1/6π) [5/(2(1-2λ)) - 2 + 3/(1-2λ)²]

        (Exact coefficients depend on gauge choice; these are for β=0 gauge.)
        """
        w = -2 * lam  # mass argument for graviton: w = -2λ

        tf = self.threshold

        # TT graviton contribution (5 dof in d=4)
        # Phi^1_1(-2λ) and Phi^2_1(-2λ)
        Phi_1_1_w = tf.Phi(1, 1, w)
        Phi_2_1_w = tf.Phi(2, 1, w)
        Phi_1_1_0 = tf.Phi(1, 1, 0)

        tPhi_1_1_w = tf.Phi_tilde(1, 1, w)
        tPhi_2_1_w = tf.Phi_tilde(2, 1, w)
        tPhi_1_1_0 = tf.Phi_tilde(1, 1, 0)

        # A: η_N-independent part
        # From spin-2 (TT): 5 modes, each contributing
        # From ghosts: d = 4 vector ghost modes
        # A = (1/3π) [5 Φ^1_1(-2λ) - 4 Φ^1_1(0) + 6 Φ^2_1(-2λ)]
        A = Rational(1, 3) / pi * (
            5 * Phi_1_1_w
            - 4 * Phi_1_1_0
            + 6 * Phi_2_1_w
        )

        # B: coefficient of η_N
        # B = -(1/6π) [5 Φ̃^1_1(-2λ)/2 - 2 Φ̃^1_1(0)/2 + 3 Φ̃^2_1(-2λ)/2]
        # Note: the factor 1/(2(n+1)) = 1/4 for n=1 threshold functions
        B = -Rational(1, 6) / pi * (
            5 * tPhi_1_1_w
            - 4 * tPhi_1_1_0
            + 6 * tPhi_2_1_w
        )

        return A, B

    def eta_N_einstein_hilbert(self, g: Symbol, lam: Symbol,
                                d: int = 4) -> Expr:
        """Full expression for η_N in the Einstein-Hilbert truncation.

        Returns:
            η_N(g, λ) = g·A(λ) / (1 - g·B(λ))
        """
        A, B = self.compute_AB_einstein_hilbert(lam, d)
        return self.solve(g, A, B)
