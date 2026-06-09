"""Self-consistent anomalous dimension solver.

The anomalous dimension η_N = -∂_t ln(Z_N) = +∂_t ln(G) appears on both
sides of the flow equation because the regulator contains the
wave-function renormalization: R_k ∝ Z_k.  (With g = G k^{d-2} and
β_g = (d-2+η_N) g, consistency forces η_N = +∂_t ln G; at the NGFP
η_N = -(d-2), i.e. G ~ k^{-(d-2)}.)

The structure is always linear in η_N:
    η_N = g · A(λ, α, β) + g · η_N · B(λ, α, β)

which gives the algebraic solution:
    η_N = g · A / (1 - g · B)

References:
    Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    Reuter & Saueressig (2002), Phys. Rev. D 65, 065016
    Litim (2004), Phys. Rev. Lett. 92, 201301 [hep-th/0312114]
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

        Single-metric Einstein-Hilbert truncation with Type Ia cutoff
        and de Donder (harmonic) gauge (Reuter 1998 [hep-th/9605030];
        d-dimensional form as in Reuter & Saueressig [0708.1317],
        Eqs. (4.40)-(4.43)):

            A(λ) = (1/3) (4π)^{1-d/2} [ d(d+1) Φ^1_{d/2-1}(-2λ)
                                        - 6d(d-1) Φ^2_{d/2}(-2λ)
                                        - 4d Φ^1_{d/2-1}(0)
                                        - 24 Φ^2_{d/2}(0) ]
            B(λ) = -(1/6) (4π)^{1-d/2} [ d(d+1) Φ̃^1_{d/2-1}(-2λ)
                                         - 6d(d-1) Φ̃^2_{d/2}(-2λ) ]

        For the Litim regulator in d=4, with x = 1/(1-2λ):

            A(λ) = (1/3π) [5x - 9x² - 7]
            B(λ) = -(1/12π) [5x - 6x²]

        These reproduce the benchmark NGFP g* ≈ 0.707, λ* ≈ 0.193 with
        θ = 1.475 ± 3.043 i (Litim PRL 92, 201301 [hep-th/0312114];
        Codello, Percacci & Rahmede [0805.2909]).
        """
        w = -2 * lam  # mass argument for graviton: w = -2λ
        nd2 = Rational(d, 2)
        prefactor = (4 * pi)**(1 - nd2)

        tf = self.threshold

        # A: η_N-independent part of the R-projection
        #   graviton trace: d(d+1) Φ^1_{d/2-1}(-2λ) - 6d(d-1) Φ^2_{d/2}(-2λ)
        #   ghost trace:    -4d Φ^1_{d/2-1}(0) - 24 Φ^2_{d/2}(0)
        A = Rational(1, 3) * prefactor * (
            d * (d + 1) * tf.Phi(1, nd2 - 1, w)
            - 6 * d * (d - 1) * tf.Phi(2, nd2, w)
            - 4 * d * tf.Phi(1, nd2 - 1, 0)
            - 24 * tf.Phi(2, nd2, 0)
        )

        # B: coefficient of η_N (regulator R_k ∝ Z_N inserts Φ̃ terms;
        # the ghost wave function is not renormalized, so no Φ̃(0) terms)
        B = -Rational(1, 6) * prefactor * (
            d * (d + 1) * tf.Phi_tilde(1, nd2 - 1, w)
            - 6 * d * (d - 1) * tf.Phi_tilde(2, nd2, w)
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
