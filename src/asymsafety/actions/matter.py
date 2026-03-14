"""Matter field actions: scalars and gauge fields.

Provides matter field contributions to the gravitational beta functions.
Supports:
    - Minimally coupled scalars: S = (1/2) ∫√g g^μν ∂_μφ ∂_νφ
    - Non-minimally coupled scalars: S = (1/2) ∫√g (g^μν ∂_μφ ∂_νφ + ξRφ²)
    - Abelian gauge fields: S = (1/4) ∫√g F_μν F^μν
    - Non-abelian gauge fields: S = (1/4) ∫√g tr(F_μν F^μν)

Matter fields contribute to the gravitational beta functions through
loop corrections, parameterized by the number of fields (N_s, N_D, N_v).

References:
    Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035
    Dona, Eichhorn, Labus & Percacci (2016), Phys. Rev. D 93, 044049
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.frg.threshold import ThresholdFunctions


@dataclass
class MatterContent:
    """Specification of matter field content.

    Attributes:
        n_scalars: Number of real scalar fields.
        n_dirac: Number of Dirac fermions.
        n_vectors: Number of abelian gauge fields.
        xi: Non-minimal coupling for scalars (ξ R φ²).
    """

    n_scalars: int = 0
    n_dirac: int = 0
    n_vectors: int = 0
    xi: Expr = sympy.S.Zero  # ξ = 0 is minimal coupling

    @property
    def total_bosonic_dof(self) -> int:
        """Total bosonic degrees of freedom (d=4)."""
        return self.n_scalars + 2 * self.n_vectors

    @property
    def total_fermionic_dof(self) -> int:
        """Total fermionic degrees of freedom (d=4, Dirac)."""
        return 4 * self.n_dirac

    def is_empty(self) -> bool:
        return self.n_scalars == 0 and self.n_dirac == 0 and self.n_vectors == 0


class ScalarFieldAction:
    """Scalar field action with optional non-minimal coupling.

    S = (1/2) ∫d^dx √g [g^μν ∂_μφ ∂_νφ + ξ R φ² + m² φ²]

    The operator acting on φ is: Δ_ξ = -D² + ξR + m².
    The endomorphism is: E = -ξR - m².
    """

    def __init__(self, xi: Expr = sympy.S.Zero,
                 mass_sq: Expr = sympy.S.Zero):
        self.xi = xi
        self.mass_sq = mass_sq

    def operator_endomorphism(self, R_bar: Symbol) -> Expr:
        """Endomorphism E = -ξR̄ - m² for the scalar operator."""
        return -self.xi * R_bar - self.mass_sq

    def effective_mass_sq(self, R_bar: Symbol, k: Symbol) -> Expr:
        """Effective mass² in units of k²: w = (ξR̄ + m²)/k²."""
        return (self.xi * R_bar + self.mass_sq) / k**2

    @property
    def b2_coefficient(self) -> Expr:
        """Coefficient of R̄ in b_2: (1/6 - ξ).

        Vanishes at conformal coupling ξ = 1/6 (in d=4).
        """
        return Rational(1, 6) - self.xi

    def gravity_beta_contribution(self, lam: Symbol,
                                   d: int = 4) -> dict[str, Expr]:
        """Contribution to gravitational beta functions per scalar.

        Returns the contributions to η_N (A and B coefficients)
        and to β_λ from one minimally coupled scalar.
        """
        tf = ThresholdFunctions()  # Litim

        # Scalar loop contribution to the volume (cosmological const) term:
        # ∝ Φ^1_{d/2}(0) for a massless minimally coupled scalar
        Phi_1_d2_0 = tf.Phi(1, Rational(d, 2), 0)

        # Contribution to the R term (Newton coupling):
        # ∝ (1/6 - ξ) Φ^1_{d/2-1}(0)
        Phi_1_d2m1_0 = tf.Phi(1, Rational(d, 2) - 1, 0)

        volume_contrib = Rational(1, 2) * (4 * pi)**(-Rational(d, 2)) * 2 * Phi_1_d2_0
        R_contrib = (Rational(1, 2) * (4 * pi)**(-Rational(d, 2))
                     * 2 * self.b2_coefficient * Phi_1_d2m1_0)

        return {
            "volume": volume_contrib,
            "R": R_contrib,
        }


class GaugeFieldAction:
    """Gauge field action: S = (1/4) ∫√g F_μν F^μν.

    For the FRG, the gauge field contributes:
    - d-1 transverse modes (spin-1)
    - minus 1 ghost mode (scalar ghost from gauge fixing)

    Net contribution: (d-2) effective degrees of freedom per gauge field.
    """

    def __init__(self, gauge_group: str = "U(1)", rank: int = 1):
        self.gauge_group = gauge_group
        self.rank = rank  # dimension of adjoint representation

    @property
    def adjoint_dim(self) -> int:
        """Dimension of the adjoint representation.

        U(1): 1
        SU(N): N²-1
        """
        if self.gauge_group == "U(1)":
            return 1
        elif self.gauge_group.startswith("SU("):
            N = self.rank
            return N**2 - 1
        return self.rank

    def gravity_beta_contribution(self, lam: Symbol,
                                   d: int = 4) -> dict[str, Expr]:
        """Contribution to gravitational beta functions per gauge field.

        The transverse vector contributes (d-1) times the scalar result,
        minus 1 ghost, giving net (d-2) factor.
        """
        tf = ThresholdFunctions()
        Phi_1_d2_0 = tf.Phi(1, Rational(d, 2), 0)
        Phi_1_d2m1_0 = tf.Phi(1, Rational(d, 2) - 1, 0)

        prefactor = Rational(1, 2) * (4 * pi)**(-Rational(d, 2))

        # Transverse vector: (d-1) modes, endomorphism E_μν = -R̄_μν
        # Ghost: 1 scalar mode, subtracted
        # Net for b_0: (d-1) - 1 = d-2
        volume_contrib = prefactor * 2 * (d - 2) * Phi_1_d2_0

        # For b_2: vector has different R̄ coefficient than scalar ghost
        # Vector: tr(R̄/6 - E) = (d-1)(R̄/6 + R̄/d) per component
        # Ghost: R̄/6
        R_contrib = prefactor * 2 * (
            (d - 1) * (Rational(1, 6) + Rational(1, d))
            - Rational(1, 6)
        ) * Phi_1_d2m1_0

        return {
            "volume": volume_contrib * self.adjoint_dim,
            "R": R_contrib * self.adjoint_dim,
        }


def matter_eta_N_correction(matter: MatterContent,
                             d: int = 4) -> tuple[Expr, Expr]:
    """Compute matter contributions to the anomalous dimension.

    η_N = g·(A_grav + A_matter) / (1 - g·(B_grav + B_matter))

    Returns (A_matter, B_matter) contributions.

    For the Litim regulator in d=4:
        Scalar: A_s = -1/(12π), B_s = 0
        Dirac:  A_D = 1/(6π),   B_D = 0
        Vector: A_v = -(d-2)/(6π), B_v = 0

    (Matter does not contribute to B at one loop in the standard approach.)
    """
    tf = ThresholdFunctions()
    Phi_1_1_0 = tf.Phi(1, 1, 0)  # = 1

    # Scalar contribution to A
    A_scalar = -Rational(1, 12) / pi * Phi_1_1_0

    # Dirac fermion contribution to A
    A_dirac = Rational(1, 6) / pi * Phi_1_1_0

    # Vector contribution to A: (d-2) net dof
    A_vector = -Rational(d - 2, 6) / pi * Phi_1_1_0

    A_matter = (
        matter.n_scalars * A_scalar
        + matter.n_dirac * A_dirac
        + matter.n_vectors * A_vector
    )

    # B_matter = 0 at one loop (matter does not couple to ∂_t R_k^grav)
    B_matter = sympy.S.Zero

    return A_matter, B_matter
