"""Curvature invariants on maximally symmetric backgrounds.

On a maximally symmetric space (e.g., S^d) with Ricci scalar R̄, all
curvature tensors are determined by R̄ and the metric:

    R̄_μνρσ = R̄/[d(d-1)] (g_μρ g_νσ - g_μσ g_νρ)
    R̄_μν = (R̄/d) g_μν
    C_μνρσ = 0  (Weyl tensor vanishes on maximally symmetric spaces)
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol


@dataclass
class MaxSymBackground:
    """Maximally symmetric background spacetime.

    Attributes:
        d: Spacetime dimension.
        R_bar: Background Ricci scalar (symbolic).
    """

    d: int
    R_bar: Symbol

    @property
    def riemann_squared(self) -> Expr:
        """R̄_μνρσ R̄^μνρσ = 2R̄²/[d(d-1)]."""
        return 2 * self.R_bar**2 / (self.d * (self.d - 1))

    @property
    def ricci_tensor_squared(self) -> Expr:
        """R̄_μν R̄^μν = R̄²/d."""
        return self.R_bar**2 / self.d

    @property
    def ricci_scalar_squared(self) -> Expr:
        """R̄² (trivially)."""
        return self.R_bar**2

    @property
    def weyl_squared(self) -> Expr:
        """C_μνρσ C^μνρσ = 0 on maximally symmetric spaces."""
        return sympy.S.Zero

    @property
    def gauss_bonnet(self) -> Expr:
        """Gauss-Bonnet invariant E = R²_μνρσ - 4R²_μν + R².

        On maximally symmetric spaces:
            E = [2/(d(d-1)) - 4/d + 1] R̄²
        """
        return (
            Rational(2, self.d * (self.d - 1))
            - Rational(4, self.d)
            + 1
        ) * self.R_bar**2

    @property
    def kretschner_scalar(self) -> Expr:
        """Kretschner scalar (alias for riemann_squared)."""
        return self.riemann_squared

    @property
    def sphere_radius_squared(self) -> Expr:
        """Radius a² of the sphere: R̄ = d(d-1)/a²."""
        return self.d * (self.d - 1) / self.R_bar

    @property
    def volume_S4(self) -> Expr:
        """Volume of S^d with radius a. For S^4: Vol = 8π²a⁴/3."""
        if self.d == 4:
            a_sq = self.sphere_radius_squared
            return 8 * sympy.pi**2 * a_sq**2 / 3
        raise NotImplementedError(f"Volume for S^{self.d}")


class CurvatureInvariantBasis:
    """Track contributions to independent curvature-squared invariants.

    In d=4, the independent basis is {R², C²} (Gauss-Bonnet is topological).
    The heat kernel b_4 coefficient for each field type gives:
        b_4 = c_1 R² + c_2 R²_μν + c_3 R²_μνρσ + (endomorphism terms)

    We convert to the {R², C²} basis using:
        C² = R²_μνρσ - 2R²_μν + (1/3)R²     (in d=4)
        E  = R²_μνρσ - 4R²_μν + R²           (topological in d=4)

    So: R²_μνρσ = C² + 2R²_μν - (1/3)R²
    and: R²_μν = (1/2)(C² - E) + (2/3)R²     ... etc.
    """

    def __init__(self, d: int = 4):
        self.d = d

    def decompose(self, c_R2: Expr, c_Ric2: Expr,
                  c_Riem2: Expr) -> dict[str, Expr]:
        """Convert {R², R²_μν, R²_μνρσ} coefficients to {R², C²} basis.

        In d=4:
            C² = R²_μνρσ - 2 R²_μν + (1/3) R²
        So the C² coefficient is c_Riem2, and the R² coefficient gets
        contributions from all three.

        Since E = R²_μνρσ - 4R²_μν + R² is topological (does not flow),
        we can ignore its coefficient.

        Returns:
            {"R2": coefficient, "C2": coefficient, "E": coefficient}
        """
        if self.d != 4:
            raise NotImplementedError("Decomposition only implemented for d=4")

        # C² = R²_μνρσ - 2R²_μν + (1/3)R²
        c_C2 = c_Riem2
        c_C2_from_Ric = -2 * c_Ric2
        c_C2_from_R2 = Rational(1, 3) * c_R2

        # On a general background:
        # ∫√g [c₁R² + c₂R²_μν + c₃R²_μνρσ]
        # = ∫√g [(c₁ + c₃/3) R² + (c₂ - 2c₃) (R²_μν - R²/d) + c₃ C²]
        # We need to express this in terms of R² and C²:

        # Using C² = R²_μνρσ - 2R²_μν + (1/3)R²
        # and E = R²_μνρσ - 4R²_μν + R²
        # => R²_μν = (1/2)(R²_μνρσ - E) + (1/2)R²
        # But we should work directly:

        # The coefficient of C² is c₃ (from R²_μνρσ) after rewriting
        # using Gauss-Bonnet to eliminate R²_μν:
        # R²_μν = (1/2)(C² + (1/3)R² - E + 4R²_μν - R²) ... this is circular.

        # Direct approach: on the invariant basis {R², C², E}:
        # R² → R²                    contributes (1, 0, 0)
        # R²_μν → ... use R²_μν = (C² - E)/2 + (2/3)R²
        #   but this isn't right either. Let's use the inverse:

        # Forward: C² = R²_μνρσ - 2R²_μν + (1/3)R²
        #          E  = R²_μνρσ - 4R²_μν + R²
        # Inverse: R²_μνρσ = C² + 2R²_μν - (1/3)R²
        #          Subtract: C² - E = 2R²_μν - (2/3)R²
        #          => R²_μν = (C² - E)/2 + (1/3)R²

        # So: c₁ R² + c₂ R²_μν + c₃ R²_μνρσ
        #   = c₁ R² + c₂[(C²-E)/2 + (1/3)R²] + c₃[C² + 2R²_μν - (1/3)R²]
        # Need to substitute R²_μν again... Let's solve the linear system.

        # Let the basis be (R², C², E). We have:
        #   R²      = (1) R² + (0) C² + (0) E
        #   R²_μν   = (1/3) R² + (1/2) C² + (-1/2) E
        #   R²_μνρσ = (-1/3) R² + (1) C² + (1) E    ... from R²_μνρσ = C² + E + 4R²_μν - R²
        #           wait, E = R²_μνρσ - 4R²_μν + R²
        #           => R²_μνρσ = E + 4R²_μν - R²
        #                      = E + 4[(1/3)R² + (1/2)C² + (-1/2)E] - R²
        #                      = E + (4/3)R² + 2C² - 2E - R²
        #                      = (1/3)R² + 2C² - E
        # Hmm, let me redo this carefully.

        # From the definitions:
        #   C² = R²_μνρσ - 2R²_μν + (1/3)R²     ... (1)
        #   E  = R²_μνρσ - 4R²_μν + R²           ... (2)
        # Solve for R²_μν and R²_μνρσ:
        #   (1) - (2): C² - E = 2R²_μν - (2/3)R²
        #   => R²_μν = (C² - E)/2 + (1/3)R²
        #   From (2): R²_μνρσ = E + 4R²_μν - R²
        #           = E + 4[(C²-E)/2 + (1/3)R²] - R²
        #           = E + 2(C²-E) + (4/3)R² - R²
        #           = E + 2C² - 2E + (1/3)R²
        #           = 2C² - E + (1/3)R²

        # Conversion matrix M: (R², R²_μν, R²_μνρσ)^T = M · (R², C², E)^T
        # M = [[1,     0,     0   ],
        #      [1/3,   1/2,  -1/2 ],
        #      [1/3,   2,    -1   ]]

        # We want: c₁R² + c₂R²_μν + c₃R²_μνρσ = a₁R² + a₂C² + a₃E
        # where (a₁, a₂, a₃) = (c₁, c₂, c₃) · M
        # a₁ = c₁·1 + c₂·(1/3) + c₃·(1/3)
        # a₂ = c₂·(1/2) + c₃·2
        # a₃ = c₂·(-1/2) + c₃·(-1)

        a_R2 = c_R2 + c_Ric2 * Rational(1, 3) + c_Riem2 * Rational(1, 3)
        a_C2 = c_Ric2 * Rational(1, 2) + c_Riem2 * 2
        a_E = -c_Ric2 * Rational(1, 2) - c_Riem2

        return {"R2": a_R2, "C2": a_C2, "E": a_E}

    def weyl_squared_4d(self, R_bar: Symbol) -> Expr:
        """C² = R²_μνρσ - 2R²_μν + (1/3)R² in d=4."""
        bg = MaxSymBackground(d=4, R_bar=R_bar)
        return bg.riemann_squared - 2 * bg.ricci_tensor_squared + Rational(1, 3) * bg.ricci_scalar_squared
