"""Seeley-DeWitt heat kernel coefficients for trace evaluation.

The heat kernel expansion for a Laplace-type operator Δ = -D² + E
on a d-dimensional manifold (without boundary):

    Tr[exp(-sΔ)] = (4πs)^{-d/2} Σ_{n=0}^∞ s^n b_{2n}(Δ)

Integrated coefficients on a maximally symmetric space (S^d):

    b_0 = tr(I) · Vol
    b_2 = ∫√g tr[(R/6)I - E]
    b_4 = ∫√g tr[ (1/180)R²_μνρσ I - (1/180)R²_μν I + (1/72)R² I
                   + (1/12)Ω_μν Ω^μν + (1/2)E² - (1/6)RE + ...]

where Ω_μν is the curvature of the bundle connection and E is the
endomorphism (potential) term.

On S^d (maximally symmetric), the curvature invariants simplify:
    R²_μνρσ = 2R̄²/[d(d-1)]
    R²_μν = R̄²/d
    Ω_μν for various field types has known values

References:
    Vassilevich (2003), Phys. Rept. 388, 279 [hep-th/0306138]
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414
    Saueressig (2023), in Handbook of Quantum Gravity [2302.14152]
        (current review of heat kernel methods in gravitational FRG)
"""

from dataclasses import dataclass
from typing import Literal

import sympy
from sympy import Expr, Rational, Symbol, pi, sqrt

from asymsafety.geometry.curvature import MaxSymBackground


FieldType = Literal[
    "scalar", "vector", "symmetric_tensor", "TT_tensor", "ghost_vector"
]


@dataclass
class SeeleyDeWittCoefficients:
    """Heat kernel coefficients on a maximally symmetric background.

    All coefficients are expressed as polynomials in R̄ (background
    Ricci scalar) times the volume, evaluated per unit volume.
    """

    d: int  # spacetime dimension
    R_bar: Symbol  # background Ricci scalar

    @property
    def bg(self) -> MaxSymBackground:
        return MaxSymBackground(d=self.d, R_bar=self.R_bar)

    def trace_id(self, field_type: FieldType) -> int:
        """tr(I) = number of components of the field.

        scalar: 1
        vector: d
        symmetric_tensor: d(d+1)/2
        TT_tensor: (d+1)(d-2)/2
        ghost_vector: d  (but enters with -2 for Grassmann)
        """
        d = self.d
        traces = {
            "scalar": 1,
            "vector": d,
            "symmetric_tensor": d * (d + 1) // 2,
            "TT_tensor": (d + 1) * (d - 2) // 2,
            "ghost_vector": d,
        }
        return traces[field_type]

    def b0(self, field_type: FieldType) -> Expr:
        """b_0 = tr(I). The leading heat kernel coefficient."""
        return sympy.Integer(self.trace_id(field_type))

    def b2(self, field_type: FieldType,
           endomorphism: Expr = sympy.S.Zero) -> Expr:
        """b_2 per unit volume.

        b_2 = tr[(R̄/6)I - E]

        On S^d, R̄ is constant, so:
            b_2 = tr(I) · R̄/6 - tr(E)

        For standard fields with endomorphism E:
            scalar: E = 0 (minimally coupled) or E = -ξR̄ (non-minimal)
            vector: E_μ^ν = -R̄_μ^ν = -(R̄/d)δ_μ^ν
            ghost_vector: same as vector
            TT_tensor: E from the curvature of the spin connection

        Args:
            field_type: Type of field.
            endomorphism: The endomorphism E (trace over field indices
                         should already be taken if custom).
        """
        d = self.d
        R = self.R_bar
        tr_I = self.trace_id(field_type)

        # Standard endomorphisms on maximally symmetric backgrounds
        if endomorphism == sympy.S.Zero:
            endomorphism = self._standard_endomorphism_trace(field_type)

        return Rational(1, 6) * tr_I * R - endomorphism

    def b4(self, field_type: FieldType,
           endomorphism: Expr = sympy.S.Zero) -> dict[str, Expr]:
        """b_4 per unit volume, decomposed into curvature invariants.

        Returns coefficients in the basis {R², R²_μν, R²_μνρσ}
        (before conversion to the {R², C²} basis).

        b_4 = c₁ R² + c₂ R²_μν + c₃ R²_μνρσ + (endomorphism contributions)

        The universal (curvature-only) part for a field with tr(I) = N is:
            b_4^univ = N [1/180 R²_μνρσ - 1/180 R²_μν + 1/72 R²
                         + 1/30 D²R]  (D²R = 0 on max sym)
                     + 1/12 tr(Ω²) + 1/2 tr(E²) - 1/6 R tr(E) + 1/6 D²tr(E)

        On S^d (constant curvature), the D² terms vanish.
        """
        d = self.d
        R = self.R_bar
        tr_I = self.trace_id(field_type)

        # Universal curvature-squared coefficients
        c_R2 = Rational(1, 72) * tr_I
        c_Ric2 = -Rational(1, 180) * tr_I
        c_Riem2 = Rational(1, 180) * tr_I

        # Bundle curvature contribution: (1/12) tr(Ω_μν Ω^μν)
        omega_sq_trace = self._omega_squared_trace(field_type)
        # Ω² on max sym background: Ω_μν^{ab} = R_μν^{ab} for vectors/tensors
        # For vectors: tr(Ω²) = R_μνρσ R^μνρσ = R²_μνρσ, so adds to c_Riem2
        # More precisely: tr(Ω²_μν) is the trace over internal indices.
        # This contributes to the R²_μνρσ coefficient.
        c_Riem2 += omega_sq_trace["Riem2"]
        c_Ric2 += omega_sq_trace.get("Ric2", sympy.S.Zero)
        c_R2 += omega_sq_trace.get("R2", sympy.S.Zero)

        # Endomorphism contributions: (1/2)tr(E²) - (1/6)R·tr(E)
        if endomorphism == sympy.S.Zero:
            E_trace = self._standard_endomorphism_trace(field_type)
            E2_trace = self._standard_endomorphism_sq_trace(field_type)
        else:
            E_trace = endomorphism
            E2_trace = endomorphism**2 / self.trace_id(field_type)  # Rough

        # These contribute to R² coefficient (since E ∝ R on max sym bg)
        endo_R2 = Rational(1, 2) * E2_trace - Rational(1, 6) * R * E_trace
        # Express in terms of R²:
        # E_trace and E2_trace are polynomials in R, contributing to c_R2
        c_R2_endo = self._extract_R2_coefficient(endo_R2)
        c_R2 += c_R2_endo

        return {"R2": c_R2, "Ric2": c_Ric2, "Riem2": c_Riem2}

    def b4_on_sphere(self, field_type: FieldType,
                     endomorphism: Expr = sympy.S.Zero) -> Expr:
        """b_4 evaluated on S^d (single number times R̄²).

        On S^d, all curvature invariants are proportional to R̄², so
        b_4 = constant × R̄².
        """
        coeffs = self.b4(field_type, endomorphism)
        bg = self.bg

        result = (
            coeffs["R2"] * bg.ricci_scalar_squared
            + coeffs["Ric2"] * bg.ricci_tensor_squared
            + coeffs["Riem2"] * bg.riemann_squared
        )
        return sympy.simplify(result)

    # --- Internal: standard endomorphism and bundle curvature ---

    def _standard_endomorphism_trace(self, field_type: FieldType) -> Expr:
        """tr(E) for standard fields on maximally symmetric backgrounds.

        scalar (minimal): E = 0, tr(E) = 0
        vector: E_μ^ν = -R̄_μ^ν = -(R̄/d)δ_μ^ν, tr(E) = -R̄
        ghost_vector: same as vector, tr(E) = -R̄
        TT_tensor: E comes from curvature terms in the Lichnerowicz operator.
            tr(E_TT) = -2(d²-d-2)/(d(d-1)) R̄ · tr_I_TT / d_TT
            Simplified: tr(E) for TT on S^d depends on d.
        """
        d = self.d
        R = self.R_bar

        if field_type == "scalar":
            return sympy.S.Zero
        elif field_type in ("vector", "ghost_vector"):
            return -R  # tr over d components: d × (-R/d) = -R
        elif field_type == "TT_tensor":
            # Lichnerowicz operator on TT tensors: Δ_L h^TT = -D² h^TT + E h^TT
            # On S^d: E_{ab}^{cd} h^TT_{cd} where E involves Riemann
            # tr(E) = -(d²-3d+4)/(d(d-1)) R̄ × tr_I_TT ... simplified:
            # For d=4: E gives an effective mass-like term
            # tr(E_TT) = 2 × [(d+1)(d-2)/2] × (-R̄/d²) × ...
            # Using the known result: on S^d, the TT Laplacian eigenvalues
            # are shifted by -2R̄/(d(d-1)) per component
            # So tr(E_TT) = tr_I_TT × [some combination]
            # For d=4: tr(E) = 5 × (-2R̄/(4·3)) = -5R̄/6
            # General: tr_I × (-2/(d(d-1))) R̄
            return -2 * self.trace_id(field_type) * R / (d * (d - 1))
        elif field_type == "symmetric_tensor":
            return -2 * R  # Approximate
        return sympy.S.Zero

    def _standard_endomorphism_sq_trace(self, field_type: FieldType) -> Expr:
        """tr(E²) for standard fields."""
        d = self.d
        R = self.R_bar

        if field_type == "scalar":
            return sympy.S.Zero
        elif field_type in ("vector", "ghost_vector"):
            # E_μ^ν = -(R̄/d)δ_μ^ν, tr(E²) = d(R̄/d)² = R̄²/d
            return R**2 / d
        elif field_type == "TT_tensor":
            # tr(E²) for TT tensors on S^d
            # Each component gets E ∝ R̄/(d(d-1)), squared and traced
            tr_I = self.trace_id(field_type)
            return tr_I * 4 * R**2 / (d * (d - 1))**2
        return sympy.S.Zero

    def _omega_squared_trace(self, field_type: FieldType) -> dict[str, Expr]:
        """(1/12) tr(Ω_μν Ω^μν) decomposed into curvature invariants.

        For a vector field, Ω_μν = R_μν (Riemann as SO(d) curvature),
        so tr(Ω²) = R_μνρσ R^μνρσ (Kretschner scalar).

        For a scalar field, Ω = 0 (trivial bundle).
        """
        d = self.d

        if field_type == "scalar":
            return {"Riem2": sympy.S.Zero}
        elif field_type in ("vector", "ghost_vector"):
            # (1/12) × 1 × R²_μνρσ coefficient
            return {"Riem2": Rational(1, 12)}
        elif field_type == "TT_tensor":
            # For rank-2 symmetric traceless tensors, the bundle curvature
            # gives: (1/12) tr(Ω²) = (1/12) × (sum of contributions)
            # This is more complex; on S^d the result is a known multiple
            # of R²_μνρσ.
            # For d=4: (1/12) × [2(d-1)/(d(d-1))] × tr_I × R²_μνρσ
            # Simplified: scales with number of components
            return {"Riem2": Rational(1, 12) * 2 * (d - 1)}
        return {"Riem2": sympy.S.Zero}

    def _extract_R2_coefficient(self, expr: Expr) -> Expr:
        """Extract the coefficient of R̄² from an expression polynomial in R̄."""
        R = self.R_bar
        if expr == sympy.S.Zero:
            return sympy.S.Zero
        # Expand and collect R² terms
        expanded = sympy.expand(expr)
        return expanded.coeff(R, 2)
