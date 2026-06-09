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
            TT_tensor: E = 0 (bare constrained -D²); pass the trace of
                the Lichnerowicz endomorphism explicitly if needed.

        The TT tensor is a *constrained* bundle: the unconstrained
        master formula tr(I)R̄/6 - tr(E) does not apply. Its b_2 is
        computed from the exact S^4 spectrum (eigenvalues l(l+3)-2,
        degeneracies 5(2l+3)(l-1)(l+4)/6, l ≥ 2 [Rubin & Ordóñez 1984]),
        which gives b_2(-D²|_TT) = -(5/6) R̄; cf. the constrained
        spectral route of Lauscher & Reuter, Phys. Rev. D 65 (2002)
        025013. For Δ = -D²|_TT + E this becomes -(5/6)R̄ - tr(E)
        (Lichnerowicz on S^4: tr(E) = (10/3)R̄ ⇒ b_2 = -(25/6)R̄).

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

        if field_type == "TT_tensor":
            if d != 4:
                raise NotImplementedError(
                    "Constrained TT heat kernel coefficients are only "
                    "implemented on S^4 (d=4)"
                )
            return -Rational(5, 6) * R - endomorphism

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

        For the constrained TT bundle the unconstrained master formula
        does not apply; see _b4_TT_constrained.
        """
        d = self.d
        R = self.R_bar
        tr_I = self.trace_id(field_type)

        if field_type == "TT_tensor":
            return self._b4_TT_constrained(endomorphism)

        # Universal curvature-squared coefficients
        c_R2 = Rational(1, 72) * tr_I
        c_Ric2 = -Rational(1, 180) * tr_I
        c_Riem2 = Rational(1, 180) * tr_I

        # Bundle curvature contribution: (1/12) tr(Ω_μν Ω^μν)
        omega_sq_trace = self._omega_squared_trace(field_type)
        # Ω_μν is the SO(d) curvature acting on the bundle indices; its
        # internal trace is negative, tr(Ω²) = -R²_μνρσ for a vector
        # (antisymmetry of Ω makes the sign convention-independent), so
        # the bundle term *subtracts* from the R²_μνρσ coefficient:
        # spin-1 total in d=4 is 4/180 - 1/12 = -11/180
        # [Christensen & Duff 1979; Vassilevich 2003].
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

    # --- Internal: constrained TT bundle (S^4 spectral route) ---

    def _b4_TT_constrained(self, endomorphism: Expr = sympy.S.Zero
                           ) -> dict[str, Expr]:
        """b_4 for the constrained TT bundle on S^4.

        The transversality constraint changes the heat kernel
        coefficients relative to the unconstrained Sym² master formula,
        so b_4 is computed from the exact S^4 spectrum (eigenvalues
        l(l+3) - 2, degeneracies 5(2l+3)(l-1)(l+4)/6, l ≥ 2
        [Rubin & Ordóñez, J. Math. Phys. 25 (1984) 2888]); this is the
        constrained spectral route of Lauscher & Reuter, Phys. Rev. D
        65 (2002) 025013. The exact mode sum gives:

            b_4(-D²|_TT) = -(1/432) R̄²

        For Δ = -D²|_TT + E with E ∝ identity on the TT bundle (the
        only case on a maximally symmetric background),
        Tr e^{-tΔ} = e^{-t tr(E)/5} Tr e^{+t D²} yields:

            b_4 = -(1/432) R̄² + (1/6) R̄ tr(E) + (1/10) tr(E)²

        (Lichnerowicz on S^4, tr(E) = (10/3) R̄: b_4 = (719/432) R̄².)

        On S^d the three curvature invariants degenerate, so the
        decomposition into {R², Ric², Riem²} is not determined by
        sphere data: the total is returned as an effective R̄²
        coefficient, valid on maximally symmetric backgrounds only.
        """
        if self.d != 4:
            raise NotImplementedError(
                "Constrained TT heat kernel coefficients are only "
                "implemented on S^4 (d=4)"
            )
        R = self.R_bar
        if endomorphism == sympy.S.Zero:
            endomorphism = self._standard_endomorphism_trace("TT_tensor")

        if endomorphism == sympy.S.Zero:
            c_R2 = -Rational(1, 432)
        else:
            total = (
                -Rational(1, 432) * R**2
                + Rational(1, 6) * R * endomorphism
                + Rational(1, 10) * endomorphism**2
            )
            c_R2 = self._extract_R2_coefficient(total)
        return {"R2": c_R2, "Ric2": sympy.S.Zero, "Riem2": sympy.S.Zero}

    # --- Internal: standard endomorphism and bundle curvature ---

    def _standard_endomorphism_trace(self, field_type: FieldType) -> Expr:
        """tr(E) for standard fields on maximally symmetric backgrounds.

        scalar (minimal): E = 0, tr(E) = 0
        vector: E_μ^ν = -R̄_μ^ν = -(R̄/d)δ_μ^ν, tr(E) = -R̄
        ghost_vector: same as vector, tr(E) = -R̄
        TT_tensor: E = 0 — the TT row is defined for the bare
            constrained Laplacian -D²|_TT, whose coefficients are
            computed via the exact S^4 spectrum (see b2 /
            _b4_TT_constrained). For the Lichnerowicz operator on S^4
            pass tr(E) = (2/3)R̄ × 5 = (10/3)R̄ explicitly.
        """
        d = self.d
        R = self.R_bar

        if field_type == "scalar":
            return sympy.S.Zero
        elif field_type in ("vector", "ghost_vector"):
            return -R  # tr over d components: d × (-R/d) = -R
        elif field_type == "TT_tensor":
            return sympy.S.Zero  # bare -D²|_TT (constrained spectral route)
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
            return sympy.S.Zero  # bare -D²|_TT (constrained spectral route)
        return sympy.S.Zero

    def _omega_squared_trace(self, field_type: FieldType) -> dict[str, Expr]:
        """(1/12) tr(Ω_μν Ω^μν) decomposed into curvature invariants.

        Ω_μν is antisymmetric in its bundle indices, so the internal
        trace tr(Ω_μν Ω^μν) is *negative* definite in curvature-squared
        terms (basis- and convention-independent: it is quadratic in Ω).

        For a vector field, (Ω_μν)^a_b = R_μν{}^a{}_b, so
            tr(Ω²) = R_μν{}^a{}_b R^μν{}^b{}_a = -R_μνρσ R^μνρσ,
        giving the bundle contribution -(1/12) to the Riem² coefficient
        and the spin-1 total 1/180·d - 1/12 = -11/180 in d=4
        [Christensen & Duff, Nucl. Phys. B 154 (1979) 301;
         Vassilevich, Phys. Rept. 388 (2003) 279].

        For rank-2 symmetric tensors Sym²(V), the Dynkin index scales as
        T(Sym²V) = (d+2) T(V), so tr(Ω²)|_Sym² = -(d+2) R²_μνρσ and the
        bundle contribution is -(d+2)/12 (= -1/2 in d=4). The same value
        applies to the traceless part (the trace part is a scalar with
        Ω = 0). Note that for the *constrained* TT bundle this generic
        master-formula path is not used — see _b4_TT_constrained.

        For a scalar field, Ω = 0 (trivial bundle).
        """
        d = self.d

        if field_type == "scalar":
            return {"Riem2": sympy.S.Zero}
        elif field_type in ("vector", "ghost_vector"):
            # (1/12) × tr(Ω²) = -(1/12) R²_μνρσ
            return {"Riem2": -Rational(1, 12)}
        elif field_type in ("symmetric_tensor", "TT_tensor"):
            # (1/12) × tr(Ω²)|_Sym² = -(d+2)/12 R²_μνρσ
            return {"Riem2": -Rational(d + 2, 12)}
        return {"Riem2": sympy.S.Zero}

    def _extract_R2_coefficient(self, expr: Expr) -> Expr:
        """Extract the coefficient of R̄² from an expression polynomial in R̄."""
        R = self.R_bar
        if expr == sympy.S.Zero:
            return sympy.S.Zero
        # Expand and collect R² terms
        expanded = sympy.expand(expr)
        return expanded.coeff(R, 2)
