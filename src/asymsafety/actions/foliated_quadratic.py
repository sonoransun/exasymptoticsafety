"""Quadratic foliated truncation with independent FDiff couplings.

Under foliation-preserving diffeomorphisms (FDiffs), the quadratic
curvature invariants decompose into more independent structures than
in the fully covariant case.

In 3+1 dimensions, the independent quadratic invariants are:

Extrinsic-extrinsic:
    (K_ij K^ij)², (K²)², K_ij K^ij K²

Mixed:
    R^(3) K_ij K^ij, R^(3) K²

Intrinsic-intrinsic:
    (R^(3))², R^(3)_ij R^(3)ij

Under full diffeomorphism invariance, these reduce to the two
covariant invariants R² and C² via the Gauss-Codazzi relations.
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.core.coupling import CouplingSet, Coupling
from asymsafety.core.truncation import Truncation
from asymsafety.geometry.adm import FoliatedQuadraticTerms


def make_foliated_quadratic_couplings(d: int = 4) -> CouplingSet:
    """Create the full foliated quadratic coupling set."""
    cs = CouplingSet()

    # EH sector
    cs.add(Coupling("Newton", Symbol("G", positive=True), 2 - d,
                     Symbol("g", positive=True)))
    cs.add(Coupling("cosmological", Symbol("Lambda", real=True), 2,
                     Symbol("lambda", real=True)))
    cs.add(Coupling("lambda_ADM", Symbol("lambda_ADM", real=True), 0,
                     Symbol("lambda_ADM", real=True)))

    # Quadratic extrinsic curvature couplings (dimensionless in d=4)
    cs.add(Coupling("c_KK", Symbol("c_KK", real=True), d - 4,
                     Symbol("c_KK", real=True)))
    cs.add(Coupling("c_K2", Symbol("c_K2", real=True), d - 4,
                     Symbol("c_K2", real=True)))
    cs.add(Coupling("c_KK_K2", Symbol("c_KK_K2", real=True), d - 4,
                     Symbol("c_KK_K2", real=True)))

    # Mixed couplings
    cs.add(Coupling("c_RK", Symbol("c_RK", real=True), d - 4,
                     Symbol("c_RK", real=True)))
    cs.add(Coupling("c_RK2", Symbol("c_RK2", real=True), d - 4,
                     Symbol("c_RK2", real=True)))

    # Spatial curvature-squared couplings
    cs.add(Coupling("c_R2", Symbol("c_R2", real=True), d - 4,
                     Symbol("c_R2", real=True)))
    cs.add(Coupling("c_Rij2", Symbol("c_Rij2", real=True), d - 4,
                     Symbol("c_Rij2", real=True)))

    return cs


class FoliatedQuadraticTruncation(Truncation):
    """Full foliated quadratic truncation.

    Extends the foliated EH truncation with all independent
    quadratic curvature invariants under FDiffs.
    """

    def __init__(self, d: int = 4):
        self.d = d
        self._couplings = make_foliated_quadratic_couplings(d)

    def couplings(self) -> CouplingSet:
        return self._couplings

    def action_symbolic(self, d: int = 4) -> Expr:
        """Full foliated quadratic Lagrangian density."""
        quad = FoliatedQuadraticTerms()
        # Symbols for the ADM invariants
        KijKij = Symbol("KijKij")
        K_sq = Symbol("K_sq")
        R3 = Symbol("R3")
        R3ij_sq = Symbol("R3ij_sq")
        Lambda = Symbol("Lambda", real=True)
        lambda_adm = Symbol("lambda_ADM", real=True)
        G = Symbol("G", positive=True)

        eh_part = (KijKij - lambda_adm * K_sq + R3 - 2 * Lambda) / (16 * pi * G)
        quad_part = quad.lagrangian(KijKij, K_sq, R3, R3ij_sq)

        return eh_part + quad_part

    def hessian(self, sector: str, z: Symbol,
                R_bar: Symbol, d: int = 4) -> Expr | sympy.Matrix:
        """Hessian for the foliated quadratic truncation.

        The quadratic terms add 4th-order contributions to each sector,
        analogous to the covariant quadratic gravity case but with
        more independent structures due to the foliation.
        """
        # Extend the foliated EH Hessian with quadratic corrections
        from asymsafety.actions.foliated import FoliatedEHTruncation

        feh = FoliatedEHTruncation(d)
        base = feh.hessian(sector, z, R_bar, d)

        # Add quadratic corrections (sector-dependent)
        omega = Symbol("omega", real=True)
        c_KK = Symbol("c_KK", real=True)
        c_K2 = Symbol("c_K2", real=True)
        c_R2 = Symbol("c_R2", real=True)
        c_Rij2 = Symbol("c_Rij2", real=True)

        if sector == "TT":
            # The spatial curvature-squared terms c_Rij2 R^(3)_ij²
            # contribute to the TT sector through z² terms
            quad_correction = c_Rij2 * z**2 + c_KK * omega**4
            if isinstance(base, sympy.Expr):
                return base + quad_correction
            return base

        elif sector == "spin0":
            # The c_R2 (R^(3))² and c_K2 (K²)² terms modify the
            # scalar sector, analogous to α R² in the covariant case
            if isinstance(base, sympy.Matrix):
                correction = sympy.Matrix([
                    [c_K2 * omega**4 + c_R2 * z**2, 0],
                    [0, 0],
                ])
                return base + correction
            return base

        return base

    @staticmethod
    def check_gr_limit(couplings: dict[str, float],
                        tol: float = 1e-6) -> bool:
        """Check if foliated couplings are consistent with full-Diff GR.

        In the GR limit:
            λ_ADM = 1
            The quadratic couplings satisfy specific relations from
            Gauss-Codazzi decomposition of covariant R² and C².
        """
        if abs(couplings.get("lambda_ADM", 1.0) - 1.0) > tol:
            return False
        # Additional consistency checks could be added here
        return True
