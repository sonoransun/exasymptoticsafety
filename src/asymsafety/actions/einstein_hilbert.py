"""Einstein-Hilbert truncation: Γ_k = (1/16πG) ∫d^dx √g (R - 2Λ).

Provides the Hessian (second functional derivative) Γ_k^(2) in each
spin sector on a maximally symmetric S^4 background, including gauge
fixing and ghost contributions.

The dimensionless couplings are:
    g = G k^{d-2}       (Newton)
    λ = Λ / k^2         (cosmological constant)

References:
    Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030]
    Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040]
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785]
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.core.coupling import CouplingSet, make_eh_couplings
from asymsafety.core.truncation import Truncation


class EinsteinHilbertTruncation(Truncation):
    """Einstein-Hilbert truncation with G and Λ.

    The action is:
        Γ_k = (1/16πG) ∫ d^dx √g (R - 2Λ) + S_gf + S_ghost

    On a maximally symmetric background with Ricci scalar R̄,
    the Hessian in each spin sector is a function of z = -D²
    (the Laplacian eigenvalue).
    """

    def __init__(self, d: int = 4):
        self.d = d
        self._couplings = make_eh_couplings(d)

    def couplings(self) -> CouplingSet:
        return self._couplings

    def action_symbolic(self, d: int = 4) -> Expr:
        """Lagrangian density: (1/16πG)(R - 2Λ)."""
        G = self._couplings["Newton"].symbol
        Lambda = self._couplings["cosmological"].symbol
        R = Symbol("R")
        return (R - 2 * Lambda) / (16 * pi * G)

    def hessian(self, sector: str, z: Symbol,
                R_bar: Symbol, d: int = 4) -> Expr | sympy.Matrix:
        """Second functional derivative Γ_k^(2) in each spin sector.

        On the S^d background with de Donder (harmonic) gauge fixing
        (Landau-DeWitt gauge limit α → 0).

        Args:
            sector: "TT", "vector", "scalar", "ghost"
            z: Laplacian eigenvalue z = -D²
            R_bar: Background Ricci scalar
            d: Spacetime dimension

        Returns:
            Expression for the inverse propagator in that sector.
        """
        G = Symbol("G", positive=True)
        Lambda = Symbol("Lambda", real=True)
        mu = 1 / (32 * pi * G)  # = 1/(2 · 16πG)

        if sector == "TT":
            return self._hessian_TT(z, R_bar, d, mu, Lambda)
        elif sector == "scalar":
            return self._hessian_scalar(z, R_bar, d, mu, Lambda)
        elif sector == "ghost":
            return self._hessian_ghost(z, R_bar, d)
        elif sector == "vector":
            # In de Donder gauge, the vector sector is pure gauge
            # and exactly cancels the ghost. Return the gauge-fixed form.
            return self._hessian_vector(z, R_bar, d, mu, Lambda)
        else:
            raise ValueError(f"Unknown sector: {sector}")

    def _hessian_TT(self, z: Symbol, R_bar: Symbol,
                    d: int, mu: Expr, Lambda: Expr) -> Expr:
        """TT sector: spin-2 graviton.

        Γ^(2)_TT = (1/32πG) [z - 2Λ + c_TT R̄]

        where c_TT = (d²-3d+4)/(d(d-1)) depends on the dimension.
        In d=4: c_TT = (16-12+4)/(4·3) = 8/12 = 2/3.

        The 1/32πG prefactor comes from 1/(2·16πG) from the second
        variation of √g R.
        """
        c_TT = Rational(d**2 - 3*d + 4, d * (d - 1))
        return mu * (z - 2 * Lambda + c_TT * R_bar)

    def _hessian_scalar(self, z: Symbol, R_bar: Symbol,
                        d: int, mu: Expr, Lambda: Expr) -> sympy.Matrix:
        """Scalar sector: coupled (σ, h) system.

        In the Landau-DeWitt gauge (α → 0), the scalar sector involves
        the trace h and the longitudinal-traceless scalar σ. After gauge
        fixing, the Hessian is a 2×2 matrix.

        For the EH truncation on S^d in the standard decomposition,
        the effective scalar Hessian (after gauge fixing decouples σ) is:

        Γ^(2)_h = -(1/32πG)(d-2)/(4d) [z - 2Λ + c_h R̄]

        where c_h = (d²-4)/(d(d-1)) ... the precise coefficients depend
        on the gauge choice.

        The trace mode h has the wrong-sign kinetic term (conformal factor
        problem). The overall negative sign is physical.
        """
        # Simplified: return the effective scalar sector propagator
        # In the standard single-field approach (after σ is gauged away):
        c_h = Rational(d - 4, d)  # Simplified coefficient
        # The conformal mode has negative kinetic term:
        h_hessian = -mu * Rational(d - 2, 2 * d) * (
            z - 2 * Lambda + Rational(d - 2, d - 1) * R_bar
        )
        return h_hessian

    def _hessian_ghost(self, z: Symbol, R_bar: Symbol, d: int) -> Expr:
        """Ghost sector: Faddeev-Popov ghosts from de Donder gauge.

        The ghost operator is:
            M^μ_ν = -δ^μ_ν D² - R̄^μ_ν = -(z + R̄/d) δ^μ_ν

        on a maximally symmetric background where R̄_μν = (R̄/d) g_μν.
        The ghost contributes with a factor of -2 (Grassmann fields, Tr_ghost).
        """
        return -(z + R_bar / d)

    def _hessian_vector(self, z: Symbol, R_bar: Symbol,
                        d: int, mu: Expr, Lambda: Expr) -> Expr:
        """Vector sector (gauge-fixed, transverse).

        In de Donder gauge, the transverse vector mode ξ_μ gets
        a mass from gauge fixing. In the Landau gauge limit (α → 0),
        this sector completely cancels against the ghost contribution.
        """
        return mu * (z - 2 * Lambda + Rational(d + 1, d) * R_bar)


def eh_dimensionless_hessian_TT(z: Symbol, R_bar: Symbol,
                                 g: Symbol, lam: Symbol,
                                 k: Symbol, d: int = 4) -> Expr:
    """TT Hessian in dimensionless variables.

    Returns the TT sector of Γ^(2)/(k^2) in terms of dimensionless
    couplings g and λ.

    z and R̄ should be in units of k^2 (dimensionless).
    """
    c_TT = Rational(d**2 - 3*d + 4, d * (d - 1))
    return (z - 2 * lam + c_TT * R_bar) / (32 * pi * g)
