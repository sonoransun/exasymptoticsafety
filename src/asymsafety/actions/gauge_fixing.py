"""Gauge fixing for gravitational FRG computations.

De Donder (harmonic) gauge family:
    S_gf = (1/2α) ∫d^dx √ḡ F_μ ḡ^μν F_ν

where the gauge-fixing condition is:
    F_μ = D̄^ν h_μν - (1+ρ)/d D̄_μ h

Parameters:
    α: gauge-fixing parameter (α → 0 is Landau gauge)
    ρ: weight parameter (ρ = (d-2)/2 is harmonic gauge, ρ = -1 is ∂ḡ = 0)

References:
    Reuter (1998), Lauscher & Reuter (2002)
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol


@dataclass
class GaugeParameters:
    """Gauge-fixing parameters.

    Attributes:
        alpha: Gauge-fixing parameter (α → 0 is Landau gauge).
        rho: Weight parameter in the gauge condition.
        alpha_spatial: Spatial gauge parameter (foliated case).
        alpha_temporal: Temporal gauge parameter (foliated case).
    """

    alpha: Symbol = Symbol("alpha_gf", positive=True)
    rho: Symbol = Symbol("rho_gf", real=True)
    alpha_spatial: Symbol | None = None
    alpha_temporal: Symbol | None = None

    @classmethod
    def harmonic(cls, d: int = 4) -> "GaugeParameters":
        """Standard harmonic (de Donder) gauge: ρ = (d-2)/2."""
        return cls(
            alpha=Symbol("alpha_gf", positive=True),
            rho=Rational(d - 2, 2),
        )

    @classmethod
    def landau_dewitt(cls, d: int = 4) -> "GaugeParameters":
        """Landau-DeWitt gauge: α → 0 limit of harmonic gauge."""
        return cls(
            alpha=sympy.S.Zero,
            rho=Rational(d - 2, 2),
        )


def gauge_fixing_hessian_contribution(z: Symbol, R_bar: Symbol,
                                       gauge: GaugeParameters,
                                       sector: str,
                                       d: int = 4) -> Expr:
    """Contribution of gauge-fixing to the Hessian in each sector.

    In the Landau-DeWitt gauge (α → 0), the gauge fixing effectively
    constrains the longitudinal and trace parts of h_μν, leaving
    only the TT sector as the physical propagator.

    For finite α, the gauge fixing adds terms to each sector.
    """
    alpha = gauge.alpha
    rho = gauge.rho

    if sector == "TT":
        # TT sector is gauge-invariant (no gauge-fixing contribution)
        return sympy.S.Zero
    elif sector == "vector":
        # Vector (longitudinal) sector: S_gf contributes a mass-like term
        # proportional to 1/α
        return z / alpha if alpha != 0 else sympy.S.Zero
    elif sector == "scalar":
        # Scalar (trace) sector: mixed contribution from ρ parameter
        if alpha == 0:
            return sympy.S.Zero
        return (1 + rho)**2 * z / (alpha * d**2)
    return sympy.S.Zero
