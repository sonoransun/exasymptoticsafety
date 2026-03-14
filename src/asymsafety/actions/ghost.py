"""Faddeev-Popov ghost action for gravitational gauge fixing.

The ghost action arises from the gauge-fixing procedure:
    S_ghost = ∫d^dx √ḡ c̄_μ M^μ_ν c^ν

where M^μ_ν is the Faddeev-Popov operator:
    M^μ_ν = -δ^μ_ν D̄² - R̄^μ_ν + (1+ρ)/d (D̄^μ D̄_ν + R̄^μ_ν)

On a maximally symmetric background:
    M^μ_ν = -(z + R̄/d) δ^μ_ν   (for harmonic gauge, ρ = (d-2)/2)

The ghost contributes to the trace with a factor of -2 (Grassmann field).
"""

import sympy
from sympy import Expr, Rational, Symbol


def ghost_operator(z: Symbol, R_bar: Symbol, d: int = 4,
                   rho: Expr | None = None) -> Expr:
    """Faddeev-Popov ghost operator on a maximally symmetric background.

    Args:
        z: Laplacian eigenvalue z = -D̄².
        R_bar: Background Ricci scalar.
        d: Spacetime dimension.
        rho: Gauge weight parameter (default: harmonic gauge).

    Returns:
        The ghost inverse propagator (per component).
    """
    if rho is None:
        rho = Rational(d - 2, 2)

    # On max. sym. background, R̄_μν = (R̄/d) g_μν
    # M_μ^ν = -δ_μ^ν z - R̄/d δ_μ^ν + (1+ρ)/d (z δ_μ^ν + R̄/d δ_μ^ν)
    # For the trace mode: M = -z - R̄/d + (1+ρ)/d (z + R̄/d)
    # = -z(1 - (1+ρ)/d) - R̄/d(1 - (1+ρ)/d)
    # = -(1 - (1+ρ)/d)(z + R̄/d)

    # For harmonic gauge ρ = (d-2)/2:
    # (1+ρ)/d = (1+(d-2)/2)/d = (d/2)/d = 1/2
    # M = -(1-1/2)(z + R̄/d) = -(1/2)(z + R̄/d)

    # The standard result for the ghost operator eigenvalue:
    coeff = 1 - (1 + rho) / d
    return -coeff * (z + R_bar / d)


def ghost_trace_contribution(z: Symbol, R_bar: Symbol,
                              d: int = 4) -> dict[str, Expr]:
    """Ghost contribution to the Wetterich equation trace.

    The ghost has:
    - d components (vector field)
    - Factor of -2 (Grassmann: Tr_ghost = -2 × standard trace)
    - The eigenvalues of -D̄² on transverse vectors on S^d

    Returns:
        {"multiplicity": d, "grassmann_factor": -2, "operator": M}
    """
    M = ghost_operator(z, R_bar, d)
    return {
        "multiplicity": sympy.Integer(d),
        "grassmann_factor": sympy.Integer(-2),
        "operator": M,
    }
