"""Canonical mass dimensions of fields and couplings.

In d spacetime dimensions, the canonical mass dimensions are:
    [g_μν] = 0          (metric, dimensionless)
    [h_μν] = (d-2)/2    (graviton fluctuation)
    [G]    = 2-d         (Newton's constant)
    [Λ]   = 2            (cosmological constant)
    [R]    = 2            (Ricci scalar)
    [α]   = d-4          (R^2 coupling)
    [β]   = d-4          (C^2 coupling)
    [φ]   = (d-2)/2      (scalar field)
    [A_μ] = (d-2)/2      (gauge field)
    [ξ]   = 0            (non-minimal coupling)
"""

from sympy import Rational, Symbol


def newton_coupling_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of Newton's constant G: [G] = 2 - d."""
    return 2 - d


def cosmological_constant_dim() -> int:
    """Mass dimension of cosmological constant: [Λ] = 2."""
    return 2


def r_squared_coupling_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of R^2 or C^2 coupling: [α] = d - 4."""
    return d - 4


def scalar_field_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of scalar field: [φ] = (d-2)/2."""
    if isinstance(d, int):
        return Rational(d - 2, 2)
    return (d - 2) / 2


def gauge_field_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of gauge field: [A_μ] = (d-2)/2."""
    return scalar_field_dim(d)


def quartic_coupling_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of scalar quartic coupling: [λ_φ] = 4 - d."""
    return 4 - d


def yukawa_coupling_dim(d: int | Symbol = 4) -> Rational | Symbol:
    """Mass dimension of Yukawa coupling: [y] = (4 - d)/2."""
    if isinstance(d, int):
        return Rational(4 - d, 2)
    return (4 - d) / 2


def nonminimal_coupling_dim() -> int:
    """Mass dimension of non-minimal coupling: [ξ] = 0."""
    return 0
