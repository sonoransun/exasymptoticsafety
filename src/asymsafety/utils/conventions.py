"""Sign conventions and physical constants used throughout the package.

Conventions follow Reuter & Saueressig (2012) unless noted otherwise.

Metric signature:
    Lorentzian: (-,+,+,+)
    Euclidean:  (+,+,+,+)

Riemann tensor:
    R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}

Ricci tensor (first-and-third contraction):
    R_{bd} = R^a_{bad}

Cosmological constant sign:
    S_EH = (1/16πG) ∫ d^d x √g (R - 2Λ)
    Λ > 0 corresponds to de Sitter space.

RG time:
    t = log(k/k_0)
    t → +∞ is the UV limit.

Dimensionless couplings:
    g = G k^{d-2}       (Newton coupling)
    λ = Λ / k^2         (cosmological constant)
    α, β                (R^2 and C^2 couplings, dimensionless in d=4)

Anomalous dimension:
    η_N = +∂_t ln(G) = -∂_t ln(Z_N)      (G ∝ 1/Z_N)
    Consistent with β_g = (d - 2 + η_N) g for g = G k^{d-2};
    at the NGFP η_N = -(d-2), i.e. G ~ k^{-(d-2)}.
"""

from sympy import Rational

# Default spacetime dimension
DEFAULT_DIMENSION = 4

# Conformal coupling in d dimensions: ξ_conf = (d-2)/(4(d-1))
def conformal_coupling(d: int = 4) -> Rational:
    """Conformal scalar coupling ξ = (d-2)/(4(d-1))."""
    return Rational(d - 2, 4 * (d - 1))
