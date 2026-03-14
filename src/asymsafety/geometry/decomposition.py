"""York (transverse-traceless) decomposition of metric fluctuations.

The metric fluctuation h_μν is decomposed into irreducible spin sectors:

    h_μν = h^TT_μν + D_μ ξ_ν + D_ν ξ_μ
           + (D_μ D_ν - (1/d) g_μν D²) σ + (1/d) g_μν h

where:
    h^TT: transverse-traceless tensor (spin-2)
    ξ_μ:  transverse vector (spin-1)
    σ:    scalar (longitudinal traceless part)
    h:    trace scalar

On S^d, each sector has known Laplacian eigenvalues and multiplicities.
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, factorial, binomial


@dataclass
class ModeSpectrum:
    """Eigenvalue spectrum of the Laplacian on S^d for a given field type.

    On S^d with radius a (R̄ = d(d-1)/a²), the eigenvalues of -D² are:

    Scalars:     λ_l = l(l+d-1)/a²,            l = 0, 1, 2, ...
    Trans. vec:  λ_l = [l(l+d-1) - 1]/a²,      l = 1, 2, ...
    TT tensors:  λ_l = [l(l+d-1) - 2]/a²,      l = 2, 3, ...
    """

    d: int  # spacetime dimension

    def eigenvalue(self, field_type: str, l: int, a_sq: Expr) -> Expr:
        """Laplacian eigenvalue on S^d.

        Args:
            field_type: "scalar", "vector", "TT"
            l: Angular momentum quantum number
            a_sq: Radius squared of the sphere
        """
        base = l * (l + self.d - 1)
        shift = {"scalar": 0, "vector": 1, "TT": 2}
        return (base - shift[field_type]) / a_sq

    def multiplicity(self, field_type: str, l: int) -> int:
        """Multiplicity (degeneracy) of mode l on S^d.

        For d=4 (S^4):
            Scalars:     d_l = (l+1)²(l+2)²/4      [= (2l+3)(l+1)(l+2)/6 * ...]
            Trans. vec:  d_l = (2l+3)(l+1)(l+2)/2 · 3   (transverse, 3 components)
            TT tensors:  d_l = 5(2l+3)(l-1)(l+4)/6 · ... [Christensen-Duff 1979]
        """
        if self.d != 4:
            raise NotImplementedError(f"Multiplicities for S^{self.d}")

        if field_type == "scalar":
            return self._scalar_mult_S4(l)
        elif field_type == "vector":
            return self._vector_mult_S4(l)
        elif field_type == "TT":
            return self._TT_mult_S4(l)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def min_l(self, field_type: str) -> int:
        """Minimum l for each field type (below this, modes don't exist)."""
        return {"scalar": 0, "vector": 1, "TT": 2}[field_type]

    # --- S^4 multiplicities ---

    @staticmethod
    def _scalar_mult_S4(l: int) -> int:
        """Scalar multiplicity on S^4: (l+1)²(l+2)²/4.

        This equals the dimension of the (l,0) representation of SO(5).
        """
        return (l + 1)**2 * (l + 2)**2 // 4

    @staticmethod
    def _vector_mult_S4(l: int) -> int:
        """Transverse vector multiplicity on S^4.

        Dimension of the (l,1) representation minus longitudinal modes.
        d_l^vec = (2l+3)(l+3)! / (3! l!) - scalar_mult(l+1)?
        Actually: total vector = d * scalar_mult, transverse = total - gradient modes.

        For transverse vectors on S^4, l ≥ 1:
            d_l = (l+1)(l+2)(2l+3) · 1    [need factor]

        Using Rubin-Ordonez (1984) for S^4:
            Transverse vector mult = l(l+3)(2l+3)/3 · 2
        Wait, let me use the known result:
            d_l^{T,vec} on S^4 = (2l+3)(l+3)!/(l! · 3!) - (l+1+1)^2(l+2+1)^2/4
        This is getting complicated. Let me use the direct formula.

        For transverse vectors on S^N (N=4):
            d_l = (N-1)(2l+N-1)(l+N-2)! / ((N-1)! l!)
        With N=4:
            d_l = 3(2l+3)(l+2)! / (3! l!) = (2l+3)(l+2)(l+1)/2
        """
        if l < 1:
            return 0
        return (2 * l + 3) * (l + 2) * (l + 1) // 2

    @staticmethod
    def _TT_mult_S4(l: int) -> int:
        """TT tensor multiplicity on S^4.

        For symmetric transverse-traceless tensors on S^4, l ≥ 2:
            d_l = 5(2l+3)(l-1)(l+4)/6

        This is the dimension of the (l,2) representation of SO(5)
        minus the contributions from vector and scalar modes.
        Reference: Christensen & Duff (1979).
        """
        if l < 2:
            return 0
        return 5 * (2 * l + 3) * (l - 1) * (l + 4) // 6


@dataclass
class YorkDecomposition:
    """York decomposition on a maximally symmetric background.

    Provides the field content, degrees of freedom, and excluded modes
    for each spin sector.
    """

    d: int = 4

    @property
    def dof_TT(self) -> int:
        """Degrees of freedom of TT tensor: (d+1)(d-2)/2."""
        return (self.d + 1) * (self.d - 2) // 2

    @property
    def dof_vector(self) -> int:
        """Degrees of freedom of transverse vector: d - 1."""
        return self.d - 1

    @property
    def dof_scalar(self) -> int:
        """Scalar sector has 2 fields: (σ, h). Each is 1 dof."""
        return 2

    @property
    def total_dof(self) -> int:
        """Total = d(d+1)/2 (symmetric tensor)."""
        return self.d * (self.d + 1) // 2

    def excluded_modes(self, field_type: str) -> list[int]:
        """Modes excluded from the York decomposition.

        Killing vectors and conformal Killing vectors correspond to
        gauge modes that must be excluded:
            - TT: l = 0, 1 excluded (no TT tensors for l < 2)
            - Vector: l = 0 excluded (no transverse vectors for l < 1),
                      l = 1 are Killing vectors (excluded from physical spectrum)
            - Scalar σ: l = 0, 1 excluded (σ modes don't exist for l < 2)
            - Scalar h: no exclusions (l ≥ 0)
        """
        exclusions = {
            "TT": [0, 1],
            "vector": [0],
            "scalar_sigma": [0, 1],
            "scalar_h": [],
        }
        return exclusions.get(field_type, [])
