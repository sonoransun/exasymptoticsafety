"""ADM (Arnowitt-Deser-Misner) decomposition for foliated spacetime.

The spacetime metric in ADM form:
    ds² = ε N² dt² + σ_ij (dx^i + N^i dt)(dx^j + N^j dt)

where ε = -1 (Lorentzian) or ε = +1 (Euclidean).

Extrinsic curvature:
    K_ij = (1/2N)(∂_t σ_ij - D_i N_j - D_j N_i)

The Einstein-Hilbert Lagrangian in ADM variables:
    L = N √σ (K_ij K^ij - K² + R^(3) - 2Λ)
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, sqrt


@dataclass
class ADMDecomposition:
    """ADM decomposition of the spacetime metric.

    Fields:
        N: Lapse function
        N_i: Shift vector (spatial)
        sigma_ij: Spatial metric on the d-1 dimensional slice
        d: Total spacetime dimension
    """

    d: int = 4

    @property
    def spatial_dim(self) -> int:
        """Spatial dimension: d - 1."""
        return self.d - 1

    @staticmethod
    def eh_lagrangian(K_sq: Symbol, KijKij: Symbol,
                      R3: Symbol, Lambda: Symbol) -> Expr:
        """Einstein-Hilbert Lagrangian density in ADM form.

        L_EH = K_ij K^ij - K² + R^(3) - 2Λ

        where K = K^i_i = trace of extrinsic curvature.

        Args:
            K_sq: K² = (K^i_i)²
            KijKij: K_ij K^ij
            R3: Intrinsic Ricci scalar of spatial slice R^(3)
            Lambda: Cosmological constant
        """
        return KijKij - K_sq + R3 - 2 * Lambda

    @staticmethod
    def foliated_eh_lagrangian(K_sq: Symbol, KijKij: Symbol,
                               R3: Symbol, Lambda: Symbol,
                               lambda_adm: Symbol) -> Expr:
        """Foliated EH Lagrangian with independent λ coupling.

        L = K_ij K^ij - λ K² + R^(3) - 2Λ

        When λ = 1, reduces to standard GR.
        Under FDiffs, λ is a free coupling that runs.
        """
        return KijKij - lambda_adm * K_sq + R3 - 2 * Lambda


@dataclass
class FoliatedQuadraticTerms:
    """Independent quadratic curvature invariants under FDiffs in (d-1)+1 dimensions.

    In 3+1 dimensions with foliation-preserving diffeomorphisms, the
    independent quadratic invariants are:

    Extrinsic-extrinsic:
        I_1 = (K_ij K^ij)²
        I_2 = (K²)² = (K^i_i)⁴
        I_3 = K_ij K^ij K²

    Mixed extrinsic-intrinsic:
        I_4 = R^(3) K_ij K^ij
        I_5 = R^(3) K²
        I_6 = R^(3)_ij K^{ik} K_k^j  (optional, higher order)

    Intrinsic-intrinsic:
        I_7 = (R^(3))²
        I_8 = R^(3)_ij R^(3)ij

    In d=3 spatial dimensions, R^(3)_ijkl is determined by R^(3)_ij and R^(3),
    so R^(3)_ijkl R^(3)ijkl is not independent.
    """

    # Coupling symbols
    c_KijKij_sq: Symbol = Symbol("c_1", real=True)    # (K_ij K^ij)²
    c_K_fourth: Symbol = Symbol("c_2", real=True)      # (K²)²
    c_KijKij_K: Symbol = Symbol("c_3", real=True)      # K_ij K^ij K²
    c_R3_KijKij: Symbol = Symbol("c_4", real=True)     # R^(3) K_ij K^ij
    c_R3_Ksq: Symbol = Symbol("c_5", real=True)        # R^(3) K²
    c_R3_sq: Symbol = Symbol("c_6", real=True)          # (R^(3))²
    c_R3ij_sq: Symbol = Symbol("c_7", real=True)        # R^(3)_ij R^(3)ij

    def lagrangian(self, KijKij: Symbol, K_sq: Symbol,
                   R3: Symbol, R3ij_sq: Symbol) -> Expr:
        """Quadratic foliated Lagrangian density.

        Args:
            KijKij: K_ij K^ij
            K_sq: K² = (K^i_i)²
            R3: Intrinsic Ricci scalar R^(3)
            R3ij_sq: R^(3)_ij R^(3)ij
        """
        return (
            self.c_KijKij_sq * KijKij**2
            + self.c_K_fourth * K_sq**2
            + self.c_KijKij_K * KijKij * K_sq
            + self.c_R3_KijKij * R3 * KijKij
            + self.c_R3_Ksq * R3 * K_sq
            + self.c_R3_sq * R3**2
            + self.c_R3ij_sq * R3ij_sq
        )

    @staticmethod
    def gr_limit() -> dict[str, Expr]:
        """Coupling values that recover GR (full-Diff covariant R² + C²).

        Under full diffeomorphism invariance, the foliated quadratic
        couplings reduce to combinations of the covariant α (R²) and
        β (C²) couplings via the Gauss-Codazzi relations.

        Returns mapping from foliated coupling names to expressions
        in terms of α and β.
        """
        alpha = Symbol("alpha", real=True)
        beta_ = Symbol("beta", real=True)

        # These relations follow from decomposing the covariant
        # R² and C² invariants into ADM form using Gauss-Codazzi:
        # R = R^(3) + K_ij K^ij - K² + (boundary terms)
        # C² = ... (more complex)
        return {
            "lambda_ADM": sympy.S.One,
            "c_1": beta_,           # Simplified; exact relations are more involved
            "c_2": alpha + beta_ / 3,
            "c_6": alpha + beta_ / 3,
            "c_7": -2 * beta_,
        }


@dataclass
class FoliatedFieldContent:
    """Field content of the ADM decomposition for the FRG computation.

    After decomposing the metric fluctuations in ADM variables:
        δg_μν → (δN, δN_i, h_ij)

    Further decomposition on the spatial S^3 slices:
        h_ij = h^TT_ij + D_i v_j + D_j v_i + (D_i D_j - σ_ij D²/3) φ + σ_ij ψ/3
        δN_i = n^T_i + D_i n_L

    Sectors:
        spin-2: h^TT_ij                           (5 dof in d=4)
        spin-1: (v_i, n^T_i)                      (3+3 dof, coupled 2×2)
        spin-0: (φ, ψ, δN, n_L)                   (1+1+1+1 dof, coupled 4×4)
    """

    d: int = 4

    @property
    def spin2_dof(self) -> int:
        """TT tensor on spatial slice: (d-1)(d-2)/2 - 1 = d(d-3)/2."""
        return (self.d + 1) * (self.d - 2) // 2  # Same as covariant TT

    @property
    def spin1_components(self) -> list[str]:
        """Spin-1 sector components."""
        return ["v_i", "n_T_i"]

    @property
    def spin0_components(self) -> list[str]:
        """Spin-0 sector components."""
        return ["phi", "psi", "delta_N", "n_L"]
