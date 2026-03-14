"""Foliated Einstein-Hilbert action in ADM variables.

The foliated action with FDiff-independent λ coupling:
    S = (1/16πG) ∫dt d^{d-1}x N√σ (K_ij K^ij - λ K² + R^(3) - 2Λ)

When λ = 1, this is standard GR. Under foliation-preserving
diffeomorphisms (FDiffs), λ is a free coupling.

Background: S¹ × S^{d-1} with period β (time circle) and radius a
(spatial sphere). The background extrinsic curvature vanishes: K̄_ij = 0.

References:
    Manrique, Rechenberger & Saueressig (2011), Phys. Rev. Lett. 106, 251302
    Rechenberger & Saueressig (2013), JHEP 03, 010
    Biemans, Platania & Saueressig (2017), JHEP 05, 093
"""

from dataclasses import dataclass

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.core.coupling import CouplingSet, make_foliated_eh_couplings
from asymsafety.core.truncation import Truncation


class FoliatedEHTruncation(Truncation):
    """Foliated Einstein-Hilbert truncation with (G, Λ, λ_ADM).

    On the background S¹ × S³:
    - The lapse is constant: N̄ = 1
    - The shift vanishes: N̄^i = 0
    - K̄_ij = 0 (static background)
    - R̄^(3) = spatial curvature of S³

    Fluctuations:
    - δN: lapse perturbation (scalar)
    - δN^i = n^T_i + D_i n_L: shift (transverse vector + scalar)
    - h_ij: spatial metric perturbation (TT + vector + scalar)
    """

    def __init__(self, d: int = 4):
        self.d = d
        self._couplings = make_foliated_eh_couplings(d)

    def couplings(self) -> CouplingSet:
        return self._couplings

    def action_symbolic(self, d: int = 4) -> Expr:
        """Foliated EH Lagrangian density."""
        G = Symbol("G", positive=True)
        Lambda = Symbol("Lambda", real=True)
        lambda_adm = Symbol("lambda_ADM", real=True)
        KijKij = Symbol("KijKij")
        K_sq = Symbol("K_sq")
        R3 = Symbol("R3")

        return (KijKij - lambda_adm * K_sq + R3 - 2 * Lambda) / (16 * pi * G)

    def hessian(self, sector: str, z: Symbol,
                R_bar: Symbol, d: int = 4) -> Expr | sympy.Matrix:
        """Foliated Hessian on S¹ × S^{d-1} background.

        The key difference from the covariant formulation:
        - Time and space derivatives are treated asymmetrically
        - The operator depends on both temporal frequency ω and
          spatial Laplacian eigenvalue z_spatial
        - The lapse and shift enter as separate sectors

        Args:
            sector: "TT", "spin1", "spin0", "ghost_spatial", "ghost_temporal"
            z: For foliated, this is the spatial Laplacian eigenvalue.
            R_bar: R^(3), the spatial Ricci scalar.
            d: Total spacetime dimension.

        For the foliated case, the Hessian also depends on the
        temporal frequency ω (Matsubara mode).
        """
        G = Symbol("G", positive=True)
        Lambda = Symbol("Lambda", real=True)
        lambda_adm = Symbol("lambda_ADM", real=True)
        omega = Symbol("omega", real=True)  # temporal frequency
        mu = 1 / (32 * pi * G)
        d_s = d - 1  # spatial dimension

        if sector == "TT":
            return self._hessian_TT(z, omega, R_bar, d_s, mu, Lambda)
        elif sector == "spin1":
            return self._hessian_spin1(z, omega, R_bar, d_s, mu, Lambda, lambda_adm)
        elif sector == "spin0":
            return self._hessian_spin0(z, omega, R_bar, d_s, mu, Lambda, lambda_adm)
        elif sector == "ghost_spatial":
            return self._hessian_ghost_spatial(z, omega, R_bar, d_s)
        elif sector == "ghost_temporal":
            return self._hessian_ghost_temporal(z, omega, R_bar, d_s)
        else:
            raise ValueError(f"Unknown sector: {sector}")

    def _hessian_TT(self, z, omega, R_bar, d_s, mu, Lambda):
        """TT tensor sector of spatial metric fluctuation.

        Γ^(2)_TT = μ (ω² + z - 2Λ + c_TT R^(3))

        where c_TT depends on the spatial curvature coupling.
        """
        c_TT = Rational(d_s**2 - 3*d_s + 4, d_s * (d_s - 1)) if d_s > 1 else 0
        return mu * (omega**2 + z - 2 * Lambda + c_TT * R_bar)

    def _hessian_spin1(self, z, omega, R_bar, d_s, mu, Lambda, lambda_adm):
        """Spin-1 sector: coupled (v_i, n^T_i) system.

        Returns a 2×2 matrix in the (v_i, n^T_i) basis.
        The coupling between v and n^T involves the extrinsic curvature
        kinetic term and depends on λ_ADM.
        """
        # Off-diagonal coupling through the kinetic term
        # For the EH action, the relevant kinetic mixing is ∝ ω
        vv = mu * (omega**2 + z - 2 * Lambda + R_bar / d_s)
        nn = mu * z  # shift perturbation
        vn = mu * omega * sympy.sqrt(z)  # kinetic mixing

        return sympy.Matrix([[vv, vn], [vn, nn]])

    def _hessian_spin0(self, z, omega, R_bar, d_s, mu, Lambda, lambda_adm):
        """Spin-0 sector: coupled (φ, ψ, δN, n_L) system.

        This is the most complex sector, involving:
        - φ: longitudinal-traceless part of h_ij
        - ψ: trace of h_ij (conformal mode)
        - δN: lapse perturbation
        - n_L: longitudinal shift

        The λ_ADM coupling enters the ψ-ψ component through
        the K² term in the action.

        For simplicity, return the effective scalar Hessian after
        gauge-fixing removes n_L and constrains φ.
        """
        # After gauge-fixing in the Landau limit, the effective
        # scalar sector reduces to a 2×2 system in (ψ, δN):
        psi_psi = -mu * Rational(d_s - 1, 2 * d_s) * (
            (1 - lambda_adm) * omega**2 + z - 2 * Lambda
            + Rational(d_s - 2, d_s - 1) * R_bar
        )
        NN = mu * (z - 2 * Lambda + R_bar / d_s)
        psi_N = mu * omega * Rational(d_s - 1, d_s)

        return sympy.Matrix([[psi_psi, psi_N], [psi_N, NN]])

    def _hessian_ghost_spatial(self, z, omega, R_bar, d_s):
        """Spatial ghost sector: d_s-1 transverse vector ghosts."""
        return -(omega**2 + z + R_bar / d_s)

    def _hessian_ghost_temporal(self, z, omega, R_bar, d_s):
        """Temporal ghost sector: scalar ghost from time reparametrization."""
        return -(omega**2 + z)
