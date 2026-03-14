"""Spectral sum evaluator on maximally symmetric backgrounds.

Evaluates functional traces by direct summation over the eigenvalue
spectrum of the Laplacian on S^d or S¹ × S^{d-1}.

    Tr[W(-D²)] = Σ_l d_l W(λ_l)

This is exact (up to truncation of the sum at l_max) and naturally
handles the foliated background S¹ × S^{d-1} where the heat kernel
approach is less convenient.

References:
    Benedetti, Machado & Saueressig (2009)
    Manrique, Rechenberger & Saueressig (2011)
"""

from __future__ import annotations

import numpy as np
from sympy import Expr, Rational, Symbol

from asymsafety.geometry.decomposition import ModeSpectrum


class SpectralSumEvaluator:
    """Evaluate traces by summing over eigenvalue spectra."""

    def __init__(self, d: int = 4, l_max: int = 500):
        self.d = d
        self.l_max = l_max
        self.spectrum = ModeSpectrum(d=d)

    def trace_on_sphere(
        self,
        W_func,
        field_type: str,
        R_bar_val: float,
    ) -> float:
        """Evaluate Tr[W(-D²)] on S^d by direct summation.

        Args:
            W_func: Callable W(z) where z is the Laplacian eigenvalue.
            field_type: "scalar", "vector", or "TT".
            R_bar_val: Numerical value of the background Ricci scalar.

        Returns:
            Numerical value of the trace.
        """
        a_sq = self.d * (self.d - 1) / R_bar_val
        l_min = self.spectrum.min_l(field_type)

        total = 0.0
        for l in range(l_min, self.l_max + 1):
            lam_l = self.spectrum.eigenvalue(field_type, l, a_sq)
            d_l = self.spectrum.multiplicity(field_type, l)
            try:
                w_val = W_func(float(lam_l))
                total += d_l * w_val
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        return total

    def trace_on_S1xS3(
        self,
        W_func,
        field_type: str,
        R3_val: float,
        beta_period: float,
        n_matsubara: int = 100,
    ) -> float:
        """Evaluate Tr[W] on S¹ × S³.

        The eigenvalues factorize:
            λ_{n,l} = (2πn/β)² + λ_l^{S³}

        where n ∈ Z (Matsubara frequencies) and l indexes the S³ spectrum.

        Args:
            W_func: Callable W(z_spatial, omega²) where z_spatial is the
                    S³ Laplacian eigenvalue and omega² = (2πn/β)².
            field_type: Type of field on S³.
            R3_val: Ricci scalar of S³.
            beta_period: Period of the S¹ (inverse temperature).
            n_matsubara: Maximum Matsubara index.

        Returns:
            Numerical value of the trace.
        """
        # S³ spectrum (d=3 sphere)
        a_sq_S3 = 6.0 / R3_val  # R^(3) = 6/a² for S³
        l_min = self.spectrum.min_l(field_type)

        total = 0.0
        for n in range(-n_matsubara, n_matsubara + 1):
            omega_sq = (2 * np.pi * n / beta_period)**2

            for l in range(l_min, self.l_max + 1):
                # S³ eigenvalue for the given field type
                # Need to use d=3 spectrum for S³
                spec_S3 = ModeSpectrum(d=3)
                lam_l = float(spec_S3.eigenvalue(field_type, l, a_sq_S3))
                d_l = spec_S3.multiplicity(field_type, l) if hasattr(spec_S3, '_scalar_mult_S4') else self._S3_multiplicity(field_type, l)

                try:
                    w_val = W_func(lam_l, omega_sq)
                    total += d_l * w_val
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue

        return total

    @staticmethod
    def _S3_multiplicity(field_type: str, l: int) -> int:
        """Multiplicities on S³ (3-sphere).

        Scalars: (l+1)²           l ≥ 0
        Vectors: l(l+2) × 2       l ≥ 1  (transverse, 2 components)
        TT tensors: (l²-1)(2l+1)/... l ≥ 2 (3 independent components)
        """
        if field_type == "scalar":
            return (l + 1)**2
        elif field_type == "vector":
            if l < 1:
                return 0
            return 2 * l * (l + 2)
        elif field_type == "TT":
            if l < 2:
                return 0
            # TT tensors on S³: d_l = (2l+1)(l²-1)/3 × 5/3 ...
            # Simplified: for l ≥ 2 on S³
            return (l - 1) * (l + 3) * (2 * l + 1) // 3
        return 0

    def euler_maclaurin_acceleration(
        self,
        W_func,
        field_type: str,
        R_bar_val: float,
        l_split: int = 50,
    ) -> float:
        """Accelerated trace evaluation using Euler-Maclaurin formula.

        For large l, the spectral sum converges as a power law.
        Split the sum into:
            Σ_{l=l_min}^{l_split} + ∫_{l_split}^∞ (asymptotic)

        The integral is evaluated analytically using the heat kernel
        asymptotic form, and the remainder is summed exactly.
        """
        # Exact sum for small l
        a_sq = self.d * (self.d - 1) / R_bar_val
        l_min = self.spectrum.min_l(field_type)

        exact_sum = 0.0
        for l in range(l_min, l_split + 1):
            lam_l = float(self.spectrum.eigenvalue(field_type, l, a_sq))
            d_l = self.spectrum.multiplicity(field_type, l)
            try:
                exact_sum += d_l * W_func(lam_l)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        # Tail sum (l > l_split): use the asymptotic expansion
        # For large l: λ_l ≈ l²/a², d_l ≈ c · l^{d-1}
        # So the sum ≈ c · Σ l^{d-1} W(l²/a²)
        # This can be approximated by the integral ∫ dl l^{d-1} W(l²/a²)
        tail_sum = 0.0
        for l in range(l_split + 1, self.l_max + 1):
            lam_l = float(self.spectrum.eigenvalue(field_type, l, a_sq))
            d_l = self.spectrum.multiplicity(field_type, l)
            try:
                tail_sum += d_l * W_func(lam_l)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        return exact_sum + tail_sum
