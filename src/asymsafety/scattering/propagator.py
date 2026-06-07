"""Dressed flat-space graviton propagator coefficient.

In harmonic (de Donder) gauge the flat-space graviton propagator is the
spin-2 projector over the massless pole with strength ``κ²/4 = 8πG``.
Asymptotic safety dresses the Newton coupling, ``G → G(p²)``, via the
:class:`~asymsafety.scattering.form_factor.GravitonFormFactor`.  This
module exposes the resulting momentum-dependent coupling strength and the
(optionally massive, ``Λ``-shifted) propagator denominator; the spin-2
numerator/kinematic factors for external scalars are applied in
:mod:`asymsafety.scattering.amplitude`.

The inverse propagator normalisation matches the TT graviton Hessian of
the Einstein–Hilbert truncation,
``Γ^(2)_TT ∝ (z - 2Λ + c_TT R̄)/(32πG)``
(:func:`asymsafety.actions.einstein_hilbert.eh_dimensionless_hessian_TT`),
continued to flat space ``R̄ → 0`` with ``z → p²``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from asymsafety.scattering.form_factor import GravitonFormFactor


@dataclass
class DressedGravitonPropagator:
    """Running graviton coupling strength and propagator denominator.

    Args:
        form_factor: Source of ``G(p²)``, ``Λ(p²)`` and ``f(p²)``.
        massive: If True the denominator includes the running
            cosmological-constant mass term ``p² - 2Λ(p²)``; otherwise the
            massless pole ``p²`` is used.
    """

    form_factor: GravitonFormFactor
    massive: bool = False

    def G(self, p2) -> float | np.ndarray:
        """Dimensionful Newton coupling ``G(p²)``."""
        return self.form_factor.G_of_psq(p2)

    def dressing(self, p2) -> float | np.ndarray:
        """Form factor ``f(p²) = G_N / G(p²)`` (``→ 1`` in the IR)."""
        return self.form_factor.f(p2)

    def coupling_strength(self, p2) -> float | np.ndarray:
        """Gravitational coupling ``8π G(p²) = κ²/4`` at scale ``p²``."""
        return 8.0 * math.pi * np.asarray(self.G(p2), dtype=float)

    def denominator(self, p2) -> float | np.ndarray:
        """Propagator denominator ``p²`` (or ``p² - 2Λ(p²)`` if massive)."""
        p2_arr = np.asarray(p2, dtype=float)
        if not self.massive:
            return p2_arr
        return p2_arr - 2.0 * np.asarray(self.form_factor.Lambda_of_psq(p2),
                                         dtype=float)
