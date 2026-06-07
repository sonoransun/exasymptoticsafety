"""Momentum-dependent graviton form factor from an RG trajectory.

The dimensionless running couplings ``g(k), λ(k), ξ(k)`` stored in an
:class:`~asymsafety.analysis.flow.RGTrajectory` are promoted to
momentum-dependent dimensionful quantities by evaluating them at the
scale ``k(p²)`` supplied by a :class:`~asymsafety.scattering.scale.MomentumScale`:

    G(p²) = g(k(p²)) / k(p²)² ,
    Λ(p²) = λ(k(p²)) · k(p²)² .

This mirrors exactly the construction used for RG-improved black holes
(:class:`asymsafety.cosmology.rg_improved_bh.RGImprovedSchwarzschild`),
with the radial scale ``k(r)`` replaced by the momentum scale ``k(p²)``.

The **graviton form factor** is the dimensionless dressing of the
propagator coefficient relative to its low-energy (Newtonian) value,

    f(p²) = G_N / G(p²) ,     G_N ≡ G(p² → 0) ,

so that ``f → 1`` in the infrared and ``f`` grows in the ultraviolet.
At the non-Gaussian fixed point ``g(k) → g*`` so ``G(p²) → g*/k² ∝ 1/p²``
and therefore ``f(p²) ∝ p²``: the propagator coefficient
``G(p²) ∝ 1/p²`` softens the graviton exchange, the mechanism behind the
finite ultraviolet amplitude of asymptotic safety.

.. warning::
    RG-improvement model, not a first-principles form factor; see the
    module docstring of :mod:`asymsafety.scattering.scale`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.scattering.scale import EnergyScale, MomentumScale


@dataclass
class GravitonFormFactor:
    """Momentum-dependent ``G(p²)``, ``Λ(p²)``, ``ξ(p²)`` and ``f(p²)``.

    Args:
        trajectory: RG trajectory carrying at least a ``'g'`` coupling
            (and optionally ``'lambda'``, ``'xi'``) versus RG time
            ``t = log(k/k0)``.
        scale: Momentum-scale identification ``k(p²)`` (default
            ``EnergyScale(ξ=1)``, i.e. ``k = √|p²|``).
        k0: Reference scale at which ``g`` is normalised (Planck scale;
            defaults to 1).
        p2_ref: Infrared reference momentum-squared used to define the
            Newton constant ``G_N = G(p2_ref)``.  Should sit in the IR
            part of the trajectory's range.
        g_newton: Optional explicit override for ``G_N``.  If ``None`` it
            is computed from ``p2_ref``.
        xi_default: Value returned by :meth:`xi_of_psq` when the trajectory
            carries no running ``'xi'`` coupling (minimal coupling: 0).
    """

    trajectory: RGTrajectory
    scale: MomentumScale = field(default_factory=lambda: EnergyScale(xi=1.0))
    k0: float = 1.0
    p2_ref: float = 1e-4
    g_newton: float | None = None
    xi_default: float = 0.0

    def __post_init__(self):
        if "g" not in self.trajectory.coupling_names:
            raise ValueError(
                "Trajectory must contain a 'g' coupling for the graviton "
                "form factor."
            )
        if self.g_newton is None:
            self.g_newton = float(np.atleast_1d(self.G_of_psq(self.p2_ref))[0])

    # -- internal ----------------------------------------------------------

    def _t_of_psq(self, p2: np.ndarray) -> np.ndarray:
        """RG time ``t = log(k(p²)/k0)``."""
        k = np.atleast_1d(self.scale.k_of_psq(p2))
        return np.log(np.maximum(k, 1e-300) / self.k0)

    def _coupling_at(self, name: str, p2) -> np.ndarray:
        p2_arr = np.atleast_1d(np.asarray(p2, dtype=float))
        t = self._t_of_psq(p2_arr)
        return np.array([self.trajectory.at_scale(float(ti))[name] for ti in t])

    @staticmethod
    def _scalarize(arr: np.ndarray, p2):
        return float(arr[0]) if np.ndim(p2) == 0 else arr

    # -- public API --------------------------------------------------------

    def g_dimensionless(self, p2) -> float | np.ndarray:
        """Dimensionless ``g(k(p²))`` along the trajectory."""
        return self._scalarize(self._coupling_at("g", p2), p2)

    def G_of_psq(self, p2) -> float | np.ndarray:
        """Dimensionful Newton coupling ``G(p²) = g(k(p²)) / k(p²)²``."""
        p2_arr = np.atleast_1d(np.asarray(p2, dtype=float))
        k = np.atleast_1d(self.scale.k_of_psq(p2_arr))
        g_vals = self._coupling_at("g", p2_arr)
        result = g_vals / np.maximum(k, 1e-300) ** 2
        return self._scalarize(result, p2)

    def Lambda_of_psq(self, p2) -> float | np.ndarray:
        """Dimensionful cosmological constant ``Λ(p²) = λ(k(p²)) · k(p²)²``."""
        if "lambda" not in self.trajectory.coupling_names:
            zeros = np.zeros_like(np.atleast_1d(np.asarray(p2, dtype=float)))
            return self._scalarize(zeros, p2)
        p2_arr = np.atleast_1d(np.asarray(p2, dtype=float))
        k = np.atleast_1d(self.scale.k_of_psq(p2_arr))
        lam_vals = self._coupling_at("lambda", p2_arr)
        return self._scalarize(lam_vals * k**2, p2)

    def xi_of_psq(self, p2) -> float | np.ndarray:
        """Non-minimal scalar coupling ``ξ(k(p²))``.

        Returns :attr:`xi_default` (constant) if the trajectory carries no
        running ``'xi'``.
        """
        if "xi" not in self.trajectory.coupling_names:
            const = self.xi_default * np.ones_like(
                np.atleast_1d(np.asarray(p2, dtype=float))
            )
            return self._scalarize(const, p2)
        return self._scalarize(self._coupling_at("xi", p2), p2)

    def newton_constant(self) -> float:
        """Infrared Newton constant ``G_N = G(p² → 0)``."""
        return float(self.g_newton)

    def f(self, p2) -> float | np.ndarray:
        """Graviton form factor ``f(p²) = G_N / G(p²)`` (``→ 1`` in the IR)."""
        G = np.atleast_1d(np.asarray(self.G_of_psq(p2), dtype=float))
        result = self.g_newton / np.where(np.abs(G) < 1e-300, np.nan, G)
        return self._scalarize(result, p2)
