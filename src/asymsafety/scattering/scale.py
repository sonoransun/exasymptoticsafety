"""Momentum-scale identification: momentum² p² → RG scale k.

This is the scattering analogue of
:mod:`asymsafety.cosmology.scale_identification`.  There, a *position*
scale (radius r, cosmic time t) is mapped to the RG cutoff k so that the
running couplings can be promoted to position-dependent quantities.  Here
we map a *momentum* scale to k so that the couplings become
momentum-dependent form factors of a scattering amplitude:

    G(p²) = g(k(p²)) / k(p²)² ,   Λ(p²) = λ(k(p²)) · k(p²)² .

The canonical identification ties the cutoff to the momentum transfer or
centre-of-mass energy,

    k(p²) = ξ · sqrt(|p²|) ,

with ``ξ`` a dimensionless O(1) matching coefficient.  Different
invariants (s, t, u) feed the *same* map; which one is used is decided by
the amplitude, per scattering channel.

.. warning::
    This is RG-improvement at the level of an observable — the same
    epistemic status as the RG-improved black holes in
    :mod:`asymsafety.cosmology`.  It is **not** a first-principles
    momentum-dependent form factor extracted from the effective action.
    See Knorr (2026) [arXiv:2602.21285] for why naive RG improvement can
    fail quantitatively and why a safe fixed point does not by itself
    guarantee a bounded amplitude.

References:
    Draper, Knorr, Ripken & Saueressig (2020), PRL 125, 181301
        [arXiv:2007.04396]
    Knorr, Ripken & Saueressig (2023), "Form Factors in Asymptotically
        Safe Quantum Gravity" [arXiv:2210.16072]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class MomentumScale(ABC):
    """Abstract base: map a momentum-squared ``p2`` to an RG scale ``k``.

    Subclasses implement :meth:`k_of_psq`.  ``p2`` is a Mandelstam
    invariant (s, t or u) in units of the reference scale ``k0`` squared.
    """

    @abstractmethod
    def k_of_psq(self, p2: float | np.ndarray) -> float | np.ndarray:
        """RG scale ``k`` at momentum-squared ``p2``."""


@dataclass
class EnergyScale(MomentumScale):
    """``k(p²) = ξ · sqrt(|p²|)`` — identify the cutoff with the momentum.

    The default and physically standard choice.  Applied to the
    centre-of-mass energy ``s`` it gives ``k = ξ√s``; applied to the
    momentum transfer ``|t|`` it gives ``k = ξ√|t|``.  The absolute value
    keeps the map well defined for spacelike (negative) invariants.
    """

    xi: float = 1.0

    def k_of_psq(self, p2):
        return self.xi * np.sqrt(np.abs(np.asarray(p2, dtype=float)))


@dataclass
class TransferScale(EnergyScale):
    """Alias of :class:`EnergyScale` documenting a momentum-transfer match.

    Mathematically identical to :class:`EnergyScale`; provided so callers
    can make the intended channel (``k = ξ√|t|``) explicit at the call
    site.
    """


@dataclass
class FixedScale(MomentumScale):
    """``k(p²) = k_fixed`` — a constant cutoff (no running).

    Using this scale freezes every coupling at its value at ``k_fixed``,
    which reproduces the *undressed* (classical-GR) amplitude.  It is the
    reference against which the RG-improved amplitude is compared.
    """

    k_fixed: float = 1.0

    def k_of_psq(self, p2):
        return self.k_fixed * np.ones_like(np.asarray(p2, dtype=float))
