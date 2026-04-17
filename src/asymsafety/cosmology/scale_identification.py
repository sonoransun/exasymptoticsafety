"""Scale identification: physical scales x → RG scale k.

In RG-improved gravity the dimensionful Newton constant G(k) and
cosmological constant Λ(k) inherit a position dependence by identifying
the cutoff k with a characteristic scale of the classical solution.
Different choices give different physics; the canonical ones are listed
below.

Conventions:
    - ``xi`` is a dimensionless O(1) coefficient capturing the matching
      ambiguity (typically xi ~ 1).
    - ``k0`` is a reference scale at which the dimensionless couplings
      are normalized; usually the Planck scale.

References:
    Bonanno & Reuter (2000), Phys. Rev. D 62, 043008
    Reuter & Saueressig (2012), New J. Phys. 14, 055022
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ScaleIdentification(ABC):
    """Abstract base: maps a physical scale to an RG scale ``k``.

    Static spherically symmetric problems use ``k_of_r``; cosmological
    problems use ``k_of_t``. Subclasses may implement either or both.
    """

    @abstractmethod
    def k_of_r(self, r: float | np.ndarray) -> float | np.ndarray:
        """RG scale at radius ``r``."""

    def k_of_t(self, t: float | np.ndarray,
                hubble: float | np.ndarray | None = None,
                ) -> float | np.ndarray:
        """RG scale at cosmic time ``t`` (default: 1/t identification).

        Subclasses may override to use Hubble-rate matching or other
        prescriptions. ``hubble`` is supplied by RG-improved FLRW solvers
        when needed.
        """
        del hubble  # default: ignore
        return 1.0 / np.asarray(t)


@dataclass
class InverseDistanceScale(ScaleIdentification):
    """``k(r) = xi / r`` — the simplest position-scale matching.

    Diverges at the origin, which is exactly the regime where
    asymptotic safety is supposed to regularize the geometry.
    """

    xi: float = 1.0

    def k_of_r(self, r):
        return self.xi / np.asarray(r)


@dataclass
class GeodesicDistanceScale(ScaleIdentification):
    """``k(r) = xi / sqrt(r² + delta²)`` — Bonanno-Reuter softening.

    The ``delta`` parameter avoids the divergence at ``r = 0``. Setting
    ``delta = 0`` reduces to ``InverseDistanceScale``. A common choice
    is ``delta = sqrt(omega * G_N) ~ Planck length``.
    """

    xi: float = 1.0
    delta: float = 0.0

    def k_of_r(self, r):
        r_arr = np.asarray(r, dtype=float)
        return self.xi / np.sqrt(r_arr**2 + self.delta**2)


@dataclass
class ProperDistanceScale(ScaleIdentification):
    """``k(r) = xi / d(r)`` where ``d(r)`` is the proper distance from a horizon.

    For Schwarzschild, ``d(r) ≈ ∫(1 - 2GM/r')^{-1/2} dr'`` close to the
    horizon. For practical use we approximate ``d(r) ≈ r - r_h`` outside
    the horizon (regular as r → r_h⁺ if augmented with Bonanno-Reuter
    softening).
    """

    xi: float = 1.0
    r_h: float = 0.0
    delta: float = 0.0

    def k_of_r(self, r):
        r_arr = np.asarray(r, dtype=float)
        d = np.maximum(r_arr - self.r_h, 0.0)
        return self.xi / np.sqrt(d**2 + self.delta**2)


@dataclass
class HubbleScale(ScaleIdentification):
    """``k(t) = xi * H(t)`` for cosmological matching.

    ``k_of_r`` raises NotImplementedError; use one of the radial scales
    above for static problems. ``k_of_t`` requires the Hubble rate H(t)
    to be supplied by the FLRW solver via the ``hubble`` kwarg.
    """

    xi: float = 1.0

    def k_of_r(self, r):  # noqa: D401
        raise NotImplementedError(
            "HubbleScale is for cosmology; use a radial scale for BHs."
        )

    def k_of_t(self, t, hubble=None):
        if hubble is None:
            raise ValueError(
                "HubbleScale.k_of_t requires the Hubble rate as `hubble=`."
            )
        return self.xi * np.asarray(hubble)
