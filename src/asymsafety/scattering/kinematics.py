"""Mandelstam kinematics for 2 → 2 scattering.

For a 2 → 2 process with all external masses equal to ``m`` the Mandelstam
invariants satisfy the closure relation

    s + t + u = 4 m² .

In the centre-of-mass frame, with CM three-momentum ``p`` and scattering
angle ``θ``,

    s = 4 (p² + m²) ,
    t = -2 p² (1 - cosθ) ,
    u = -2 p² (1 + cosθ) ,

so that ``t`` and ``u`` are non-positive in the physical region
``s ≥ 4 m²``, ``-1 ≤ cosθ ≤ 1`` (massless: ``s ≥ 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mandelstam:
    """Mandelstam invariants ``(s, t, u)`` for equal external mass ``m``.

    Construct directly, or via :meth:`from_s_t` / :meth:`from_s_angle`,
    which enforce the closure relation ``s + t + u = 4 m²``.
    """

    s: float
    t: float
    u: float
    m: float = 0.0

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_s_t(cls, s: float, t: float, m: float = 0.0) -> "Mandelstam":
        """Build from ``(s, t)``; ``u`` is fixed by closure."""
        u = 4.0 * m**2 - s - t
        return cls(s=float(s), t=float(t), u=float(u), m=float(m))

    @classmethod
    def from_s_angle(
        cls, s: float, cos_theta: float, m: float = 0.0
    ) -> "Mandelstam":
        """Build from CM energy ``s`` and scattering angle ``cosθ``."""
        p_sq = s / 4.0 - m**2
        t = -2.0 * p_sq * (1.0 - cos_theta)
        u = -2.0 * p_sq * (1.0 + cos_theta)
        return cls(s=float(s), t=float(t), u=float(u), m=float(m))

    # -- derived quantities ------------------------------------------------

    @property
    def cm_momentum_sq(self) -> float:
        """Squared CM three-momentum ``p² = s/4 - m²``."""
        return self.s / 4.0 - self.m**2

    @property
    def cos_theta(self) -> float:
        """Scattering angle cosine recovered from ``t``."""
        p_sq = self.cm_momentum_sq
        if p_sq <= 0.0:
            return float("nan")
        return 1.0 + self.t / (2.0 * p_sq)

    @property
    def closure_residual(self) -> float:
        """``s + t + u - 4 m²`` — should be ~0 for a valid point."""
        return self.s + self.t + self.u - 4.0 * self.m**2

    @property
    def is_physical(self) -> bool:
        """True in the s-channel physical region (``s ≥ 4m²``, ``t,u ≤ 0``)."""
        return (
            self.s >= 4.0 * self.m**2 - 1e-12
            and self.t <= 1e-12
            and self.u <= 1e-12
            and abs(self.closure_residual) < 1e-9 * max(1.0, abs(self.s))
        )

    def crossed_s_t(self) -> "Mandelstam":
        """Return the ``s ↔ t`` crossed point (identical-particle crossing)."""
        return Mandelstam(s=self.t, t=self.s, u=self.u, m=self.m)


def cm_momentum(s: float, m: float = 0.0) -> float:
    """CM three-momentum magnitude ``p = sqrt(s/4 - m²)``."""
    return float(np.sqrt(max(s / 4.0 - m**2, 0.0)))


def cos_theta_of_t(s: float, t: float, m: float = 0.0) -> float:
    """Scattering-angle cosine from ``(s, t)``."""
    p_sq = s / 4.0 - m**2
    return float(1.0 + t / (2.0 * p_sq))
