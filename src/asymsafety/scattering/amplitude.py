"""Graviton-mediated 2 → 2 scalar scattering amplitude in asymptotic safety.

Tree-level graviton exchange between four scalars sums the s-, t- and
u-channel pole terms.  With the standard four-scalar graviton-exchange
numerators (``κ²/4 = 8πG``)

    M_GR(s,t,u) = -8π G_N [ t u / s + s u / t + s t / u ] ,

asymptotic safety promotes the Newton coupling to a momentum-dependent
form factor *per channel*, ``G_N → G(x) = G_N / f(x)`` with ``x`` the
channel invariant:

    M(s,t,u) = -8π [ G(s) · t u / s + G(t) · s u / t + G(u) · s t / u ] .

**Overall-sign convention.**  The overall sign of ``M`` above is a
convention, and this module's choice is *opposite* to
Draper-Knorr-Ripken-Saueressig, Phys. Rev. Lett. 125, 181301
[2007.04396], who obtain ``M = +8πG_N · tu/s`` per channel in the
mostly-minus signature with ``S = 1 + iT`` (the sign chain that yields
an attractive Newtonian potential from the t-channel Born term).  The
minus sign used here corresponds to the opposite ``S = 1 - iT``
(Weinberg) map; equivalently ``M_here = -M_DKRS``.  Every internal
consumer (the :mod:`~asymsafety.scattering.consistency` checks, the
bootstrap comparison, the plots) uses ``|M|`` or ``|M|²`` and is
insensitive to this choice, but users extracting an interaction
potential or interference terms via the ``S = 1 + iT`` Born map must
flip the overall sign first.

**Infrared:** ``G(x) → G_N`` recovers the classical (Newtonian) amplitude
— in particular the forward ``1/t`` graviton pole.
**Ultraviolet:** at the non-Gaussian fixed point ``G(x) ∝ 1/x``, so each
channel term tends to a constant and the fixed-angle amplitude approaches
a finite ultraviolet value instead of growing without bound.  This
softening is what restores tree-level unitarity at high energy (see
:mod:`asymsafety.scattering.consistency`).

The non-minimal scalar coupling ``ξ`` improves the stress tensor,
``T_{μν} → T_{μν} - ξ(∂_μ∂_ν - η_{μν}□)φ²``; in this model it contributes
a per-channel term ``∝ ξ(x) · x`` to the vertex (enabled with
``include_xi=True``).

.. warning::
    RG-improvement model — see :mod:`asymsafety.scattering.scale`.  The
    GR numerators are taken from the known tree result, not re-derived
    symbolically from the effective action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.scattering.kinematics import Mandelstam

_8PI = 8.0 * math.pi


@dataclass
class GravitonMediatedAmplitude:
    """Tree-level graviton-mediated 2 → 2 scalar amplitude.

    Args:
        form_factor: Momentum-dependent ``G(x)``/``ξ(x)`` source.
        m: External scalar mass (default massless).
        include_xi: Include the non-minimal ``ξ`` vertex improvement.
        xi_coeff: Model coefficient of the ``ξ`` improvement term.
        forward_reg: Small additive regulator on channel denominators to
            tame the exact forward/backward massless poles in numerical
            grids (default 0 — keep the physical pole).
    """

    form_factor: GravitonFormFactor
    m: float = 0.0
    include_xi: bool = False
    xi_coeff: float = 1.0
    forward_reg: float = 0.0

    # -- per-channel pieces ------------------------------------------------

    def _G(self, x, dressed: bool):
        """Newton coupling for channel invariant ``x``."""
        if dressed:
            return np.asarray(self.form_factor.G_of_psq(x), dtype=float)
        return self.form_factor.newton_constant() * np.ones_like(
            np.asarray(x, dtype=float)
        )

    def _channel(self, x, numerator, dressed: bool):
        """``-8π G(x) · numerator / x`` with the ξ improvement term."""
        x_arr = np.asarray(x, dtype=float)
        denom = x_arr + np.sign(x_arr + (x_arr == 0)) * self.forward_reg
        term = numerator / denom
        if self.include_xi:
            xi = np.asarray(self.form_factor.xi_of_psq(x), dtype=float)
            term = term + self.xi_coeff * xi * x_arr
        return -_8PI * self._G(x_arr, dressed) * term

    # -- public API --------------------------------------------------------

    def eval(self, s, t, u, *, dressed: bool = True):
        """Amplitude ``M(s, t, u)`` (s-, t-, u-channel graviton exchange).

        Accepts scalars or broadcastable arrays.  Set ``dressed=False`` for
        the undressed classical-GR reference (``G(x) = G_N``).
        """
        s_a = np.asarray(s, dtype=float)
        t_a = np.asarray(t, dtype=float)
        u_a = np.asarray(u, dtype=float)
        m_s = self._channel(s_a, t_a * u_a, dressed)
        m_t = self._channel(t_a, s_a * u_a, dressed)
        m_u = self._channel(u_a, s_a * t_a, dressed)
        total = m_s + m_t + m_u
        return complex(total) if total.ndim == 0 else total.astype(complex)

    def eval_point(self, mand: Mandelstam, *, dressed: bool = True):
        """Amplitude at a :class:`Mandelstam` point."""
        return self.eval(mand.s, mand.t, mand.u, dressed=dressed)

    def gr_amplitude(self, s, t, u):
        """Undressed classical-GR amplitude (``G(x) = G_N``)."""
        return self.eval(s, t, u, dressed=False)

    def amplitude_vs_s(self, s_values, cos_theta: float, *, dressed: bool = True):
        """Amplitude over an array of ``s`` at fixed scattering angle."""
        s_arr = np.asarray(s_values, dtype=float)
        p_sq = s_arr / 4.0 - self.m**2
        t = -2.0 * p_sq * (1.0 - cos_theta)
        u = -2.0 * p_sq * (1.0 + cos_theta)
        return self.eval(s_arr, t, u, dressed=dressed)

    def differential_cross_section(self, s: float, cos_theta: float,
                                   *, dressed: bool = True) -> float:
        """``dσ/dΩ = |M|² / (64π² s)`` (massless normalisation)."""
        mand = Mandelstam.from_s_angle(s, cos_theta, self.m)
        M = self.eval_point(mand, dressed=dressed)
        return float(abs(M) ** 2 / (64.0 * math.pi**2 * s))

    # -- limit diagnostics -------------------------------------------------

    def ir_limit(self, cos_theta: float = 0.3, s_small: float = 1e-3) -> dict:
        """Check the IR amplitude matches classical GR.

        Returns the dressed/undressed ratio at small ``s``; ``→ 1`` when
        the trajectory reproduces the Newtonian limit.
        """
        mand = Mandelstam.from_s_angle(s_small, cos_theta, self.m)
        dressed = self.eval_point(mand, dressed=True)
        gr = self.eval_point(mand, dressed=False)
        ratio = float(abs(dressed) / abs(gr)) if abs(gr) > 0 else float("nan")
        return {
            "s": s_small,
            "cos_theta": cos_theta,
            "dressed": complex(dressed),
            "gr": complex(gr),
            "ratio": ratio,
            "matches_gr": abs(ratio - 1.0) < 0.1,
        }

    def uv_limit(self, cos_theta: float = 0.3,
                 s_values=None) -> dict:
        """Check the UV amplitude is bounded (vs the growing GR amplitude).

        Returns the dressed and undressed |M| at the largest sampled ``s``
        and whether the dressed amplitude is bounded relative to GR.
        """
        if s_values is None:
            s_values = np.geomspace(1.0, 1e6, 64)
        s_arr = np.asarray(s_values, dtype=float)
        dressed = np.abs(self.amplitude_vs_s(s_arr, cos_theta, dressed=True))
        gr = np.abs(self.amplitude_vs_s(s_arr, cos_theta, dressed=False))
        return {
            "s_max": float(s_arr[-1]),
            "dressed_max": float(np.nanmax(dressed)),
            "gr_at_s_max": float(gr[-1]),
            "dressed_at_s_max": float(dressed[-1]),
            # GR grows ∝ s at fixed angle; AS softening keeps it bounded.
            "bounded": bool(dressed[-1] < gr[-1]),
        }
