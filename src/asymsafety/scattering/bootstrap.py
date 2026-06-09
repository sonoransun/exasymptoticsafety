"""Physical-scattering bootstrap: the Veneziano / Virasoro–Shapiro amplitudes.

This is the "physical scattering" side of the combination — the amplitude
bootstrap of C. Cheung, G. N. Remmen, F. Sciotti & M. Tarquini, *Strings
from Almost Nothing*, Phys. Rev. Lett. (DOI ``cw4p-cqh7`` /
[arXiv:2508.09246]).  That paper shows that a few physical-scattering
assumptions — analyticity, crossing, ultrasoft (faster-than-power-law)
high-energy behaviour, and an infinite sequence of momentum-transfer
values at which higher-spin exchanges cancel — single out *uniquely* the
Veneziano (open-string) and Virasoro–Shapiro (closed-string) amplitudes,
with the Regge mass spectrum appearing as an **output** of the bootstrap.

We implement those reference amplitudes and the diagnostics for their
defining physical properties, so the asymptotically-safe graviton-mediated
amplitude can be cross-checked against the *same* criteria in
:mod:`asymsafety.scattering.bridge`.

Conventions:
    Linear Regge trajectory ``α(x) = α0 + α' x``.  Open bosonic string
    (Veneziano): ``α0 = 1``, ``α' = 1``.  Poles of ``Γ(-α(x))`` at
    ``α(x) = 0,1,2,…`` give the tower ``x_n = (n - α0)/α'`` of squared
    masses, with the residue at level ``n`` a polynomial in the crossed
    channel of degree ``n`` (maximal exchanged spin ``n``).  Closed
    string (Virasoro–Shapiro): massless external states, ``α0 = 0`` with
    ``u = -s - t``, so the trajectories satisfy the constraint
    ``α_s + α_t + α_u = 0`` on which the bootstrap operates
    (Virasoro 1969; Polchinski *String Theory* Vol. 1, §6.2;
    arXiv:2508.09246).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma


def regge_trajectory(x, alpha0: float = 1.0, alphap: float = 1.0):
    """Linear Regge trajectory ``α(x) = α0 + α' x``."""
    return alpha0 + alphap * np.asarray(x, dtype=float)


def mass_spectrum(
    n_max: int = 6, alpha0: float = 1.0, alphap: float = 1.0
) -> np.ndarray:
    """Squared-mass tower ``x_n = (n - α0)/α'`` for ``n = 0 … n_max``.

    These are the locations of the ``Γ(-α(x))`` poles — the string states.
    For the open bosonic string (``α0=α'=1``) this gives the tachyon at
    ``-1``, the massless level at ``0``, then ``1, 2, …``.
    """
    n = np.arange(n_max + 1)
    return (n - alpha0) / alphap


def max_spin_at_level(n: int) -> int:
    """Maximal exchanged spin at mass level ``n`` (= ``n``)."""
    return int(n)


def veneziano(s, t, *, alpha0: float = 1.0, alphap: float = 1.0):
    """Veneziano (open-string four-point) amplitude.

    ``A(s,t) = B(-α(s), -α(t)) = Γ(-α_s)Γ(-α_t)/Γ(-α_s-α_t)``.
    Manifestly ``s ↔ t`` crossing symmetric; poles at ``α(s),α(t) ∈ ℤ≥0``.
    """
    a_s = regge_trajectory(s, alpha0, alphap)
    a_t = regge_trajectory(t, alpha0, alphap)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = gamma(-a_s) * gamma(-a_t) / gamma(-a_s - a_t)
    return out if np.ndim(out) else complex(out)


def virasoro_shapiro(s, t, u, *, alpha0: float = 0.0, alphap: float = 0.25):
    """Virasoro–Shapiro (closed-string four-point) amplitude.

    ``A(s,t,u) = Γ(-α_s)Γ(-α_t)Γ(-α_u) /
                 [Γ(1+α_s)Γ(1+α_t)Γ(1+α_u)]``
    (Virasoro 1969; Polchinski *String Theory* Vol. 1, Eqs.
    (6.2.39)–(6.2.40); arXiv:2508.09246), manifestly fully crossing
    symmetric in ``s, t, u``.

    The defaults are the massless closed-string convention: ``α0 = 0``
    with ``alphap`` playing the role of ``α'/4``, so on the massless
    surface ``s + t + u = 0`` the trajectories obey
    ``α_s + α_t + α_u = 0``.  Each channel then carries the infinite
    Regge tower of ``Γ(-α)`` poles at ``α(x) = 0, 1, 2, …``
    (``x_n = n/alphap``), the residue in ``α(s)`` at level ``n`` is the
    positive polynomial ``[∏_{k=1}^{n-1}(α_t+k)]²/(n!)²`` of degree
    ``2(n-1)``, and the fixed-angle falloff is super-polynomial — the
    ultrasoft behaviour required by the bootstrap of arXiv:2508.09246.
    """
    a_s = regge_trajectory(s, alpha0, alphap)
    a_t = regge_trajectory(t, alpha0, alphap)
    a_u = regge_trajectory(u, alpha0, alphap)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = gamma(-a_s) * gamma(-a_t) * gamma(-a_u)
        den = gamma(1 + a_s) * gamma(1 + a_t) * gamma(1 + a_u)
        out = num / den
    return out if np.ndim(out) else complex(out)


def veneziano_residue(n: int, t, *, alpha0: float = 1.0, alphap: float = 1.0):
    """Residue of the Veneziano amplitude at the ``α(s) = n`` pole.

    Near ``α_s = n`` one has ``Γ(-α_s) ≈ -(-1)^n/[n!(α_s - n)]`` and
    ``Γ(-α_t)/Γ(-n-α_t) = (-1)^n ∏_{k=1}^{n}(α_t + k)``, so the residue
    in the trajectory variable ``α(s)`` is

    ``Res = -(1/n!) · ∏_{k=1}^{n} (α(t) + k)``

    with a sign *uniform* in ``n`` (the two ``(-1)^n`` factors cancel;
    cf. arXiv:2508.09246) — a polynomial in ``α(t)`` of degree exactly
    ``n`` (the higher-spin content up to spin ``n``), with zeros at
    ``α(t) = -1, -2, …, -n``.
    """
    from math import factorial

    a_t = regge_trajectory(t, alpha0, alphap)
    poch = np.ones_like(np.atleast_1d(a_t), dtype=float)
    for kk in range(1, n + 1):
        poch = poch * (a_t + kk)
    res = -poch / factorial(n)
    return res if np.ndim(t) else float(res[0])


def residue_zeros(n: int, *, alpha0: float = 1.0, alphap: float = 1.0):
    """Momentum-transfer values where the level-``n`` residue vanishes.

    The higher-spin cancellation points of *Strings from Almost Nothing*:
    ``α(t) = -1,…,-n`` ⇒ ``t = (-k - α0)/α'``.
    """
    ks = np.arange(1, n + 1)
    return (-ks - alpha0) / alphap


@dataclass
class StringAmplitude:
    """Adapter exposing a bootstrap amplitude with the AS-amplitude API.

    Provides :meth:`eval` and :meth:`amplitude_vs_s` so the bridge can run
    the same consistency/high-energy diagnostics on a string amplitude as
    on the asymptotically-safe one.  External states are treated as
    effectively massless for the fixed-angle high-energy comparison.

    Args:
        kind: ``"veneziano"`` or ``"virasoro_shapiro"``.
        alpha0, alphap: Regge-trajectory parameters.  ``alpha0`` defaults
            per kind: ``0`` for ``virasoro_shapiro`` (massless closed
            string, so the ``u = -s-t`` kinematics built here satisfy
            the constraint ``α_s+α_t+α_u = 0``) and ``1`` for
            ``veneziano`` (open bosonic string).
        m: External mass used to build kinematics (default massless).
    """

    kind: str = "veneziano"
    alpha0: float | None = None
    alphap: float = 0.25
    m: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha0 is None:
            self.alpha0 = 0.0 if self.kind == "virasoro_shapiro" else 1.0

    def eval(self, s, t, u=None, *, dressed: bool = True):  # noqa: D401
        del dressed  # the bootstrap amplitude has no "undressed" variant
        if self.kind == "veneziano":
            return veneziano(s, t, alpha0=self.alpha0, alphap=self.alphap)
        if u is None:
            u = -np.asarray(s, dtype=float) - np.asarray(t, dtype=float)
        return virasoro_shapiro(
            s, t, u, alpha0=self.alpha0, alphap=self.alphap
        )

    def amplitude_vs_s(self, s_values, cos_theta: float, *, dressed: bool = True):
        s = np.asarray(s_values, dtype=float)
        p_sq = s / 4.0 - self.m**2
        t = -2.0 * p_sq * (1.0 - cos_theta)
        u = -2.0 * p_sq * (1.0 + cos_theta)
        return self.eval(s, t, u, dressed=dressed)


def ultrasoft_falloff(
    amplitude,
    *,
    cos_theta: float = 0.3,
    s_lo: float = 5.0,
    s_hi: float = 50.0,
    n: int = 60,
    s_values=None,
) -> dict:
    """Test for faster-than-power-law (ultrasoft) fixed-angle decay.

    A string amplitude falls off exponentially in the hard-scattering
    (fixed-angle) regime, so its local log–log slope ``d log|A|/d log s``
    grows ever more negative; a power law has a constant slope and a
    UV-constant amplitude has slope ``≈ 0``.  Returns the low- and
    high-energy slopes and whether the falloff is super-polynomial.

    Args:
        s_values: Optional explicit sample energies (overrides
            ``s_lo``/``s_hi``/``n``).  For a meromorphic string
            amplitude, pass points midway between the Regge poles —
            e.g. the dual ladder ``s = (n + 1/2)/alphap`` for the
            massless Virasoro–Shapiro convention — so the fit window
            is pole-free and the slopes track the decaying envelope
            rather than pole spikes.
    """
    if s_values is not None:
        s = np.asarray(s_values, dtype=float)
    else:
        s = np.geomspace(s_lo, s_hi, n)
    A = np.abs(np.asarray(amplitude.amplitude_vs_s(s, cos_theta)))
    good = np.isfinite(A) & (A > 0)
    s, A = s[good], A[good]
    ls, lA = np.log(s), np.log(A)
    half = len(s) // 2
    slope_lo = float(np.polyfit(ls[:half], lA[:half], 1)[0])
    slope_hi = float(np.polyfit(ls[half:], lA[half:], 1)[0])
    return {
        "cos_theta": cos_theta,
        "slope_lo": slope_lo,
        "slope_hi": slope_hi,
        # Exponential decay: the effective power keeps decreasing.
        "super_polynomial": slope_hi < slope_lo - 0.5,
        "ultrasoft": slope_hi < -1.0 and slope_hi < slope_lo - 0.5,
    }
