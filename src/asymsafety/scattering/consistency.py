"""Physical-scattering consistency checks for a 2 → 2 amplitude.

These are the "physical scattering" requirements that any sensible
amplitude is asked to satisfy — the same family of conditions the
amplitude bootstrap of Cheung, Remmen, Sciotti & Tarquini, *Strings from
Almost Nothing* (PRL, DOI ``cw4p-cqh7`` / [arXiv:2508.09246]) elevates to
axioms.  Here they are applied as **diagnostics** to the
asymptotically-safe graviton-mediated amplitude:

* **Partial-wave unitarity (boundedness)** — the decisive test:
  classical gravity's partial waves ``a_ℓ(s)`` grow ``∝ s`` and run off to
  infinity near the Planck scale, whereas the asymptotically-safe form
  factor keeps them **bounded** (a finite high-energy plateau).  Note this
  RG-improvement model does not claim the literal elastic bound
  ``|a_ℓ| ≤ 1`` — that value is normalisation-dependent and a safe fixed
  point alone does not guarantee it (Knorr 2026, [arXiv:2602.21285]).  The
  physical, scheme-robust statement is boundedness vs unbounded growth.
* **UV finiteness** — the fixed-angle amplitude stays finite as
  ``s → ∞``.
* **Froissart-type growth** — the high-energy growth exponent of the
  amplitude (AS ``≈ 0``; GR ``≈ 1``).
* **Causality / no ghosts** — the form factor ``f(p²) > 0`` is finite and
  has no zeros, i.e. the dressed propagator introduces no spurious poles
  or ghost degrees of freedom (Draper et al. 2020).  (Forward *positivity*
  bounds are deliberately not gated: the massless graviton ``t``-channel
  pole makes them famously subtle for gravity.)
* **Crossing** — symmetry of ``M(s,t,u)`` under ``s ↔ t`` for identical
  scalars (exact by construction for graviton exchange).

Every function returns a plain ``dict`` in the same style as the
:mod:`asymsafety.validation` modules (computed values + ``passed`` flags),
so they compose into a report and into the AS-vs-string bridge.

.. note::
    The massless graviton ``t``/``u``-channel poles make the forward and
    backward angular integrals divergent; the partial-wave projections use
    a symmetric angular cutoff ``cos_max`` (default 0.99) and this is
    reported, not hidden.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import eval_legendre


def _amplitude_vs_angle(amplitude, s, cos_grid, *, dressed=True, m=0.0):
    """Evaluate ``M(s, cosθ)`` on a grid of scattering angles."""
    p_sq = s / 4.0 - m**2
    t = -2.0 * p_sq * (1.0 - cos_grid)
    u = -2.0 * p_sq * (1.0 + cos_grid)
    s_arr = np.full_like(cos_grid, s)
    return amplitude.eval(s_arr, t, u, dressed=dressed)


def partial_wave(
    amplitude,
    ell: int,
    s: float,
    *,
    dressed: bool = True,
    cos_max: float = 0.99,
    n_theta: int = 400,
    m: float = 0.0,
) -> complex:
    """Partial-wave amplitude ``a_ℓ(s)``.

    ``M(s,θ) = 16π Σ_ℓ (2ℓ+1) a_ℓ(s) P_ℓ(cosθ)`` ⇒
    ``a_ℓ(s) = (1/32π) ∫ M(s,cosθ) P_ℓ(cosθ) d cosθ`` over
    ``[-cos_max, cos_max]`` (symmetric forward/backward cutoff).
    """
    cos_grid = np.linspace(-cos_max, cos_max, n_theta)
    M = _amplitude_vs_angle(amplitude, s, cos_grid, dressed=dressed, m=m)
    integrand = M * eval_legendre(ell, cos_grid)
    return complex(trapezoid(integrand, cos_grid) / (32.0 * math.pi))


def unitarity(
    amplitude,
    s_values=None,
    *,
    ell: int = 0,
    cos_max: float = 0.99,
    m: float = 0.0,
    bound: float = 1.0,
) -> dict:
    """Partial-wave unitarity over a range of ``s`` (dressed vs GR).

    Returns the ``|a_ℓ(s)|`` curves for the asymptotically-safe and the
    classical-GR amplitudes, whether the AS curve stays under ``bound``,
    and whether GR violates it (the expected high-energy breakdown).
    """
    if s_values is None:
        s_values = np.geomspace(1e-2, 1e6, 60)
    s_values = np.asarray(s_values, dtype=float)
    a_dressed = np.array([
        abs(partial_wave(amplitude, ell, float(s), dressed=True,
                         cos_max=cos_max, m=m)) for s in s_values
    ])
    a_gr = np.array([
        abs(partial_wave(amplitude, ell, float(s), dressed=False,
                         cos_max=cos_max, m=m)) for s in s_values
    ])
    as_max = float(np.nanmax(a_dressed))
    gr_max = float(np.nanmax(a_gr))

    # High-energy growth exponents of the partial wave: AS plateaus (≈0),
    # GR grows (≈1). This is the scheme-robust statement of the headline.
    def _growth(a):
        hi = s_values >= s_values[-1] / 1e3
        good = hi & (a > 0)
        if good.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(s_values[good]), np.log(a[good]), 1)[0])

    as_growth = _growth(a_dressed)
    gr_growth = _growth(a_gr)
    return {
        "ell": ell,
        "s_values": s_values,
        "a_dressed": a_dressed,
        "a_gr": a_gr,
        "as_max": as_max,
        "gr_max": gr_max,
        "as_growth_exponent": as_growth,
        "gr_growth_exponent": gr_growth,
        "cos_max": cos_max,
        # Informational: the literal elastic bound (model-dependent).
        "as_satisfies_elastic_bound": as_max <= bound,
        # Physical claim: AS partial waves are bounded (plateau, exponent≈0)
        # while GR grows without bound (exponent≈1).
        "as_bounded": as_growth < 0.2,
        "gr_unbounded": gr_growth > 0.5 and gr_max > as_max,
        "passed": (as_growth < 0.2) and (gr_growth > 0.5),
    }


def optical_theorem(amplitude, s: float, *, m: float = 0.0) -> dict:
    """Forward absorptive part ``Im M(s, t→0)`` (tree level ≈ 0).

    A tree amplitude is real away from poles, so the optical-theorem
    absorptive part vanishes — recorded here to make explicit that the
    cross section's imaginary part is a loop effect, not a tree one.
    """
    fwd = amplitude.eval(s, -1e-6 * s, -(s - 1e-6 * s), dressed=True)
    im = float(abs(fwd.imag))
    return {
        "s": s,
        "im_forward": im,
        "tree_level_real": im < 1e-6 * max(1.0, abs(fwd.real)),
        "passed": True,
    }


def uv_finiteness(
    amplitude, *, cos_theta: float = 0.3, s_max: float = 1e10, n: int = 120,
) -> dict:
    """The fixed-angle dressed amplitude stays finite as ``s → ∞``."""
    s = np.geomspace(1.0, s_max, n)
    M = np.abs(amplitude.amplitude_vs_s(s, cos_theta, dressed=True))
    finite = bool(np.isfinite(M).all())
    tail = M[-n // 5:]
    plateau = float(np.nanmax(tail))
    return {
        "s_max": s_max,
        "finite": finite,
        "uv_plateau": plateau,
        "amplitude_max": float(np.nanmax(M)),
        # Bounded if the global max is not far above the UV plateau.
        "passed": finite and np.nanmax(M) < 50.0 * plateau + 1.0,
    }


def froissart_growth(
    amplitude, *, cos_theta: float = 0.3, dressed: bool = True,
    s_lo: float = 1e2, s_hi: float = 1e8,
) -> dict:
    """High-energy growth exponent ``p`` of ``|M| ~ s^p`` at fixed angle.

    Asymptotic safety gives ``p ≈ 0`` (a UV-constant amplitude); classical
    gravity gives ``p ≈ 1``.  A Froissart-respecting amplitude has
    ``p ≲ 1`` (forward growth no faster than ``s log²s``).
    """
    s = np.geomspace(s_lo, s_hi, 80)
    M = np.abs(amplitude.amplitude_vs_s(s, cos_theta, dressed=dressed))
    mask = M > 0
    p = float(np.polyfit(np.log(s[mask]), np.log(M[mask]), 1)[0])
    return {
        "growth_exponent": p,
        "dressed": dressed,
        "froissart_respected": p <= 1.0 + 0.2,
        "uv_constant": abs(p) < 0.2,
        "passed": p <= 1.0 + 0.2,
    }


def causality_no_ghosts(
    amplitude, *, p2_lo: float = 1e-4, p2_hi: float = 1e8, n: int = 200,
) -> dict:
    """No-ghost / causality test on the graviton form factor.

    A causal dressing must keep ``f(p²) = G_N/G(p²) > 0`` and finite with
    no zeros: a zero of ``f`` is a pole of the dressed propagator, i.e. a
    spurious ghost degree of freedom.  Asymptotic safety's monotonically
    growing ``f(p²) ∝ p²`` introduces no such extra poles (Draper et al.
    2020 — UV finiteness/unitarity/causality "without additional degrees of
    freedom").
    """
    p2 = np.geomspace(p2_lo, p2_hi, n)
    f = np.asarray(amplitude.form_factor.f(p2), dtype=float)
    f_ir = float(f[0])
    positive = bool(np.all(f > 0))
    finite = bool(np.all(np.isfinite(f)))
    # A ghost pole of the dressed propagator is a *zero* of f.  The test is
    # therefore that f stays bounded away from zero (no sign change / pole),
    # not strict monotonicity — the form factor is ≈1 in the IR (with
    # sub-permille interpolation jitter) and grows in the UV.
    no_ghost_pole = positive and float(np.nanmin(f)) > 0.5 * abs(f_ir)
    softens_uv = bool(f[-1] > 10.0 * f_ir)  # f(p²) ∝ p² in the UV
    return {
        "p2_range": (p2_lo, p2_hi),
        "f_min": float(np.nanmin(f)),
        "f_ir": f_ir,
        "f_uv": float(f[-1]),
        "f_positive": positive,
        "f_finite": finite,
        "f_no_ghost_pole": no_ghost_pole,
        "f_softens_uv": softens_uv,
        "passed": positive and finite and no_ghost_pole,
    }


def crossing(
    amplitude, s: float = 5.0, t: float = -2.0, *, m: float = 0.0,
) -> dict:
    """Symmetry of ``M(s,t,u)`` under ``s ↔ t`` (identical scalars)."""
    u = 4.0 * m**2 - s - t
    direct = amplitude.eval(s, t, u, dressed=True)
    crossed = amplitude.eval(t, s, u, dressed=True)
    resid = float(abs(direct - crossed))
    scale = max(1.0, abs(direct))
    return {
        "residual": resid,
        "relative_residual": resid / scale,
        "passed": resid / scale < 1e-9,
    }


def consistency_report(amplitude, *, cos_theta: float = 0.3) -> dict:
    """Run the full battery and return a flat ``dict`` of results."""
    report = {
        "unitarity": unitarity(amplitude),
        "uv_finiteness": uv_finiteness(amplitude, cos_theta=cos_theta),
        "froissart": froissart_growth(amplitude, cos_theta=cos_theta),
        "causality": causality_no_ghosts(amplitude),
        "crossing": crossing(amplitude),
    }
    report["all_passed"] = all(
        v["passed"] for v in report.values() if isinstance(v, dict)
    )
    return report
