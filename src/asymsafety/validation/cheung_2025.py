"""Benchmark: Cheung, Remmen, Sciotti & Tarquini (2025).

*Strings from Almost Nothing*, Phys. Rev. Lett. (DOI ``cw4p-cqh7``)
[arXiv:2508.09246].  This is the "physical scattering" reference for the
combined-with-asymptotic-safety analysis in
:mod:`asymsafety.scattering.bridge`.

Physics summary
---------------
From a few physical-scattering assumptions — analyticity, crossing,
ultrasoft (faster-than-power-law) high-energy behaviour, and an infinite
sequence of momentum-transfer values at which higher-spin exchanges cancel
— the space of minimally consistent four-point amplitudes collapses onto
the Veneziano and Virasoro–Shapiro string amplitudes, with the Regge mass
spectrum as an *output*.

Reproduced properties (open bosonic string ``α0 = α' = 1`` for the
Veneziano checks; massless closed string ``α0 = 0``, ``α'_eff = 1/4``
for the Virasoro–Shapiro checks):

    - **Regge spectrum** ``m_n² = (n − α0)/α'`` = ``-1, 0, 1, 2, …``.
    - **Crossing** of the Veneziano (``s ↔ t``) and Virasoro–Shapiro
      (full ``s,t,u``, on the massless surface ``s+t+u = 0``) amplitudes.
    - **Higher-spin structure:** the level-``n`` residue
      ``-(1/n!)∏_{k=1}^{n}(α(t)+k)`` is a polynomial of degree ``n``
      (maximal spin ``n``), vanishing at the prescribed momentum-transfer
      points ``α(t) = -1,…,-n``; here the residues are extracted
      *numerically* from the amplitude itself, so the check is
      independent of the closed form it certifies.
    - **Ultrasoft falloff:** super-polynomial fixed-angle decay of the
      Virasoro–Shapiro amplitude, fitted on a pole-free window between
      the Regge poles.
"""

from __future__ import annotations

import numpy as np

CHEUNG_SPECTRUM = {
    # open bosonic string tower x_n = n - 1 for n = 0..5
    "alpha0": 1.0,
    "alphap": 1.0,
    "levels": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
}

VALIDATION_TOL = {
    "spectrum_atol": 1e-9,
    "crossing_atol": 1e-9,
    # Limited by the Richardson-extrapolated numerical residue
    # extraction (accuracy ~1e-10), not by the closed forms.
    "residue_atol": 1e-8,
}


def _veneziano_residue_numeric(
    n: int, t: float, *, alpha0: float, alphap: float
) -> float:
    """Residue of the Veneziano amplitude at ``α(s) = n`` in ``α(s)``.

    Extracted from the amplitude itself via a Richardson-extrapolated
    limit of ``(α(s) - n)·A(s, t)``, so it is *independent* of the
    closed-form :func:`asymsafety.scattering.bootstrap.veneziano_residue`
    (evaluating the closed form at its own roots could never fail).
    """
    from asymsafety.scattering import bootstrap as B

    s_n = (n - alpha0) / alphap
    r1 = 1e-5 * B.veneziano(s_n + 1e-5 / alphap, t, alpha0=alpha0, alphap=alphap)
    r2 = 1e-6 * B.veneziano(s_n + 1e-6 / alphap, t, alpha0=alpha0, alphap=alphap)
    return float(np.real((10.0 * r2 - r1) / 9.0))


def validate_bootstrap() -> dict:
    """Validate the bootstrap reference amplitudes and spectrum.

    The higher-spin check extracts the level residues numerically from
    the Veneziano amplitude and tests (i) that they vanish at the
    prescribed cancellation points and (ii) that the closed form
    ``veneziano_residue`` reproduces them at generic momentum transfer
    (a sign-sensitive comparison).  The ultrasoft check fits the
    massless (``α0 = 0``) Virasoro–Shapiro fixed-angle falloff on the
    pole-free dual ladder ``s = (n + 1/2)/α'_eff`` between the Regge
    poles ``s_n = n/α'_eff``.
    """
    from asymsafety.scattering import bootstrap as B

    a0 = CHEUNG_SPECTRUM["alpha0"]
    ap = CHEUNG_SPECTRUM["alphap"]
    results: dict = {}

    # Regge spectrum
    spec = B.mass_spectrum(5, alpha0=a0, alphap=ap)
    spec_err = float(np.max(np.abs(spec - np.array(CHEUNG_SPECTRUM["levels"]))))
    results["regge_spectrum"] = {
        "computed": spec.tolist(),
        "reference": CHEUNG_SPECTRUM["levels"],
        "relative_error": spec_err,
        "passed": spec_err < VALIDATION_TOL["spectrum_atol"],
    }

    # Veneziano crossing
    cross = abs(B.veneziano(3.3, -1.7) - B.veneziano(-1.7, 3.3))
    results["veneziano_crossing"] = {
        "computed": float(cross),
        "reference": 0.0,
        "relative_error": float(cross),
        "passed": cross < VALIDATION_TOL["crossing_atol"],
    }

    # Virasoro-Shapiro full crossing (massless surface s + t + u = 0)
    vs = B.virasoro_shapiro(2.1, -1.3, -0.8)
    vs_cross = abs(vs - B.virasoro_shapiro(-1.3, 2.1, -0.8))
    results["virasoro_shapiro_crossing"] = {
        "computed": float(vs_cross),
        "reference": 0.0,
        "relative_error": float(vs_cross),
        "passed": vs_cross < VALIDATION_TOL["crossing_atol"],
    }

    # Higher-spin residue cancellation -- non-circular: residues come
    # from the amplitude itself (numerical pole limit), checked to
    # vanish at the cancellation points AND to match the closed form
    # at generic t (which catches any sign/normalization error).
    max_resid = 0.0
    max_form_err = 0.0
    for n in (1, 2, 3):
        for tz in B.residue_zeros(n, alpha0=a0, alphap=ap):
            num = _veneziano_residue_numeric(n, float(tz), alpha0=a0, alphap=ap)
            max_resid = max(max_resid, abs(num))
        for tg in (-2.7, -3.9, -0.6):
            num = _veneziano_residue_numeric(n, tg, alpha0=a0, alphap=ap)
            form = float(B.veneziano_residue(n, tg, alpha0=a0, alphap=ap))
            max_form_err = max(max_form_err, abs(form - num))
    results["higher_spin_cancellation"] = {
        "computed": max_resid,
        "closed_form_error": max_form_err,
        "reference": 0.0,
        "relative_error": max_resid,
        "passed": (
            max_resid < VALIDATION_TOL["residue_atol"]
            and max_form_err < VALIDATION_TOL["residue_atol"]
        ),
    }

    # Ultrasoft falloff of the Virasoro-Shapiro amplitude.  With the
    # massless convention (alpha0 = 0) the s-channel poles sit at
    # s_n = n/alphap; sampling the dual ladder s = (n + 1/2)/alphap
    # keeps the fixed-angle fit window pole-free, so the
    # super-polynomial decay measured here is that of the amplitude's
    # envelope, not a pole-spike artefact.
    sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
    s_mid = (np.arange(1, 26) + 0.5) / sa.alphap
    soft = B.ultrasoft_falloff(sa, cos_theta=0.3, s_values=s_mid)
    results["ultrasoft_falloff"] = {
        "computed": soft["slope_hi"],
        "slope_lo": soft["slope_lo"],
        "reference": "super-polynomial",
        "relative_error": 0.0,
        "passed": bool(soft["ultrasoft"]),
    }

    results["all_passed"] = all(
        v["passed"] for v in results.values()
        if isinstance(v, dict) and "passed" in v
    )
    return results
