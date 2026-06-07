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

Reproduced properties (open bosonic string, ``α0 = α' = 1``):

    - **Regge spectrum** ``m_n² = (n − α0)/α'`` = ``-1, 0, 1, 2, …``.
    - **Crossing** of the Veneziano (``s ↔ t``) and Virasoro–Shapiro
      (full ``s,t,u``) amplitudes.
    - **Higher-spin structure:** the level-``n`` residue is a polynomial
      of degree ``n`` (maximal spin ``n``), vanishing at the prescribed
      momentum-transfer points ``α(t) = -1,…,-n``.
    - **Ultrasoft falloff:** super-polynomial fixed-angle decay.
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
    "residue_atol": 1e-9,
}


def validate_bootstrap() -> dict:
    """Validate the bootstrap reference amplitudes and spectrum."""
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

    # Virasoro-Shapiro full crossing
    vs = B.virasoro_shapiro(2.1, -1.3, -2.8)
    vs_cross = abs(vs - B.virasoro_shapiro(-1.3, 2.1, -2.8))
    results["virasoro_shapiro_crossing"] = {
        "computed": float(vs_cross),
        "reference": 0.0,
        "relative_error": float(vs_cross),
        "passed": vs_cross < VALIDATION_TOL["crossing_atol"],
    }

    # Higher-spin residue cancellation
    max_resid = 0.0
    for n in (1, 2, 3):
        zeros = B.residue_zeros(n, alpha0=a0, alphap=ap)
        r = B.veneziano_residue(n, zeros, alpha0=a0, alphap=ap)
        max_resid = max(max_resid, float(np.max(np.abs(r))))
    results["higher_spin_cancellation"] = {
        "computed": max_resid,
        "reference": 0.0,
        "relative_error": max_resid,
        "passed": max_resid < VALIDATION_TOL["residue_atol"],
    }

    # Ultrasoft falloff of the Virasoro-Shapiro amplitude
    sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
    soft = B.ultrasoft_falloff(sa, cos_theta=0.3, s_lo=5.0, s_hi=50.0)
    results["ultrasoft_falloff"] = {
        "computed": soft["slope_hi"],
        "reference": "super-polynomial",
        "relative_error": 0.0,
        "passed": bool(soft["ultrasoft"]),
    }

    results["all_passed"] = all(
        v["passed"] for v in results.values()
        if isinstance(v, dict) and "passed" in v
    )
    return results
