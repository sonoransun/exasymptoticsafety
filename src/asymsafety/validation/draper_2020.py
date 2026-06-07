"""Benchmark: Draper, Knorr, Ripken & Saueressig (2020).

*Graviton-Mediated Scattering Amplitudes from the Quantum Effective
Action*, Phys. Rev. Lett. 125, 181301 (2020) [arXiv:2007.04396].

Physics summary
---------------
A curvature/form-factor expansion of the gravitational effective action
yields gauge-invariant graviton-mediated scattering amplitudes for
non-minimally coupled scalars.  The qualitative, scheme-robust statements
the toolkit reproduces:

    - **Infrared:** the amplitude reduces to the classical (Newtonian)
      graviton-exchange result — the dressed/undressed ratio → 1.
    - **Ultraviolet:** at the non-Gaussian fixed point the form factor
      grows ``f(p²) ∝ p²``, softening ``G(p²) ∝ 1/p²`` so that the
      fixed-angle amplitude tends to a finite constant (UV finiteness).
    - **No new degrees of freedom:** the dressing has no extra poles /
      ghosts — ``f(p²) > 0`` with no zeros — so causality holds without
      non-localities or additional fields.

These are *qualitative* benchmarks: the present toolkit uses an
RG-improvement model (momentum-scale identification), not the full
curvature-expansion form factors of the paper, so only the limits and the
structural consistency are compared.
"""

from __future__ import annotations

DRAPER_AMPLITUDE = {
    # Infrared: dressed amplitude matches classical GR.
    "ir_ratio_to_gr": 1.0,
    # Ultraviolet: finite, bounded amplitude (no power-law growth).
    "uv_finite": True,
    "uv_growth_exponent": 0.0,
    # Causality: form factor has no ghost pole.
    "ghost_free": True,
}

VALIDATION_TOL = {
    "ir_ratio_rtol": 0.1,        # 10% on the IR Newtonian limit
    "uv_growth_atol": 0.2,       # |exponent| < 0.2 counts as UV-constant
}


def validate_graviton_amplitude(amplitude) -> dict:
    """Validate an AS graviton-mediated amplitude against Draper et al.

    Args:
        amplitude: a
            :class:`asymsafety.scattering.amplitude.GravitonMediatedAmplitude`.

    Returns a dict of per-check ``{"computed", "reference", "passed"}``
    plus an ``all_passed`` flag.
    """
    from asymsafety.scattering import consistency as C

    results: dict = {}

    ir = amplitude.ir_limit()
    ir_err = abs(ir["ratio"] - DRAPER_AMPLITUDE["ir_ratio_to_gr"])
    results["ir_newtonian_limit"] = {
        "computed": ir["ratio"],
        "reference": DRAPER_AMPLITUDE["ir_ratio_to_gr"],
        "relative_error": ir_err,
        "passed": ir_err < VALIDATION_TOL["ir_ratio_rtol"],
    }

    fro = C.froissart_growth(amplitude, dressed=True)
    results["uv_finite"] = {
        "computed": fro["growth_exponent"],
        "reference": DRAPER_AMPLITUDE["uv_growth_exponent"],
        "relative_error": abs(fro["growth_exponent"]),
        "passed": abs(fro["growth_exponent"]) < VALIDATION_TOL["uv_growth_atol"],
    }

    ghost = C.causality_no_ghosts(amplitude)
    results["ghost_free"] = {
        "computed": ghost["passed"],
        "reference": DRAPER_AMPLITUDE["ghost_free"],
        "relative_error": 0.0,
        "passed": bool(ghost["passed"]),
    }

    results["all_passed"] = all(
        v["passed"] for v in results.values()
        if isinstance(v, dict) and "passed" in v
    )
    return results
