"""Benchmark: Knorr (2026), *Asymptotically (un)safe scattering amplitudes*.

*Asymptotically (un)safe scattering amplitudes from scratch: a deep dive
into the IR jungle* [arXiv:2602.21285].

Physics summary
---------------
The central lesson: the existence of a non-Gaussian fixed point does
**not** by itself guarantee a well-behaved (bounded) scattering amplitude.
Whether the amplitude is "safe" (bounded) or "unsafe" (growing) depends on
the *momentum dependence* of the form factors, not merely on the fixed
point; naive renormalization-group improvement / derivative expansion can
fail to capture this.

Toolkit realisation
-------------------
We contrast two amplitudes built from the *same* couplings:

    - **safe:** a momentum-scale identification (``EnergyScale``) so that
      the graviton form factor softens, ``f(p²) ∝ p²`` — bounded amplitude
      (UV growth exponent ≈ 0).
    - **unsafe:** a frozen scale (``FixedScale``, no running) so the form
      factor does not soften — the amplitude grows like classical gravity
      (UV growth exponent ≈ 1).

This reproduces, at the level of this RG-improvement model, the paper's
qualitative "safe vs unsafe" dichotomy.  It is deliberately *not* a claim
that the toolkit captures the quantitative IR structure Knorr analyses.
"""

from __future__ import annotations

KNORR_BOUNDEDNESS = {
    "safe_growth_exponent": 0.0,     # softened form factor -> bounded
    "unsafe_growth_exponent": 1.0,   # no softening -> grows like GR
}

VALIDATION_TOL = {
    "safe_atol": 0.2,        # |exponent| < 0.2 => bounded ("safe")
    "unsafe_min": 0.5,       # exponent > 0.5 => growing ("unsafe")
}


def make_unsafe_amplitude(trajectory, k_fixed: float = 1.0):
    """Build an "unsafe" amplitude (frozen scale, no UV softening)."""
    from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
    from asymsafety.scattering.form_factor import GravitonFormFactor
    from asymsafety.scattering.scale import FixedScale

    ff = GravitonFormFactor(trajectory, scale=FixedScale(k_fixed=k_fixed))
    return GravitonMediatedAmplitude(ff)


def validate_safe_vs_unsafe(safe_amplitude, unsafe_amplitude) -> dict:
    """Check the safe/unsafe boundedness dichotomy.

    Args:
        safe_amplitude: AS amplitude with a softening (running) form factor.
        unsafe_amplitude: amplitude with a non-softening (frozen) form
            factor — see :func:`make_unsafe_amplitude`.
    """
    from asymsafety.scattering import consistency as C

    safe = C.froissart_growth(safe_amplitude, dressed=True)["growth_exponent"]
    unsafe = C.froissart_growth(
        unsafe_amplitude, dressed=True
    )["growth_exponent"]

    results = {
        "safe_bounded": {
            "computed": safe,
            "reference": KNORR_BOUNDEDNESS["safe_growth_exponent"],
            "relative_error": abs(safe),
            "passed": abs(safe) < VALIDATION_TOL["safe_atol"],
        },
        "unsafe_grows": {
            "computed": unsafe,
            "reference": KNORR_BOUNDEDNESS["unsafe_growth_exponent"],
            "relative_error": abs(unsafe - 1.0),
            "passed": unsafe > VALIDATION_TOL["unsafe_min"],
        },
    }
    results["dichotomy_holds"] = (
        results["safe_bounded"]["passed"] and results["unsafe_grows"]["passed"]
    )
    results["all_passed"] = results["dichotomy_holds"]
    return results
