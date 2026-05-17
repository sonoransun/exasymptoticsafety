"""Benchmark values from Bonati, Pelissetto & Vicari (2025).

Three-dimensional Abelian and non-Abelian gauge-Higgs theories — a
Physics Reports review consolidating lattice Monte-Carlo and field-theory
results for the *charged fixed point* of gauge-Higgs systems.

Used here as a cross-disciplinary analogue for the asymptotic-safe
non-Gaussian fixed point (NGFP): the charged FP is the closest 3D
statistical-mechanics analogue of an interacting UV fixed point, in
that its existence and critical exponents depend non-trivially on the
matter content (``Nf``, ``Nc``) and on the regulator/scheme — mirroring
debates about the gravitational NGFP under different cutoff schemes.

Phenomena reproduced as reference data:

* Continuous Coulomb↔Higgs transitions in SU(2) + ``Nf`` fundamental
  scalars, requiring ``Nf > Nf*`` (≈ 375 in 4-ε, observed ≪ 30 in 3D).
* Large-``Nf`` field-theory prediction for the correlation-length
  exponent at the charged FP:  ν ≈ 1 − C(Nc)/Nf  +  O(Nf⁻²).
* Lattice MC values of (β_c, ν) at Nf ∈ {30, 40, 60} for the SU(2)
  Coulomb↔Higgs transition.

References:
    Bonati, Pelissetto & Vicari (2025),
        "Three-dimensional Abelian and non-Abelian gauge-Higgs theories",
        Phys. Rep. [arXiv:2410.05823]
    Bonati, Pelissetto & Vicari (2024),
        "Charged critical behavior and nonperturbative continuum limit of
        SU(N_c) gauge Higgs models", [arXiv:2409.03595]
    Bonati, Pelissetto & Vicari (2022),
        "Multicritical point of the three-dimensional Z_2 gauge Higgs
        model", [arXiv:2112.01824]
"""

# Lattice Monte-Carlo critical points for SU(2) + Nf fundamental scalars,
# 3D Coulomb↔Higgs transition. Keyed by Nf.
BONATI_SU2_MC = {
    30: {"beta_c": 1.22435, "beta_c_err": 1.0e-4, "nu": 0.64,  "nu_err": 0.02},
    40: {"beta_c": 1.18630, "beta_c_err": 1.0e-4, "nu": 0.745, "nu_err": 0.015},
    60: {"beta_c": 1.14160, "beta_c_err": 1.0e-4, "nu": 0.81,  "nu_err": 0.02},
}

# Large-Nf field-theory prediction:  ν ≈ 1 − C/Nf  with C = 9.727 for SU(2).
BONATI_LARGE_NF = {
    "Nc": 2,
    "nu_coeff": 9.727,        # C in ν = 1 − C/Nf
    "form": "1 - C/Nf",
}

# Lower critical Nf above which the charged FP is stable.
# Paper only gives "≪ 30" in d=3 (no point estimate); we store a range.
CHARGED_FP_THRESHOLDS = {
    "4d_eps": 375,            # ε-expansion at d = 4 − ε, SU(2)
    "3d_estimate_max": 30,    # upper bound consistent with paper's "≪ 30"
}

# Tolerances for cross-analogue validation. Looser than gravitational
# benchmarks because the toolkit's one-loop 4−ε β-system is a perturbative
# proxy, not a Monte-Carlo simulation.
VALIDATION_TOL = {
    "nu_rtol": 0.10,                # MC ν agreement (loose)
    "large_nf_coeff_rtol": 0.05,    # large-Nf coefficient
    "monotonicity": True,           # ν must increase with Nf
}


def validate_nu_vs_nf(computed_nu: dict[int, float]) -> dict:
    """Validate a computed ν(Nf) curve against the MC table.

    Parameters
    ----------
    computed_nu
        Mapping ``Nf -> ν`` for at least the Nf values in
        :data:`BONATI_SU2_MC`.

    Returns
    -------
    dict
        Per-Nf checks plus ``"monotonic"`` and ``"all_passed"`` flags.
    """
    results: dict = {}

    for Nf, ref in BONATI_SU2_MC.items():
        if Nf not in computed_nu:
            results[Nf] = {"passed": False, "reason": "missing"}
            continue
        nu = computed_nu[Nf]
        err = abs(nu - ref["nu"]) / ref["nu"]
        results[Nf] = {
            "computed": nu,
            "reference": ref["nu"],
            "reference_err": ref["nu_err"],
            "relative_error": err,
            "passed": err < VALIDATION_TOL["nu_rtol"],
        }

    # Monotonicity check: ν should grow with Nf toward 1 (mean-field limit).
    Nf_sorted = sorted(BONATI_SU2_MC)
    nus = [computed_nu.get(Nf) for Nf in Nf_sorted]
    if all(n is not None for n in nus):
        monotonic = all(nus[i] <= nus[i + 1] for i in range(len(nus) - 1))
    else:
        monotonic = False
    results["monotonic"] = monotonic

    results["all_passed"] = monotonic and all(
        v.get("passed") for k, v in results.items()
        if isinstance(v, dict) and "passed" in v
    )
    return results


def validate_charged_fp_existence(Nf: int, d: int) -> bool:
    """Return ``True`` if the charged FP is expected to exist for ``(Nf, d)``.

    Uses :data:`CHARGED_FP_THRESHOLDS`: in 4D the ε-expansion bound
    ``Nf > 375`` is required; in 3D the paper places the threshold
    well below 30, so any ``Nf ≥ 30`` is safely above it.
    """
    if d == 4:
        return Nf > CHARGED_FP_THRESHOLDS["4d_eps"]
    if d == 3:
        return Nf >= CHARGED_FP_THRESHOLDS["3d_estimate_max"]
    return False


def large_nf_nu(Nf: int, Nc: int = 2) -> float:
    """Large-Nf prediction for ν at the charged FP, ``ν = 1 − C/Nf``."""
    if Nc != BONATI_LARGE_NF["Nc"]:
        raise NotImplementedError(
            f"Large-Nf coefficient only tabulated for Nc={BONATI_LARGE_NF['Nc']}"
        )
    return 1.0 - BONATI_LARGE_NF["nu_coeff"] / Nf
