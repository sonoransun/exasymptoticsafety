"""Benchmark values from Korver, Saueressig & Wang (2024).

Global flows of foliated gravity-matter systems: bounds on the number
of matter fields compatible with asymptotic safety in the foliated
Einstein-Hilbert truncation.

Key results:
    - The foliated NGFP annihilates against a second fixed point along
      (approximately) straight lines in the (N_s, N_v) plane:
          N_s + 6.4 N_v ≈ 23.1   (p = 0 regulator scheme)
          N_s + 4.7 N_v ≈ 22.4   (p-vec regulator scheme)
      so scalars alone admit N_s ≲ 22–23 and vectors alone N_v ≲ 3–4.
    - IR fixed point for graviton mass prevents negative squared mass
    - Phase diagram qualitatively stable under matter addition

These dicts hold *literature* values for reference and figure shading;
the toolkit's own foliated system (:mod:`asymsafety.beta.foliated`) is
a schematic truncation that does not reproduce these annihilation
bounds.

References:
    Korver, Saueressig & Wang (2024),
        Phys. Lett. B 855, 138789 [2402.01260]
    Eichhorn & Schiffer (2022),
        in Handbook of Quantum Gravity [2212.07456]
"""

# Matter field bounds from foliated asymptotic safety
# (Korver, Saueressig & Wang 2024 [2402.01260]): the NGFP-annihilation
# wedge N_s + 6.4 N_v ≲ 23.1 in the p = 0 regulator scheme.
FOLIATED_MATTER_BOUNDS = {
    "wedge_N_v_coefficient": 6.4,  # N_s + 6.4·N_v ≤ wedge_rhs (p = 0)
    "wedge_rhs": 23.1,
    "max_N_s": 23,        # Max scalars with N_v = 0 (wedge intercept ≈ 23.1)
    "max_N_v": 3,         # Max vectors with N_s = 0 (⌊23.1/6.4⌋ = 3)
    "graviton_mass_ir_fp": True,  # IR FP prevents negative mass squared
    "phase_diagram_stable": True,  # Qualitatively stable under matter
    "reference": "Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789",
}

# Updated covariant matter bounds from Eichhorn & Schiffer (2022)
# These are from the standard (non-foliated) EH truncation.
COVARIANT_MATTER_BOUNDS = {
    "max_N_s_minimal": 22,   # Minimally coupled scalars (upper estimate)
    "max_N_s_nonminimal": 4,  # Non-minimally coupled (more restrictive)
    "max_N_D": 10,            # Dirac fermions (approximate)
    "max_N_v": 12,            # Gauge vectors (approximate)
    "reference": "Eichhorn & Schiffer (2022), Handbook of Quantum Gravity",
}


def validate_foliated_matter_bounds(
    N_s: int,
    N_v: int,
    ngfp_exists: bool,
) -> dict:
    """Check computed NGFP existence against the KSW annihilation wedge.

    The Korver–Saueressig–Wang bound is the half-plane
    ``N_s + 6.4 N_v ≤ 23.1`` (p = 0 regulator scheme of [2402.01260]):
    inside the wedge the foliated NGFP exists, beyond it the NGFP has
    annihilated. The check fails (``consistent=False``) whenever the
    computed ``ngfp_exists`` disagrees with the wedge prediction in
    *either* direction — an NGFP found outside the wedge or one missing
    inside it both flag an inconsistency.

    Args:
        N_s: Number of minimally coupled scalars.
        N_v: Number of gauge vector fields.
        ngfp_exists: Whether the NGFP was found numerically.

    Returns:
        Dict with consistency check results.
    """
    bounds = FOLIATED_MATTER_BOUNDS
    within_bounds = (
        N_s + bounds["wedge_N_v_coefficient"] * N_v <= bounds["wedge_rhs"]
    )

    results = {
        "N_s": N_s,
        "N_v": N_v,
        "within_foliated_bounds": within_bounds,
        "ngfp_found": ngfp_exists,
        "consistent": within_bounds == ngfp_exists,
        "reference": bounds["reference"],
    }
    return results
