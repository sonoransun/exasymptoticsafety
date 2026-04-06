"""Benchmark values from Korver, Saueressig & Wang (2024).

Global flows of foliated gravity-matter systems: bounds on the number
of matter fields compatible with asymptotic safety in the foliated
Einstein-Hilbert truncation.

Key results:
    - Matter field bounds derived from foliated AS
    - IR fixed point for graviton mass prevents negative squared mass
    - Phase diagram qualitatively stable under matter addition
    - Bounds cast by asymptotic safety on N_s (scalars) and N_v (vectors)

References:
    Korver, Saueressig & Wang (2024),
        Phys. Lett. B 855, 138789 [2402.01260]
    Eichhorn & Schiffer (2022),
        in Handbook of Quantum Gravity [2212.07456]
"""

# Matter field bounds from foliated asymptotic safety
# (Korver, Saueressig & Wang 2024)
# These are approximate bounds — the NGFP ceases to exist beyond them.
FOLIATED_MATTER_BOUNDS = {
    "max_N_s": 12,        # Maximum minimally coupled scalars
    "max_N_v": 6,         # Maximum gauge vectors
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
    """Check whether computed NGFP existence is consistent with bounds.

    Args:
        N_s: Number of minimally coupled scalars.
        N_v: Number of gauge vector fields.
        ngfp_exists: Whether the NGFP was found numerically.

    Returns:
        Dict with consistency check results.
    """
    bounds = FOLIATED_MATTER_BOUNDS
    within_bounds = N_s <= bounds["max_N_s"] and N_v <= bounds["max_N_v"]

    results = {
        "N_s": N_s,
        "N_v": N_v,
        "within_foliated_bounds": within_bounds,
        "ngfp_found": ngfp_exists,
        "consistent": (within_bounds == ngfp_exists) or within_bounds,
        "reference": bounds["reference"],
    }
    return results
