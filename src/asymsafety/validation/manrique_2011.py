"""Benchmark values from Manrique, Rechenberger & Saueressig (2011).

Foliated Einstein-Hilbert truncation with ADM decomposition on S¹ × S³.

Key results:
    - NGFP exists in the foliated formulation
    - λ_ADM = 1 at the NGFP (full-Diff restoration)
    - g* ≈ 0.96, λ* ≈ 0.20 (comparable to covariant result)
    - Critical exponents: complex conjugate pair + one real

The agreement between covariant (S⁴) and foliated (S¹ × S³)
results is a non-trivial consistency check of the program.

The 2025 Wick rotation result (Saueressig et al.) confirms that
λ_ADM → 1 at the NGFP is preserved under analytic continuation
to Lorentzian signature, and that the Lorentzian graviton propagator
has the correct Feynman causal structure.

References:
    Manrique, Rechenberger & Saueressig (2011), Phys. Rev. Lett. 106, 251302
    Rechenberger & Saueressig (2013), JHEP 03, 010
    Biemans, Platania & Saueressig (2017), JHEP 05, 093 (Lorentzian)
    Knorr, Ripken & Saueressig (2023), JHEP 09, 064 [2306.10408]
        (fluctuation approach)
    Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752]
        (Wick rotation confirmation)
"""

FOLIATED_EH_FP = {
    "g_star": 0.96,
    "lambda_star": 0.20,
    "lambda_ADM_star": 1.0,
    "n_relevant": 2,  # Same as covariant EH
}

# Lorentzian foliated results (Biemans et al. 2017)
LORENTZIAN_FP = {
    "g_star": 1.57,
    "lambda_star": 0.12,
    "lambda_ADM_star": 1.0,
}

VALIDATION_TOL = {
    "lambda_ADM_atol": 0.05,  # λ_ADM should be very close to 1
    "fp_rtol": 0.20,          # 20% tolerance on g*, λ*
}
