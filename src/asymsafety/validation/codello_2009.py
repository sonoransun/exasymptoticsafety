"""Benchmark values from Codello, Percacci & Rahmede (2009).

Quadratic gravity truncation with R² and C² couplings.

Key results:
    - NGFP exists in the (g, λ, α, β) coupling space
    - The C² coupling β is asymptotically free (β_β < 0 at the GFP)
    - The R² coupling α has a NGFP value
    - 4 critical exponents (possibly complex)

References:
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785]
"""

# Benchmark values (approximate, scheme/gauge dependent)
QUADRATIC_FP = {
    "g_star": 0.97,
    "lambda_star": 0.14,
    "alpha_star": 0.006,  # Small R² coupling at the FP
    "beta_star": 0.002,   # Small C² coupling at the FP
    "n_relevant": 4,       # All 4 directions are relevant at the NGFP
}

# One-loop universal coefficients (scheme-independent)
ONE_LOOP_UNIVERSAL = {
    # β_α^{1-loop} coefficient (scheme independent at one loop)
    "beta_alpha_1loop": 53 / 45,  # (1/16π²) × this

    # β_β^{1-loop} coefficient (asymptotic freedom!)
    "beta_beta_1loop": -196 / 45,  # (1/16π²) × this, NEGATIVE = AF
}

VALIDATION_TOL = {
    "fp_rtol": 0.5,  # 50% tolerance (quadratic gravity FP values
                      # are highly scheme-dependent)
}
