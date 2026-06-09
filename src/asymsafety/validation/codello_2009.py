"""Reference values for quadratic (R² + C²) gravity.

Two distinct kinds of numbers live in this module — do not conflate them:

1. ``ONE_LOOP_UNIVERSAL`` — the exact, scheme- and gauge-independent
   one-loop runnings of the four-derivative coefficients, stated in the
   *coefficient basis* of the toolkit action
   (``actions/quadratic.py``: Γ ⊃ α R² + β C²). These are what
   :func:`asymsafety.beta.quadratic.build_quadratic_beta_system`
   reproduces exactly. Because both constants are nonzero, the one-loop
   system has **no interior fixed point** in (α, β); the couplings run
   logarithmically. β_β > 0 is asymptotic freedom of the C² sector in
   this convention: the coefficient β grows toward the UV, i.e. the
   inverse coupling λ_C = 1/(2β) → 0⁺.

2. ``QUADRATIC_FP`` — NGFP coordinates quoted from the full
   nonperturbative FRG calculation of Codello, Percacci & Rahmede
   (2009). These are *external literature reference values only*: the
   toolkit's one-loop truncation does **not** possess this fixed point
   (β_α and β_β are nonzero constants, and their Jacobian columns
   vanish, so ``n_relevant = 4`` is not realizable here either).

References:
    Fradkin & Tseytlin (1982), Nucl. Phys. B201, 469
    Avramidi & Barvinsky (1985), Phys. Lett. B159, 269
    Codello & Percacci (2006) [hep-th/0607128]
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785]
"""

# Literature NGFP coordinates from the full nonperturbative calculation
# (Codello-Percacci-Rahmede 2009; approximate, scheme/gauge dependent).
# NOT a fixed point of the toolkit's one-loop truncation — external
# reference values only.
QUADRATIC_FP = {
    "g_star": 0.97,
    "lambda_star": 0.14,
    "alpha_star": 0.006,  # Small R² coupling at the literature FP
    "beta_star": 0.002,   # Small C² coupling at the literature FP
    "n_relevant": 4,       # CPR result; unrealizable in the one-loop system
}

# One-loop universal coefficients (scheme- and gauge-independent),
# coefficient basis: d_t(coefficient) = (1/16π²) × value below.
# Reproduced exactly by build_quadratic_beta_system.
ONE_LOOP_UNIVERSAL = {
    # ∂_t α |_{α=0} = (1/16π²) × 5/36: R² coefficient running at ω=0,
    # from β_ω(0) = −(25/60)λ_C/(16π²) (Avramidi–Barvinsky 1985;
    # Codello–Percacci hep-th/0607128).
    "beta_alpha_1loop": 5 / 36,

    # ∂_t β = (1/16π²) × 133/20: C² coefficient running, from
    # β_{λ_C} = −(133/10)λ_C²/(16π²) (Fradkin–Tseytlin 1982).
    # POSITIVE = asymptotic freedom in the coefficient convention
    # (β → +∞, λ_C = 1/(2β) → 0⁺ in the UV).
    "beta_beta_1loop": 133 / 20,
}

VALIDATION_TOL = {
    "fp_rtol": 0.5,  # 50% tolerance (the literature QUADRATIC_FP values
                      # are highly scheme-dependent)
}
