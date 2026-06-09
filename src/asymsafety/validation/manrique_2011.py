"""Benchmark values from Manrique, Rechenberger & Saueressig (2011).

Foliated Einstein-Hilbert truncation with ADM decomposition on S¹ × S³
("Asymptotically Safe Lorentzian Gravity", Phys. Rev. Lett. 106,
251302 [1102.5012]).

Key results (MRS Eq. (10)):
    - Euclidean NGFP:  g* ≈ 0.19, λ* ≈ 0.31, θ = 1.07 ± 3.31 i
    - Lorentzian NGFP: g* ≈ 0.21, λ* ≈ 0.30, θ = 0.94 ± 3.10 i
    - Two relevant directions in both signatures; the close agreement
      between the Euclidean and Lorentzian fixed points is the paper's
      headline result.

Important caveats:
    - The MRS truncation contains NO running λ_ADM coupling: their
      ansatz is diffeomorphism invariant, so λ_ADM = 1 is *imposed by
      construction*, not derived as a fixed-point value. The
      ``lambda_ADM_star`` entries below record that ansatz choice for
      interface compatibility only.
    - These are literature reference values. The toolkit's schematic
      foliated system
      (:func:`asymsafety.beta.foliated.build_foliated_eh_beta_system`)
      does NOT realize this NGFP — its (g, λ) sector admits no
      non-Gaussian root at physical λ (see :mod:`asymsafety.beta.foliated`).

The 2025 Wick rotation result (Saueressig et al.) revisits the
analytic continuation of the foliated flow to Lorentzian signature and
the Feynman causal structure of the graviton propagator (see
:mod:`asymsafety.validation.lorentzian_2024`).

References:
    Manrique, Rechenberger & Saueressig (2011),
        Phys. Rev. Lett. 106, 251302 [1102.5012]
    Rechenberger & Saueressig (2013), JHEP 03, 010
    Biemans, Platania & Saueressig (2017), JHEP 05, 093 [1609.02803]
    Knorr, Ripken & Saueressig (2023), JHEP 09, 064 [2306.10408]
        (fluctuation approach)
    Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752]
        (Wick rotation)
"""

# Euclidean foliated NGFP — MRS PRL 106, 251302, Eq. (10).
FOLIATED_EH_FP = {
    "g_star": 0.19,
    "lambda_star": 0.31,
    "theta_real": 1.07,
    "theta_imag": 3.31,
    # λ_ADM = 1 is imposed by the Diff-invariant ansatz (MRS have no
    # running λ_ADM coupling); recorded for interface compatibility.
    "lambda_ADM_star": 1.0,
    "n_relevant": 2,  # Same as covariant EH
}

# Lorentzian foliated NGFP — same paper, Eq. (10), after Wick rotation.
LORENTZIAN_FP = {
    "g_star": 0.21,
    "lambda_star": 0.30,
    "theta_real": 0.94,
    "theta_imag": 3.10,
    "lambda_ADM_star": 1.0,  # imposed by the ansatz, as above
}

VALIDATION_TOL = {
    "lambda_ADM_atol": 0.05,  # ansatz value λ_ADM = 1 (not dynamical)
    "fp_rtol": 0.20,          # 20% tolerance on g*, λ*
}
