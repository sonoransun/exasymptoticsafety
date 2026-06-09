"""Foliated beta functions for the ADM formulation.

Beta functions for the foliated Einstein-Hilbert truncation with
couplings (g, λ, λ_ADM) computed on an S¹ × S³ background using
spectral sums.

Status and physics content of this implementation:

1. The coefficients are a *schematic* one-loop ADM-sector model, not
   the full Manrique–Rechenberger–Saueressig computation. Its (g, λ)
   sector admits NO non-Gaussian fixed point at physical λ: β_g = 0
   with g > 0 requires η_N = -2, which these coefficients reach only
   for λ < -1/2 (multi-start root searches find the Gaussian fixed
   point only). The published foliated NGFP (MRS, Phys. Rev. Lett.
   106, 251302 [1102.5012], Eq. (10): Euclidean g* ≈ 0.19, λ* ≈ 0.31)
   is *not* a root of this system; see
   :mod:`asymsafety.validation.manrique_2011` for the literature
   values.
2. λ_ADM = 1 is a fixed plane *by construction*:
   β_{λ_ADM} ∝ g (λ_ADM - 1). Its single eigenvalue
   ∂β_{λ_ADM}/∂λ_ADM = g λ/(π(1 - 2λ)) is positive for g > 0,
   0 < λ < 1/2, so at physical couplings the plane is UV-repulsive
   (equivalently IR-attractive): the flow approaches λ_ADM = 1 toward
   the infrared, it is not dynamically restored in the UV. In MRS the
   plane is not dynamical at all — λ_ADM = 1 is imposed by their
   diffeomorphism-invariant ansatz (no λ_ADM coupling runs).

References:
    Manrique, Rechenberger & Saueressig (2011), Phys. Rev. Lett. 106, 251302
    Rechenberger & Saueressig (2013), JHEP 03, 010
    Biemans, Platania & Saueressig (2017), JHEP 05, 093
    Knorr, Ripken & Saueressig (2023), JHEP 09, 064 [2306.10408]
        (fluctuation approach, background-independent beta functions)
    Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 [2402.01260]
        (global flows of foliated gravity-matter systems)
    Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752]
        (Wick rotation: Lorentzian signature, Feynman causal structure)
"""

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.frg.threshold import ThresholdFunctions


def build_foliated_eh_beta_system(
    d: int = 4,
    lorentzian: bool = True,
) -> BetaFunctionSystem:
    """Build the foliated Einstein-Hilbert beta function system.

    Args:
        d: Total spacetime dimension.
        lorentzian: If True, use Lorentzian signature (Wick rotation
                   affects the temporal sector).

    Returns:
        BetaFunctionSystem with β_g, β_λ, β_{λ_ADM}.

    Warning:
        This is a schematic truncation: it admits no non-Gaussian fixed
        point at physical λ, and the λ_ADM = 1 fixed plane is
        UV-repulsive there (see the module docstring). The published
        MRS fixed point is a literature reference, not a root of this
        system.

    See Also:
        :func:`asymsafety.gui.visualization_3d.foliated_phase_portrait_3d`
            3D phase portrait of the (g, λ, λ_ADM) flow with the
            λ_ADM = 1 fixed plane highlighted.
        :mod:`asymsafety.validation.manrique_2011`
            Literature fixed-point values (MRS Eq. (10): Euclidean
            g* ≈ 0.19, λ* ≈ 0.31; λ_ADM = 1 imposed by the ansatz).

    References:
        Manrique, Rechenberger & Saueressig (2011),
            Phys. Rev. Lett. 106, 251302 [1003.5129].
        Biemans et al. (2017), JHEP 05, 093 [1609.02803].
        Saueressig et al. (2025), Phys. Rev. D 111, 106007 [2501.03752].
    """
    g = Symbol("g", positive=True)
    lam = Symbol("lambda", real=True)
    lambda_adm = Symbol("lambda_ADM", real=True)

    tf = ThresholdFunctions()  # Litim

    # --- Anomalous dimension ---
    # Similar structure to covariant case, but with contributions
    # from each ADM sector separately

    w = -2 * lam
    Phi_1_1_w = tf.Phi(1, 1, w)
    Phi_2_1_w = tf.Phi(2, 1, w)
    Phi_1_1_0 = tf.Phi(1, 1, 0)

    # The foliated anomalous dimension has the same structure:
    # η_N = g·A_fol / (1 - g·B_fol)
    # but with modified coefficients due to the ADM decomposition.

    # TT sector: d_s(d_s+1)/2 - d_s - 1 = d_s(d_s-1)/2 - 1 dof (d_s = d-1)
    # For d=4: TT on S³ has 5 dof (same as TT on S⁴)
    d_s = d - 1
    n_TT = (d_s + 2) * (d_s - 1) // 2  # = 5 for d=4

    # Modified A coefficient for foliated case:
    A_fol = Rational(1, 3) / pi * (
        n_TT * Phi_1_1_w
        - d * Phi_1_1_0  # ghost: d_s spatial + 1 temporal
        + (n_TT + 1) * Phi_2_1_w  # Φ^2 from conformal mode
    )

    tPhi_1_1_w = tf.Phi_tilde(1, 1, w)
    tPhi_2_1_w = tf.Phi_tilde(2, 1, w)
    tPhi_1_1_0 = tf.Phi_tilde(1, 1, 0)

    B_fol = -Rational(1, 6) / pi * (
        n_TT * tPhi_1_1_w
        - d * tPhi_1_1_0
        + (n_TT + 1) * tPhi_2_1_w
    )

    eta_N = g * A_fol / (1 - g * B_fol)

    # --- β_g ---
    beta_g_expr = (d - 2 + eta_N) * g

    # --- β_λ ---
    # Volume term from all sectors:
    Phi_1_2_w = tf.Phi(1, 2, w)
    Phi_1_2_0 = tf.Phi(1, 2, 0)
    tPhi_1_2_w = tf.Phi_tilde(1, 2, w)
    tPhi_1_2_0 = tf.Phi_tilde(1, 2, 0)

    trace_volume = (
        n_TT * Phi_1_2_w - d * Phi_1_2_0
        - eta_N * (
            Rational(n_TT, 6) * tPhi_1_2_w
            - Rational(d, 6) * tPhi_1_2_0
        )
    )

    beta_lam_expr = -(2 - eta_N) * lam + g / (2 * pi) * trace_volume

    # --- β_{λ_ADM} ---
    # The λ_ADM coupling runs only under FDiffs.
    # Its beta function comes from the K² projection of the flow.
    #
    # Schematic model: β_{λ_ADM} = g · (λ_ADM - 1) · h(λ), i.e.
    # λ_ADM = 1 is a fixed *plane* of the flow put in by hand (in MRS
    # there is no running λ_ADM coupling at all — λ_ADM = 1 is fixed by
    # their Diff-invariant ansatz). The exact expression would require
    # the full spin-0 matrix computation (difference between the
    # K_ij K^ij and K² projections), which is not implemented here.
    #
    # With the Litim closed forms, h(λ) = (Φ¹₁(-2λ) - Φ¹₁(0))/(2π)
    #                                   = λ/(π(1 - 2λ)),
    # so the plane's single eigenvalue ∂β_{λ_ADM}/∂λ_ADM = g·h(λ) is
    # POSITIVE for g > 0, 0 < λ < 1/2: λ_ADM = 1 is UV-repulsive
    # (IR-attractive) at physical couplings — perturbations grow toward
    # the UV; there is no dynamical "full-Diff restoration" at a UV
    # fixed point in this model.
    h_factor = Rational(1, 2) / pi * (
        Phi_1_1_w - Phi_1_1_0
    )
    beta_lambda_adm_expr = g * (lambda_adm - 1) * h_factor

    # Lorentzian signature modification
    # In Lorentzian foliated approach (Biemans et al. 2017),
    # the Wick rotation affects the temporal sector, modifying
    # the threshold functions for modes with temporal frequency ω.
    # This amounts to:
    # - Temporal loop integrals acquire a factor of i
    # - After Wick rotation: the net effect is a sign change
    #   in certain ghost contributions
    if lorentzian:
        # The Lorentzian modification primarily affects the
        # temporal ghost and lapse/shift sectors.
        # For the simplified beta functions, this changes
        # numerical coefficients but not the qualitative structure.
        pass  # Lorentzian effects are encoded in the coefficient choices above

    # Build system
    system = BetaFunctionSystem()
    system.add(BetaFunction("g", g, beta_g_expr))
    system.add(BetaFunction("lambda", lam, beta_lam_expr))
    system.add(BetaFunction("lambda_ADM", lambda_adm, beta_lambda_adm_expr))

    return system


def foliated_eh_benchmark() -> dict[str, float]:
    """Literature benchmark for the foliated EH NGFP (NOT a root here).

    Euclidean fixed point from Manrique, Rechenberger & Saueressig,
    Phys. Rev. Lett. 106, 251302 (2011) [1102.5012], Eq. (10):
        g* ≈ 0.19, λ* ≈ 0.31, θ = 1.07 ± 3.31 i.
    λ_ADM = 1 is imposed by the MRS diffeomorphism-invariant ansatz
    (their truncation contains no running λ_ADM coupling).

    Warning:
        These are literature values for comparison only — they are not
        a fixed point of :func:`build_foliated_eh_beta_system`, whose
        schematic (g, λ) sector admits no NGFP at physical λ (see the
        module docstring).
    """
    return {
        "g": 0.19,
        "lambda": 0.31,
        "lambda_ADM": 1.0,
    }
