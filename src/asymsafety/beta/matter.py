"""Matter contributions to gravitational beta functions.

Computes how N_s scalars, N_D Dirac fermions, and N_v gauge fields
modify the gravitational beta functions and the existence/properties
of the Reuter fixed point.

The key question: for how many matter fields does the NGFP survive?

Known results (approximate, gauge/regulator dependent):
    Pure gravity: NGFP exists
    + scalars: survives up to N_s ~ 4-20 (depends on truncation)
    + vectors: survives with modifications
    + fermions: can enhance or destroy the NGFP

References:
    Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035
    Meibohm, Pawlowski & Reichert (2016), Phys. Rev. D 93, 084035
    Eichhorn & Schiffer (2022), in Handbook of Quantum Gravity [2212.07456]
        (updated matter bounds in covariant setting)
    Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 [2402.01260]
        (foliated gravity-matter bounds)
    Buccio, Percacci et al. (2025), Phys. Rev. D 111, 085030 [2501.10307]
        (AS with non-vanishing scalar quartic coupling)
"""

import sympy
from sympy import Expr, Rational, Symbol, pi

from asymsafety.actions.matter import MatterContent, matter_eta_N_correction
from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.frg.anomalous_dim import AnomalousDimensionSolver
from asymsafety.frg.threshold import ThresholdFunctions


def build_eh_matter_beta_system(
    matter: MatterContent,
    d: int = 4,
) -> BetaFunctionSystem:
    """Build EH beta functions including matter contributions.

    Args:
        matter: Specification of the matter content.
        d: Spacetime dimension.

    Returns:
        BetaFunctionSystem with β_g and β_λ including matter.
    """
    g = Symbol("g", positive=True)
    lam = Symbol("lambda", real=True)
    tf = ThresholdFunctions()
    solver = AnomalousDimensionSolver(tf)

    # Gravitational contribution
    A_grav, B_grav = solver.compute_AB_einstein_hilbert(lam, d)

    # Matter contribution
    A_matter, B_matter = matter_eta_N_correction(matter, d)

    # Combined anomalous dimension
    A_total = A_grav + A_matter
    B_total = B_grav + B_matter
    eta_N = solver.solve(g, A_total, B_total)

    # β_g
    beta_g_expr = (d - 2 + eta_N) * g

    # β_λ: gravitational + matter volume contributions
    w = -2 * lam
    Phi_1_2_w = tf.Phi(1, 2, w)
    Phi_1_2_0 = tf.Phi(1, 2, 0)
    tPhi_1_2_w = tf.Phi_tilde(1, 2, w)
    tPhi_1_2_0 = tf.Phi_tilde(1, 2, 0)

    # Gravitational trace (TT + ghost)
    grav_volume = (
        5 * Phi_1_2_w - 4 * Phi_1_2_0
        - eta_N * (Rational(5, 6) * tPhi_1_2_w
                   - Rational(2, 3) * tPhi_1_2_0)
    )

    # Matter volume contributions:
    # Each massless scalar: +1 × Φ^1_2(0)
    # Each Dirac fermion: -4 × Φ^1_2(0) (Grassmann)
    # Each gauge boson: +(d-2) × Φ^1_2(0)
    Phi_1_d2_0 = tf.Phi(1, Rational(d, 2), 0)
    matter_volume = (
        matter.n_scalars * Phi_1_2_0
        - 4 * matter.n_dirac * Phi_1_2_0  # Grassmann sign
        + (d - 2) * matter.n_vectors * Phi_1_2_0
    )

    beta_lam_expr = -(2 - eta_N) * lam + g / (2 * pi) * (
        grav_volume + matter_volume
    )

    system = BetaFunctionSystem()
    system.add(BetaFunction("g", g, beta_g_expr))
    system.add(BetaFunction("lambda", lam, beta_lam_expr))

    return system


def scan_matter_content(
    n_scalars_range: range = range(0, 20),
    d: int = 4,
) -> list[dict]:
    """Scan the number of scalar fields and track the NGFP.

    Args:
        n_scalars_range: Range of scalar field numbers to scan.
        d: Spacetime dimension.

    Returns:
        List of dicts with keys: n_scalars, fp_exists, g_star, lambda_star,
        theta_1, theta_2.
    """
    from asymsafety.analysis.fixed_points import FixedPointFinder

    results = []
    guess = {"g": 0.7, "lambda": 0.2}

    for n_s in n_scalars_range:
        matter = MatterContent(n_scalars=n_s)
        system = build_eh_matter_beta_system(matter, d)
        finder = FixedPointFinder(system)

        fp = finder.find_fixed_point(guess)
        if fp is not None and not fp.is_gaussian and fp.location.get("g", 0) > 0:
            results.append({
                "n_scalars": n_s,
                "fp_exists": True,
                "g_star": fp.location["g"],
                "lambda_star": fp.location["lambda"],
                "theta_1": complex(fp.critical_exponents[0]) if len(fp.critical_exponents) > 0 else None,
                "theta_2": complex(fp.critical_exponents[1]) if len(fp.critical_exponents) > 1 else None,
            })
            # Update guess for continuation
            guess = fp.location.copy()
        else:
            results.append({
                "n_scalars": n_s,
                "fp_exists": False,
                "g_star": None,
                "lambda_star": None,
                "theta_1": None,
                "theta_2": None,
            })

    return results
