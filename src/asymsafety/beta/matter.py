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

This module also provides ``build_gravity_matter_fp_system`` which
promotes selected matter couplings (scalar quartic, Yukawa, non-minimal)
to dynamical running variables, enabling the study of gravity-matter
fixed points with non-trivial matter self-interactions.

References:
    Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035
    Meibohm, Pawlowski & Reichert (2016), Phys. Rev. D 93, 084035
    Eichhorn & Schiffer (2022), in Handbook of Quantum Gravity [2212.07456]
        (updated matter bounds in covariant setting)
    Korver, Saueressig & Wang (2024), Phys. Lett. B 855, 138789 [2402.01260]
        (foliated gravity-matter bounds)
    Buccio, Percacci et al. (2025), Phys. Rev. D 111, 085030 [2501.10307]
        (AS with non-vanishing scalar quartic coupling)
    Eichhorn, Held & Pawlowski (2020), Phys. Rev. D 101, 026006 [1903.10411]
        (Yukawa couplings in asymptotic safety)
    Narain & Percacci (2010), Class. Quant. Grav. 27, 075001 [0911.1948]
        (running non-minimal coupling)
"""

from __future__ import annotations

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

    See Also:
        :func:`asymsafety.visualization.fixed_point_plot.plot_matter_content_continuation`
            NGFP-vs-matter-content visualisation with the Korver
            (2024) bound shaded.
        :mod:`asymsafety.validation.korver_2024`
            Foliated and covariant matter bounds.

    References:
        Dona, Eichhorn & Percacci (2014),
            Phys. Rev. D 89, 084035 [1311.2898].
        Eichhorn & Schiffer (2022) [2212.07456].
        Korver, Saueressig & Wang (2024),
            Phys. Lett. B 855, 138789 [2402.01260].
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


def scalar_anomalous_dimension(
    g: Symbol,
    lam: Symbol,
    d: int = 4,
) -> Expr:
    """Scalar field anomalous dimension from graviton loops.

    eta_phi = -5/(12*pi) * g / (1 - 2*lambda)^2

    This is the one-loop gravitational contribution to the scalar
    wave-function renormalization. The denominator (1-2*lambda)^2
    comes from the graviton propagator threshold.

    References:
        Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035, Eq. (4.12)
    """
    one_m_2l = 1 - 2 * lam
    return -Rational(5, 12) / pi * g / one_m_2l**2


def build_gravity_matter_fp_system(
    matter: MatterContent,
    d: int = 4,
    scalar_quartic: bool = False,
    yukawa: bool = False,
    running_xi: bool = False,
    include_eta_phi: bool = True,
) -> BetaFunctionSystem:
    """Build coupled gravity-matter beta system with dynamical matter couplings.

    Uses the same gravitational sector as ``build_eh_beta_system`` (N/D
    anomalous dimension formulation) and adds matter loop corrections
    to the cosmological constant running.  Optional extensions promote
    selected matter couplings to dynamical variables:

        scalar_quartic: lambda_phi (phi^4 quartic self-coupling)
        yukawa: y (Yukawa scalar-fermion coupling)
        running_xi: xi (non-minimal scalar-gravity coupling)

    Args:
        matter: Specification of the matter field content.
        d: Spacetime dimension.
        scalar_quartic: Add lambda_phi as a dynamical coupling.
        yukawa: Add y as a dynamical coupling (requires n_dirac >= 1).
        running_xi: Promote xi to a dynamical coupling (requires
            scalar_quartic=True).
        include_eta_phi: Include the scalar anomalous dimension
            self-consistently in beta_{lambda_phi}.

    Returns:
        BetaFunctionSystem with (g, lambda, [lambda_phi], [y], [xi]).

    Raises:
        ValueError: If yukawa=True but matter.n_dirac == 0.
        ValueError: If running_xi=True but scalar_quartic=False.

    References:
        Buccio, Percacci et al. (2025), Phys. Rev. D 111, 085030
        Eichhorn, Held & Pawlowski (2020), Phys. Rev. D 101, 026006
        Narain & Percacci (2010), Class. Quant. Grav. 27, 075001
    """
    if yukawa and matter.n_dirac == 0:
        raise ValueError(
            "yukawa=True requires matter.n_dirac >= 1"
        )
    if running_xi and not scalar_quartic:
        raise ValueError(
            "running_xi=True requires scalar_quartic=True "
            "(xi running is driven by lambda_phi)"
        )

    # --- Symbols ---
    g = Symbol("g", positive=True)
    lam = Symbol("lambda", real=True)
    lambda_phi_sym = Symbol("lambda_phi", real=True) if scalar_quartic else None
    y_sym = Symbol("y", positive=True) if yukawa else None
    xi_sym = Symbol("xi", real=True) if running_xi else None

    one_m_2l = 1 - 2 * lam

    # --- Gravitational anomalous dimension (same as build_eh_beta_system) ---
    # eta_N = g*N(lambda) / (1 - g*D(lambda))
    # N and D encode graviton + ghost traces in single-metric EH truncation
    # with Type I Litim cutoff and de Donder gauge.
    N_eta = (Rational(4, 3) / one_m_2l + 8 - 8 / one_m_2l**2)
    D_eta = (Rational(1, 3) / one_m_2l + 2 - 2 / one_m_2l**2)

    # Matter correction to N_eta: each matter field shifts the graviton
    # self-energy through one-loop diagrams. The leading correction is
    # proportional to the scalar b_2 heat kernel coefficient.
    # Per scalar: delta_N = -1/(6(1-2lambda))
    # Per Dirac fermion: delta_N = +1/(3(1-2lambda))
    # Per gauge vector: delta_N = -(d-2)/(6(1-2lambda))
    delta_N_matter: Expr = (
        matter.n_scalars * (-Rational(1, 6) / one_m_2l)
        + matter.n_dirac * (Rational(1, 3) / one_m_2l)
        + matter.n_vectors * (-Rational(d - 2, 6) / one_m_2l)
    )

    N_total = N_eta + delta_N_matter
    eta_N = g * N_total / (1 - g * D_eta)

    # Scalar anomalous dimension from graviton loops
    eta_phi_expr: Expr = sympy.S.Zero
    if include_eta_phi and scalar_quartic:
        eta_phi_expr = scalar_anomalous_dimension(g, lam, d)

    # --- beta_g ---
    beta_g_expr = (2 + eta_N) * g

    # --- beta_lambda ---
    # Gravitational volume (same as build_eh_beta_system)
    vol = (3 / one_m_2l - 4
           - eta_N * (1 / one_m_2l - Rational(4, 3)))

    # Matter volume corrections (matched to EH module normalisation):
    # Each scalar: +1/32, each Dirac: -1/8, each vector: +(d-2)/32
    delta_vol_matter: Expr = (
        matter.n_scalars * Rational(1, 32)
        - matter.n_dirac * Rational(1, 8)
        + matter.n_vectors * Rational(d - 2, 32)
    )

    # Backreaction of quartic coupling on beta_lambda (scalar mass shift)
    if scalar_quartic:
        delta_vol_matter = delta_vol_matter + (
            matter.n_scalars * lambda_phi_sym * Rational(1, 64)
        )

    # Backreaction of Yukawa coupling on beta_lambda (fermion mass shift)
    if yukawa:
        delta_vol_matter = delta_vol_matter + (
            -matter.n_dirac * y_sym**2 * Rational(1, 16)
        )

    beta_lam_expr = (-(2 - eta_N) * lam
                     + 8 * g / pi * (vol + delta_vol_matter))

    # --- Build system ---
    system = BetaFunctionSystem()
    system.add(BetaFunction("g", g, beta_g_expr))
    system.add(BetaFunction("lambda", lam, beta_lam_expr))

    # --- beta_{lambda_phi} ---
    if scalar_quartic:
        # One-loop coefficients from Buccio, Percacci et al. (2025)
        c1 = Rational(3, 1)        # phi^4 self-loop
        c2 = -Rational(5, 4) / pi  # graviton vertex correction
        c3 = Rational(5, 32) / pi**2  # graviton box diagram

        beta_lambda_phi_expr = (
            (d - 4 + 2 * eta_phi_expr) * lambda_phi_sym
            + c1 * lambda_phi_sym**2 / (16 * pi**2)
            + c2 * g * lambda_phi_sym
            + c3 * g**2
        )

        system.add(BetaFunction(
            "lambda_phi", lambda_phi_sym, beta_lambda_phi_expr
        ))

    # --- beta_y ---
    if yukawa:
        # One-loop coefficients from Eichhorn, Held & Pawlowski (2020)
        c_y1 = Rational(5, 2)       # Yukawa self-loop
        c_y2 = -Rational(5, 8) / pi  # graviton vertex correction

        beta_y_expr = (
            Rational(d - 4, 2) * y_sym
            + c_y1 * y_sym**3 / (16 * pi**2)
            + c_y2 * g * y_sym
        )

        system.add(BetaFunction("y", y_sym, beta_y_expr))

    # --- beta_xi ---
    if running_xi:
        from asymsafety.utils.conventions import conformal_coupling

        xi_conf = conformal_coupling(d)
        c_xi1 = Rational(1, 1)         # scalar loop
        c_xi2 = -Rational(5, 12) / pi  # graviton correction
        c_xi3 = Rational(5, 72) / pi   # graviton loop

        beta_xi_expr = (
            (xi_sym - xi_conf) * (
                c_xi1 * lambda_phi_sym / (16 * pi**2)
                + c_xi2 * g
            )
            + c_xi3 * g
        )

        system.add(BetaFunction("xi", xi_sym, beta_xi_expr))

    return system


def scan_gravity_matter_fps(
    n_scalars_range: range = range(1, 10),
    d: int = 4,
    scalar_quartic: bool = True,
    yukawa: bool = False,
    running_xi: bool = False,
) -> list[dict]:
    """Scan matter content and track gravity-matter fixed points.

    For each n_scalars, builds the extended system and searches for
    the shifted NGFP with non-zero matter couplings.

    Args:
        n_scalars_range: Range of scalar field numbers to scan.
        d: Spacetime dimension.
        scalar_quartic: Include lambda_phi as dynamical coupling.
        yukawa: Include y as dynamical coupling.
        running_xi: Include xi as dynamical coupling.

    Returns:
        List of dicts with keys: n_scalars, fp_exists, plus all
        coupling values and critical exponents at the FP.
    """
    from asymsafety.analysis.fixed_points import FixedPointFinder

    results: list[dict] = []
    guess: dict[str, float] = {"g": 0.65, "lambda": 0.14}
    if scalar_quartic:
        guess["lambda_phi"] = 0.01
    if yukawa:
        guess["y"] = 0.0
    if running_xi:
        guess["xi"] = 1.0 / 6

    for n_s in n_scalars_range:
        matter = MatterContent(
            n_scalars=n_s,
            n_dirac=1 if yukawa else 0,
        )
        system = build_gravity_matter_fp_system(
            matter, d,
            scalar_quartic=scalar_quartic,
            yukawa=yukawa,
            running_xi=running_xi,
        )
        finder = FixedPointFinder(system)
        fp = finder.find_fixed_point(guess)

        entry: dict = {
            "n_scalars": n_s,
            "fp_exists": (
                fp is not None
                and not fp.is_gaussian
                and fp.location.get("g", 0) > 0
            ),
        }
        if entry["fp_exists"]:
            entry.update(fp.location)
            entry["critical_exponents"] = [
                complex(ce) for ce in fp.critical_exponents
            ]
            entry["n_relevant"] = fp.relevant_directions
            guess = fp.location.copy()
        else:
            for name in system.coupling_names:
                entry[name] = None
            entry["critical_exponents"] = None
            entry["n_relevant"] = None

        results.append(entry)

    return results
