"""Einstein-Hilbert beta functions: β_g and β_λ.

Analytic expressions for the beta functions of the dimensionless
Newton coupling g and cosmological constant λ in the Einstein-Hilbert
truncation with the Litim regulator.

Conventions:
    g = G k^{d-2}               (dimensionless Newton coupling)
    λ = Λ / k^2                 (dimensionless cosmological constant)
    t = log(k/k_0)              (RG time)
    η_N = -∂_t ln(Z_N) = +∂_t ln(G)   (graviton anomalous dimension)

    β_g = (d - 2 + η_N) g
    β_λ = -(2 - η_N) λ + [trace contributions]

The anomalous dimension is determined self-consistently:
    η_N = g·A(λ) / (1 - g·B(λ))

where A(λ) and B(λ) encode the one-loop graviton and ghost traces
(see :class:`asymsafety.frg.anomalous_dim.AnomalousDimensionSolver`).

References:
    Reuter (1998), Phys. Rev. D 57, 971
    Lauscher & Reuter (2002), Phys. Rev. D 65, 025013
    Litim (2004), Phys. Rev. Lett. 92, 201301 [hep-th/0312114]
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414
    Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035
    D'Angelo, Drago, Pinamonti & Rejzner (2024), Phys. Rev. D 109, 066012
        [2310.20603] (Lorentzian FRG confirmation of Reuter FP)
"""

from sympy import Rational, Symbol, pi

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.frg.anomalous_dim import AnomalousDimensionSolver
from asymsafety.frg.threshold import ThresholdFunctions


def build_eh_beta_system(d: int = 4,
                          gauge: str = "harmonic") -> BetaFunctionSystem:
    """Build the complete Einstein-Hilbert beta function system.

    The beta functions are derived from the Wetterich equation using
    the heat kernel expansion on an S^d background with the optimised
    (Litim) cutoff: single-metric approximation, Type Ia cutoff,
    de Donder (harmonic) gauge with α = 1.

    The d-dimensional closed forms (Reuter 1998 [hep-th/9605030];
    Reuter & Saueressig [0708.1317], Eqs. (4.40)-(4.43)) are

        β_g = (d - 2 + η_N) g
        β_λ = -(2 - η_N) λ + (g/2) (4π)^{1-d/2}
                  × [ 2d(d+1) Φ^1_{d/2}(-2λ) - 8d Φ^1_{d/2}(0)
                      - d(d+1) η_N Φ̃^1_{d/2}(-2λ) ]

    with η_N = g·A(λ)/(1 - g·B(λ)) computed by
    :class:`asymsafety.frg.anomalous_dim.AnomalousDimensionSolver`.
    The graviton volume trace counts all d(d+1)/2 metric modes
    (10 in d=4) and the ghost trace the 2d Grassmann vector modes
    (-8d with the Grassmann factor).

    In d=4 with the Litim regulator this reduces to (x = 1/(1-2λ)):

        β_λ = -(2 - η_N) λ + (g/2π) [5x - 4 - (5/6) η_N x]

    and yields the benchmark NGFP g* ≈ 0.707, λ* ≈ 0.193 with
    θ = 1.475 ± 3.043 i (Litim PRL 92, 201301 [hep-th/0312114];
    Codello, Percacci & Rahmede [0805.2909]). Near the Gaussian FP
    the vacuum-energy flow is positive: ∂β_λ/∂g|_GFP = +1/(2π).

    Args:
        d: Spacetime dimension (d >= 3; the heat-kernel closed forms
            below are the d-dimensional Reuter expressions).
        gauge: Gauge choice (only "harmonic", i.e. de Donder α = 1,
            is implemented).

    Returns:
        BetaFunctionSystem with β_g and β_λ.

    Raises:
        NotImplementedError: If d < 3 or gauge != "harmonic".

    See Also:
        :func:`asymsafety.visualization.phase_portrait.annotated_eh_phase_portrait`
            Publication-grade annotated 2D phase portrait of the
            (g, λ) flow produced by this builder.
        :func:`asymsafety.visualization.phase_portrait.separatrix_overlay`
            Back-integrated UV-critical-surface separatrix overlay.
        :func:`asymsafety.gui.visualization_3d.flow_trajectories_3d`
            3D world-line view of the same flow.
        :mod:`asymsafety.validation.reuter_1998`
            Benchmark fixed-point coordinates ``g* ≈ 0.707, λ* ≈ 0.193``.

    References:
        Reuter (1998), Phys. Rev. D 57, 971 [hep-th/9605030].
        Lauscher & Reuter (2002), Phys. Rev. D 65, 025013 [hep-th/0108040].
        Litim (2001), Phys. Rev. D 64, 105007 [hep-th/0103195].
        Litim (2004), Phys. Rev. Lett. 92, 201301 [hep-th/0312114].
    """
    if d < 3:
        raise NotImplementedError(
            f"build_eh_beta_system requires d >= 3 (got d={d}): the "
            "heat-kernel trace expressions are the d-dimensional Reuter "
            "(1998) closed forms, which degenerate for d <= 2."
        )
    if gauge != "harmonic":
        raise NotImplementedError(
            f"Only the de Donder (harmonic, alpha=1) gauge is implemented "
            f"(got gauge={gauge!r})."
        )

    g = Symbol("g", positive=True)
    lam = Symbol("lambda", real=True)

    tf = ThresholdFunctions()
    solver = AnomalousDimensionSolver(tf)

    # --- Anomalous dimension η_N = g·A(λ) / (1 - g·B(λ)) ---
    # A and B encode the R-projection of the graviton and ghost traces
    # (Type Ia Litim cutoff, de Donder gauge); see AnomalousDimensionSolver.
    A_eta, B_eta = solver.compute_AB_einstein_hilbert(lam, d)
    eta_N = solver.solve(g, A_eta, B_eta)

    # β_g = (d - 2 + η_N) g
    beta_g_expr = (d - 2 + eta_N) * g

    # --- β_λ ---
    # Volume (cosmological constant) projection of the flow:
    #   graviton: d(d+1)/2 modes at mass argument w = -2λ
    #             -> 2d(d+1) Φ^1_{d/2}(-2λ)
    #   ghost:    2d Grassmann vector modes at w = 0
    #             -> -8d Φ^1_{d/2}(0)
    #   η_N insertion from R_k ∝ Z_N on the graviton modes:
    #             -> -d(d+1) η_N Φ̃^1_{d/2}(-2λ)
    w = -2 * lam
    nd2 = Rational(d, 2)
    prefactor = Rational(1, 2) * (4 * pi)**(1 - nd2)

    vol = (2 * d * (d + 1) * tf.Phi(1, nd2, w)
           - 8 * d * tf.Phi(1, nd2, 0)
           - d * (d + 1) * eta_N * tf.Phi_tilde(1, nd2, w))

    beta_lam_expr = -(2 - eta_N) * lam + prefactor * g * vol

    # Build the system
    system = BetaFunctionSystem()
    system.add(BetaFunction(
        coupling_name="g",
        coupling_symbol=g,
        expression=beta_g_expr,
    ))
    system.add(BetaFunction(
        coupling_name="lambda",
        coupling_symbol=lam,
        expression=beta_lam_expr,
    ))

    return system


def eh_fixed_point_litim_d4() -> dict[str, float]:
    """Toolkit-computed NGFP for EH + Litim (Type Ia, de Donder) in d=4.

    These are the fixed-point coordinates and the complex-conjugate
    critical-exponent pair θ = θ' ± i θ'' of the system returned by
    :func:`build_eh_beta_system` (16-digit numerics), consistent with
    Litim, Phys. Rev. Lett. 92, 201301 (2004) [hep-th/0312114] and
    Codello, Percacci & Rahmede [0805.2909]:
    g* ≈ 0.707, λ* ≈ 0.193, θ ≈ 1.475 ± 3.043 i (two relevant
    directions).
    """
    return {
        "g": 0.7073208809868445,
        "lambda": 0.19320050715078566,
        "theta_real": 1.475302425763855,
        "theta_imag": 3.043205846411925,
    }
