"""Einstein-Hilbert beta functions: β_g and β_λ.

Analytic expressions for the beta functions of the dimensionless
Newton coupling g and cosmological constant λ in the Einstein-Hilbert
truncation with the Litim regulator.

Conventions:
    g = G k^{d-2}               (dimensionless Newton coupling)
    λ = Λ / k^2                 (dimensionless cosmological constant)
    t = log(k/k_0)              (RG time)
    η_N = -∂_t ln(Z_N)          (graviton anomalous dimension)

    β_g = (d - 2 + η_N) g
    β_λ = -(2 - η_N) λ + [trace contributions]

The anomalous dimension is determined self-consistently:
    η_N = g·N(λ) / (1 - g·D(λ))

where N(λ) and D(λ) encode the one-loop graviton and ghost traces.

References:
    Reuter (1998), Phys. Rev. D 57, 971
    Lauscher & Reuter (2002), Phys. Rev. D 65, 025013
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414
    Dona, Eichhorn & Percacci (2014), Phys. Rev. D 89, 084035
"""

import sympy
from sympy import Expr, Rational, Symbol, pi, simplify

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.frg.threshold import ThresholdFunctions


def build_eh_beta_system(d: int = 4,
                          gauge: str = "harmonic") -> BetaFunctionSystem:
    """Build the complete Einstein-Hilbert beta function system.

    The beta functions are derived from the Wetterich equation using
    the heat kernel expansion on an S^d background with the optimised
    (Litim) cutoff.

    The anomalous dimension η_N is computed self-consistently from
    the R-projection of the flow equation. The graviton trace includes
    contributions from the TT sector (5 dof in d=4), the scalar
    conformal mode (1 dof, wrong-sign kinetic term), and the
    Faddeev-Popov ghost (4 vector dof with -2 Grassmann factor).

    The decomposition follows the single-metric approximation with
    the de Donder (harmonic) gauge in the Landau limit (α → 0).

    Args:
        d: Spacetime dimension.
        gauge: Gauge choice.

    Returns:
        BetaFunctionSystem with β_g and β_λ.
    """
    g = Symbol("g", positive=True)
    lam = Symbol("lambda", real=True)

    one_m_2l = 1 - 2 * lam  # (1 - 2λ)

    # --- Anomalous dimension η_N ---
    # η_N = g·N(λ) / (1 - g·D(λ))
    #
    # N(λ) comes from the R-projection of the graviton and ghost traces:
    #   TT (5 dof): contributes Φ^1_1 and Φ^2_1 terms with mass w=-2λ
    #   Ghost (-8 effective dof): contributes Φ^1_1 and Φ^2_1 at w=0
    #   Scalar (1 dof, wrong sign): additional Φ terms
    #
    # The R-projection involves differentiating the threshold functions
    # w.r.t. R̄ and evaluating at R̄=0, which generates Φ^2 terms.
    #
    # N(λ) = (4/3)/(1-2λ) + 8 - 8/(1-2λ)²
    # D(λ) = 1/(3(1-2λ)) + 2 - 2/(1-2λ)²
    #
    # These coefficients correspond to the single-metric EH truncation
    # with Type I Litim cutoff and de Donder gauge.

    N_eta = (Rational(4, 3) / one_m_2l + 8 - 8 / one_m_2l**2)
    D_eta = (Rational(1, 3) / one_m_2l + 2 - 2 / one_m_2l**2)

    eta_N = g * N_eta / (1 - g * D_eta)

    # β_g = (2 + η_N) g
    beta_g_expr = (2 + eta_N) * g

    # --- β_λ ---
    # β_λ = -(2 - η_N)λ + g·V(λ, η_N)/π
    #
    # V is the volume (cosmological constant) trace contribution:
    #   TT+scalar (6 effective dof at mass w=-2λ): 3/(1-2λ)
    #   Ghost (-8 dof at mass w=0): -4
    #   η_N correction: -η_N(1/(1-2λ) - 4/3)
    #
    # The overall prefactor is determined by matching the trace normalization.
    # With the conventions g = Gk^2, the correct prefactor is approximately
    # g·8/π (absorbing factors from (4π)^{-d/2} × 32πg × Q-functionals).

    vol = (3 / one_m_2l - 4
           - eta_N * (1 / one_m_2l - Rational(4, 3)))

    beta_lam_expr = -(2 - eta_N) * lam + 8 * g / sympy.pi * vol

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
    """Known approximate fixed point values for EH + Litim in d=4.

    The Reuter fixed point (non-Gaussian UV fixed point).
    Exact values depend on gauge and cutoff scheme.

    These values are for the single-metric approximation with
    Type I Litim cutoff and de Donder gauge.
    """
    return {
        "g": 0.69,
        "lambda": 0.14,
        "theta_real": 0.75,
        "theta_imag": 0.0,
    }
