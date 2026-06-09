"""Gauge-Higgs analogue: charged fixed point of the 3D Abelian Higgs model.

This module builds a tiny one-loop ``BetaFunctionSystem`` for the
3D charged Abelian-Higgs model (AHM) with ``N`` complex scalar flavors,
and wraps it into the toolkit's :class:`CrossAnalogueBridge`. The
purpose is to demonstrate that the *same* RG / transfer-matrix /
resolvent machinery used for asymptotic-safe gravity also resolves the
*charged fixed point* (CFP) of a 3D stat-mech gauge theory — a
non-trivial interacting UV fixed point in a completely different
physical setting. The CFP is the closest classical statistical-mechanics
analogue of the gravitational NGFP, in that its existence depends on
the matter content (``N > N*``) and on the regulator/scheme.

The beta functions are the textbook one-loop expressions in d = 4 − ε,
obtained from integrating out the gauge field at one loop and
projecting onto the dimensionless ``(α, u, r)`` system::

    α = e² k^{-ε}/(48π²)     gauge coupling (dimensionless)
    u = λ_φ k^{-ε}/(8π²)     quartic self-coupling
    r = m² k^{-2}            scalar mass-squared

    β_α =  -ε α + N α²
    β_u =  -ε u + (N + 4) u² − 6 u α + 9 α²
    β_r = (−2 + (N + 2) u − 6 α) r

The charged fixed point at large ``N`` is

    α*  =  ε / N
    u*  =  (ε / (N + 4)) · (1 + O(N⁻¹))      [the u₊ quartic root]
    r*  =  0

with correlation-length exponent (one-loop)

    ν = 1 / (2 − (N + 2) u* + 6 α*)  →  1/(2 − ε) + O(N⁻¹),

i.e. ν → 1 from below for ε = 1, *increasing* with N — the same
trend as the d = 3 large-N_f field-theory form ν = 1 − C/N_f of
Bonati, Pelissetto & Vicari (2025) [arXiv:2410.05823]. The charged FP
is the *larger* root u₊ of the one-loop quartic condition (Halperin,
Lubensky & Ma 1974); the smaller root u₋ is the tricritical FP with a
second relevant direction.

This is the standard 4 − ε result; for the d = 3 lattice numbers in
Bonati, Pelissetto & Vicari (2025) we set ε = 1, knowing that
absolute agreement with the lattice Monte-Carlo ν cannot be reached
from a perturbative one-loop expansion. The bridge tests therefore
verify only *qualitative trends* (FP existence above N*, monotonic
increase of ν with N toward 1) and the commutativity of the toolkit's
analogue methods on this β-system.

References:
    Bonati, Pelissetto & Vicari (2025), Phys. Rep. [arXiv:2410.05823]
    Halperin, Lubensky & Ma (1974), Phys. Rev. Lett. 32, 292
    Hikami (1980), Prog. Theor. Phys. 64, 1466
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
from sympy import Rational, Symbol

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem

if TYPE_CHECKING:
    from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge


def build_ahm_system(
    N: int,
    Nc: int = 1,
    epsilon: float = 1.0,
) -> BetaFunctionSystem:
    """Build the one-loop charged-AHM beta system in d = 4 − ε.

    Couplings (dimensionless, in conventions where the 1/(48π²) and
    1/(8π²) loop factors are absorbed into ``α`` and ``u``):

    * ``alpha`` — gauge coupling α
    * ``u``    — scalar quartic self-coupling
    * ``r``    — scalar mass-squared (relevant direction)

    Parameters
    ----------
    N
        Number of complex scalar flavors. The charged FP exists for
        ``N > N*`` (the perturbative threshold N* ≈ 22 Nc /3 from the
        one-loop gauge β; the *lattice* threshold in d = 3 is much
        smaller per Bonati et al. (2025)).
    Nc
        Number of "colors" — Nc = 1 is the Abelian Higgs case; values
        > 1 currently only rescale the gauge β-function and are kept
        for interface symmetry with the SU(Nc) analysis in the paper.
    epsilon
        Deviation from d = 4. Set to 1.0 for d = 3 (qualitative only).

    Returns
    -------
    BetaFunctionSystem
        System with three couplings ``alpha``, ``u``, ``r``.
    """
    alpha = Symbol("alpha", positive=True)
    u = Symbol("u", real=True)
    r = Symbol("r", real=True)

    eps = sympy.Float(epsilon)
    N_sym = sympy.Integer(N)
    Nc_sym = sympy.Integer(Nc)

    # Gauge coupling (one-loop): the Nc enters as an SU(Nc) "color
    # multiplicity"; for Nc = 1 this reduces to the AHM expression.
    beta_alpha = -eps * alpha + N_sym * Nc_sym * alpha ** 2

    # Quartic (one-loop, HLM-style): N+4 from scalar bubble, 6 from
    # gauge mixing, 9 from gauge-photon box contribution.
    beta_u = (
        -eps * u
        + (N_sym + 4) * u ** 2
        - 6 * u * alpha
        + 9 * alpha ** 2
    )

    # Relevant mass direction: η_φ from gauge dressing, γ_m² from quartic.
    beta_r = (-2 + (N_sym + 2) * u - 6 * alpha) * r

    system = BetaFunctionSystem()
    system.add(BetaFunction(
        coupling_name="alpha", coupling_symbol=alpha, expression=beta_alpha,
    ))
    system.add(BetaFunction(
        coupling_name="u", coupling_symbol=u, expression=beta_u,
    ))
    system.add(BetaFunction(
        coupling_name="r", coupling_symbol=r, expression=beta_r,
    ))
    return system


def charged_fp_guess(N: int, epsilon: float = 1.0) -> dict[str, float]:
    """Closed-form large-N initial guess for the charged FP.

    From the one-loop expressions, the gauge coupling is fixed by
    ``α* = ε / N`` (for Nc = 1). The quartic ``u*`` solves the
    quadratic ``(N + 4) u² − (ε + 6 α*) u + 9 α*² = 0``, whose two
    roots as α → 0 are ``{0, ε/(N + 4)}``: the branch continuously
    connected to the decoupled Wilson-Fisher FP is the *plus* root

        u₊ = (B + √(B² − 4AC)) / (2A),

    which is the charged (IR-stable, one relevant direction) FP of
    Halperin, Lubensky & Ma (1974), PRL 32, 292. The minus root u₋ is
    the tricritical FP (two relevant directions) and must not be used
    for ν comparisons against Bonati et al. (2025).

    The mass coordinate is exactly zero at any continuous transition.
    """
    eps = float(epsilon)
    alpha_star = eps / N

    A = N + 4
    B = eps + 6 * alpha_star
    C = 9 * alpha_star ** 2
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        # Below N*: charged FP merges into the complex plane — return
        # the bare WF result (gauge decouples) so callers can still
        # exercise the bridge with a meaningful initial guess.
        u_star = eps / (N + 4)
    else:
        u_star = (B + disc ** 0.5) / (2 * A)

    return {"alpha": alpha_star, "u": float(u_star), "r": 0.0}


def correlation_length_exponent(N: int, epsilon: float = 1.0) -> float:
    """One-loop ν at the charged FP.

    Reads off the (positive) eigenvalue along the ``r`` direction at
    the charged FP and inverts:  ν = 1 / θ_r. At the u₊ charged FP,
    ``(N + 2) u₊ → ε`` as N → ∞, so ν → 1/(2 − ε) — i.e. ν → 1 from
    below for ε = 1 — *increasing* with N, the same trend as the
    Bonati, Pelissetto & Vicari (2025) large-N_f form ν = 1 − C/N_f.
    It remains a perturbative one-loop proxy for the lattice MC
    numbers in :mod:`asymsafety.validation.bonati_2025` (agreement at
    N_f ∈ {30, 40, 60} is at the few-percent level, not exact).
    """
    fp_guess = charged_fp_guess(N, epsilon=epsilon)
    # The r-direction eigenvalue is the coefficient of r in β_r, i.e.
    # 2 − (N + 2) u* + 6 α*  (note: θ_r = − ∂β_r/∂r = 2 − (N+2)u* + 6α*).
    theta_r = 2.0 - (N + 2) * fp_guess["u"] + 6 * fp_guess["alpha"]
    if theta_r <= 0:
        # No physical correlation-length scaling; return positive infinity
        # as a sentinel for "first-order / no continuous transition".
        return float("inf")
    return 1.0 / theta_r


class GaugeHiggsAnalogue:
    """Bundle the AHM β-system with its FP and cross-analogue bridge.

    Construct, then access ``.system``, ``.fixed_point``, ``.stability``,
    and ``.bridge`` directly. The bridge can be passed to any code that
    expects a :class:`CrossAnalogueBridge` (e.g.
    ``bridge.verify_commutativity()`` — a linear-algebra regression
    check on the shared stability eigendecomposition; see its docstring).
    """

    def __init__(self, N: int, Nc: int = 1, epsilon: float = 1.0) -> None:
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.stability import analyze_stability
        from asymsafety.transforms.bridge.cross_analogue import (
            CrossAnalogueBridge,
        )

        self.N = N
        self.Nc = Nc
        self.epsilon = epsilon
        self.system = build_ahm_system(N=N, Nc=Nc, epsilon=epsilon)

        finder = FixedPointFinder(self.system)
        guess = charged_fp_guess(N, epsilon=epsilon)
        fp = finder.find_fixed_point(guess, tol=1e-8)
        if fp is None:
            raise RuntimeError(
                f"Could not locate the charged AHM fixed point at "
                f"N={N}, Nc={Nc}, ε={epsilon}. The closed-form guess "
                f"was {guess}; this typically means N is below the "
                f"perturbative threshold N* (gauge β-function loses "
                f"its non-trivial zero)."
            )

        self.fixed_point = fp
        self.stability = analyze_stability(self.system, fp)
        self.bridge: CrossAnalogueBridge = CrossAnalogueBridge(
            self.system, fp, self.stability,
        )

    @property
    def nu(self) -> float:
        """Correlation-length exponent at the charged FP."""
        relevant = self.stability.relevant_eigenvalues
        if relevant.size == 0:
            return float("inf")
        # Largest positive real critical exponent corresponds to the
        # most-relevant direction (the scalar mass r).
        theta = float(relevant.real.max())
        return 1.0 / theta if theta > 0 else float("inf")
