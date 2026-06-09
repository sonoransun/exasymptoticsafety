"""Quadratic gravity beta functions: β_α, β_β for R² and C² couplings.

At one loop the runnings of the four-derivative coefficients are pure
numbers: they come from the b₄ (Gilkey–Seeley–DeWitt) heat-kernel
coefficient of the fluctuation operator projected onto R² and C², and
are gauge- and scheme-independent (no regulator-threshold or λ
dependence survives at this order).

In the standard inverse parametrization of higher-derivative gravity
(Fradkin & Tseytlin 1982; Avramidi & Barvinsky 1985; Codello & Percacci
[hep-th/0607128]),

    S ⊃ ∫ d⁴x √g [ (1/(2λ_C)) C² − (ω/(3λ_C)) R² + (θ/λ_C) E ],

the one-loop results read

    β_{λ_C} = −(1/16π²) (133/10) λ_C²,
    β_ω     = −(1/16π²) (25 + 1098 ω + 200 ω²)/60 · λ_C,
    β_θ     = +(1/16π²) 7(56 − 171 θ)/90 · λ_C.

Converting to the *coefficient* basis used by the toolkit action
(``actions/quadratic.py``: Γ ⊃ α R² + β C², so β ≡ f_C2 = 1/(2λ_C) and
α ≡ f_R2 = −ω/(3λ_C)):

    ∂_t β = −β_{λ_C}/(2λ_C²)        = +(1/16π²) (133/20)   (exact),
    ∂_t α |_{α=0} = −β_ω(ω=0)/(3λ_C) = +(1/16π²) (5/36),
    ∂_t f_E = +(1/16π²) (196/45)     (Gauss–Bonnet; *not* in this
                                      truncation — it must not appear
                                      in β_α or β_β).

Because both runnings are nonzero constants, the one-loop system has
**no interior fixed point** for (α, β): the couplings run
logarithmically forever. The positive constant ∂_t β is asymptotic
freedom of the Weyl-squared sector — the C² coefficient grows toward
the UV, i.e. the inverse coupling λ_C = 1/(2β) → 0⁺, exactly analogous
to a non-abelian gauge coupling.

References:
    Fradkin & Tseytlin (1982), Nucl. Phys. B201, 469
    Avramidi & Barvinsky (1985), Phys. Lett. B159, 269
    Codello & Percacci (2006) [hep-th/0607128]
    Codello, Percacci & Rahmede (2009), Ann. Phys. 324, 414 [0812.0785]
    Fehre, Litim, Sherrill & Sherrill (2023), [2311.12097]
        (momentum-dependent field redefs remove ghost poles)
"""

from sympy import Rational, Symbol, pi

from asymsafety.beta.system import BetaFunction, BetaFunctionSystem
from asymsafety.beta.einstein_hilbert import build_eh_beta_system


def build_quadratic_beta_system(d: int = 4) -> BetaFunctionSystem:
    """Build the quadratic gravity beta function system.

    The system holds 4 couplings (g, λ, α, β): the Einstein–Hilbert
    pair (g, λ) flows as in :func:`build_eh_beta_system`, while the
    four-derivative coefficients α (R²) and β (C²) run by the exact
    one-loop universal constants (d=4)

        β_α = +(1/16π²) · 5/36,
        β_β = +(1/16π²) · 133/20,

    independent of *all* couplings (see the module docstring for the
    derivation and sources).

    Consequently this truncation has **no interior NGFP**: β_α and β_β
    are nonzero constants, so no point in (g, λ, α, β) makes all four
    beta functions vanish. β_β > 0 is asymptotic freedom of the C²
    sector in the coefficient convention (β → +∞ logarithmically,
    λ_C = 1/(2β) → 0⁺). Literature NGFP coordinates such as
    Codello–Percacci–Rahmede (2009) come from the full nonperturbative
    FRG calculation and are *external reference values only* — they are
    not fixed points of this one-loop system.

    Args:
        d: Spacetime dimension. The α/β coefficients are the d=4
            one-loop universals.

    Returns:
        BetaFunctionSystem with β_g, β_λ, β_α, β_β.

    See Also:
        :func:`asymsafety.visualization.phase_portrait.quadratic_pairwise_grid`
            2x3 grid of pairwise phase portraits over the four
            coupling axes (with frozen-coupling annotation).
        :mod:`asymsafety.validation.codello_2009`
            One-loop universal coefficients (reproduced exactly here)
            and the CPR literature fixed-point coordinates (external
            reference values, not reproduced by this truncation).

    References:
        Fradkin & Tseytlin (1982), Nucl. Phys. B201, 469.
        Avramidi & Barvinsky (1985), Phys. Lett. B159, 269.
        Codello & Percacci (2006) [hep-th/0607128].
        Codello, Percacci & Rahmede (2009),
            Ann. Phys. 324, 414 [0812.0785].
    """
    # Start with the EH system for g and λ
    eh_system = build_eh_beta_system(d)

    alpha = Symbol("alpha", real=True)
    beta_ = Symbol("beta", real=True)

    # β_α: running of the R² coefficient (α ≡ f_R2 = −ω/(3λ_C)).
    # From the b_4 heat-kernel coefficient projected onto R²:
    # β_ω(ω=0) = −(1/16π²)(25/60) λ_C in the Avramidi–Barvinsky
    # parametrization, hence ∂_t f_R2 |_{f_R2=0} = −β_ω(0)/(3λ_C)
    # = +(1/16π²)·5/36 (Avramidi–Barvinsky 1985; Codello–Percacci
    # hep-th/0607128). Truncation: only the constant ω=0 piece is
    # kept — the full one-loop β_ω carries (25 + 1098ω + 200ω²)/60
    # with ω = −3α/(2β), which is dropped here.
    beta_alpha_expr = Rational(5, 36) / (16 * pi**2)

    # β_β: running of the C² coefficient (β ≡ f_C2 = 1/(2λ_C)).
    # From the b_4 coefficient projected onto C²:
    # β_{λ_C} = −(1/16π²)(133/10) λ_C² (Fradkin–Tseytlin 1982;
    # Avramidi–Barvinsky 1985), hence ∂_t f_C2 = −β_{λ_C}/(2λ_C²)
    # = +(1/16π²)·133/20, exact at one loop. The POSITIVE sign is
    # asymptotic freedom in this coefficient convention: β → +∞
    # logarithmically, λ_C = 1/(2β) → 0⁺ in the UV, analogous to a
    # non-abelian gauge coupling.
    beta_beta_expr = Rational(133, 20) / (16 * pi**2)

    # Build complete system
    system = BetaFunctionSystem()

    # Add EH betas (modified by α, β back-reaction in principle)
    for name in eh_system.coupling_names:
        system.add(eh_system.beta(name))

    # Add quadratic betas
    system.add(BetaFunction(
        coupling_name="alpha",
        coupling_symbol=alpha,
        expression=beta_alpha_expr,
    ))
    system.add(BetaFunction(
        coupling_name="beta",
        coupling_symbol=beta_,
        expression=beta_beta_expr,
    ))

    return system
