# Physical Scattering Amplitudes in Asymptotic Safety

The `asymsafety.scattering` package computes **graviton-mediated 2 → 2 scalar
scattering amplitudes** that are RG-improved by the asymptotically-safe fixed
point, and compares them against the **physical-scattering bootstrap** of
*Strings from Almost Nothing* (Cheung, Remmen, Sciotti & Tarquini, PRL
`cw4p-cqh7`, [arXiv:2508.09246](https://arxiv.org/abs/2508.09246)).

> **Epistemic status.** This is RG-improvement at the level of an *observable* —
> the same standing as the RG-improved black holes in `asymsafety.cosmology`. It
> is **not** a first-principles momentum-dependent form factor extracted from the
> effective action. A safe fixed point does not by itself guarantee a bounded
> amplitude (Knorr 2026, [arXiv:2602.21285]); the package states this and ships
> the safe-vs-unsafe diagnostic rather than over-claiming.

## The idea

Promote the dimensionless running couplings `g(k), λ(k), ξ(k)` along an
`RGTrajectory` to momentum-dependent dimensionful quantities by identifying the
cutoff with the momentum scale, `k(p²) = ξ·√|p²|`:

```
G(p²) = g(k(p²)) / k(p²)²        f(p²) = G_N / G(p²)        G_N ≡ G(p²→0)
```

The flat-space TT graviton inverse propagator is `Γ⁽²⁾_TT ∝ p²/(32πG)`, so the
tree-level graviton-exchange amplitude is

```
M(s,t,u) = -8π [ G(s)·tu/s + G(t)·su/t + G(u)·st/u ].
```

- **Infrared** (`G → G_N`): the classical Newtonian amplitude — the forward `1/t`
  graviton pole.
- **Ultraviolet** (`g → g*`, so `G(p²) ∝ 1/p²`): each channel tends to a
  constant, so the fixed-angle amplitude approaches a **finite UV value**. This
  softening restores tree-level unitarity: classical gravity's partial waves grow
  past `|a_ℓ| = 1` near the Planck scale, the dressed ones stay bounded.

## Package layout

| Module | Contents |
|--------|----------|
| `scale.py` | `MomentumScale` ABC; `EnergyScale`, `TransferScale`, `FixedScale` |
| `kinematics.py` | `Mandelstam` (closure `s+t+u=4m²`), angle/momentum helpers |
| `form_factor.py` | `GravitonFormFactor` — `G_of_psq`, `Lambda_of_psq`, `xi_of_psq`, `f` |
| `propagator.py` | `DressedGravitonPropagator` — running coupling strength / denominator |
| `amplitude.py` | `GravitonMediatedAmplitude` — `eval`, `amplitude_vs_s`, `ir_limit`, `uv_limit` |
| `consistency.py` | partial waves, unitarity, UV-finiteness, Froissart, no-ghost, crossing |
| `bootstrap.py` | `veneziano`, `virasoro_shapiro`, `mass_spectrum`, residues, `ultrasoft_falloff`, `StringAmplitude` |
| `bridge.py` | `ScatteringBridge` — runs the same battery on both; `verify()` |

## Quick start

```python
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.bridge import ScatteringBridge

system = build_eh_beta_system(d=4)
fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
traj = FlowIntegrator(system).integrate(
    {"g": fp.location["g"] - 1e-3, "lambda": fp.location["lambda"] + 1e-3},
    t_span=(10.0, -10.0), max_step=0.05,
)

amp = GravitonMediatedAmplitude(GravitonFormFactor(traj))
print(amp.ir_limit())     # ratio → 1 (recovers GR)
print(amp.uv_limit())     # bounded → True

verdict = ScatteringBridge(amp).verify()
print(verdict["summary"])
```

## Command line

```bash
asymsafety amplitude --truncation eh --guess g=0.7,lambda=0.14 \
    --s-range 1e-2:1e8:200 --angle 0.3 --checks --compare-string --output amp.npz
```

`--scale {energy,fixed}` chooses the momentum-scale identification (`fixed`
freezes the couplings, reproducing the *unsafe* growing amplitude). The output
bundles `s`, `abs_M_as`, `abs_M_gr`, `form_factor`, plus consistency/bridge flags
in the metadata.

## The bridge: two routes to UV completeness

`ScatteringBridge.verify()` records that the asymptotically-safe amplitude passes
the **foundational** physical-scattering requirements (crossing, UV finiteness,
no ghosts, bounded partial waves) but reaches UV completeness by **softening**
the graviton coupling — a **UV-constant** amplitude — whereas the string
bootstrap enforces **ultrasoft** (super-polynomial) falloff via an infinite
higher-spin Regge tower. They are distinct, mutually consistent points in the
space of physical amplitudes.

## Figures

`scripts/generate_figures.py` produces `amplitude_vs_energy`,
`graviton_form_factor`, `partial_wave_unitarity`, `regge_trajectory`, and
`as_vs_string` (see `asymsafety.visualization.amplitude_plot`).

## Validation

`asymsafety.validation.{draper_2020, knorr_2026, cheung_2025}` reproduce,
respectively, the IR/UV limits and ghost-freedom (Draper et al. 2020), the
safe-vs-unsafe boundedness dichotomy (Knorr 2026), and the bootstrap facts —
Regge spectrum, crossing, higher-spin cancellation, ultrasoft falloff (Cheung et
al. 2025). They are exercised by `tests/test_benchmarks_published.py`.

## References

- T. Draper, B. Knorr, C. Ripken & F. Saueressig, *Graviton-Mediated Scattering
  Amplitudes from the Quantum Effective Action*, PRL **125**, 181301 (2020)
  [[2007.04396](https://arxiv.org/abs/2007.04396)].
- B. Knorr, C. Ripken & F. Saueressig, *Form Factors in Asymptotically Safe
  Quantum Gravity* (2023) [[2210.16072](https://arxiv.org/abs/2210.16072)].
- B. Knorr, *Asymptotically (un)safe scattering amplitudes from scratch* (2026)
  [[2602.21285](https://arxiv.org/abs/2602.21285)].
- C. Cheung, G. N. Remmen, F. Sciotti & M. Tarquini, *Strings from Almost
  Nothing*, PRL (2025) [[2508.09246](https://arxiv.org/abs/2508.09246)].
