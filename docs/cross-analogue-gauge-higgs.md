# Cross-analogue: 3D gauge-Higgs ↔ asymptotic safety

This note documents the cross-disciplinary bridge between the
toolkit's gravity-side asymptotic-safety machinery and the 3D
gauge-Higgs / Abelian-Higgs review by

> Bonati, Pelissetto & Vicari (2025),
> *"Three-dimensional Abelian and non-Abelian gauge-Higgs theories"*,
> Phys. Rep. [[arXiv:2410.05823](https://arxiv.org/abs/2410.05823)].

The headline overlap is the existence of an *interacting UV fixed point*
controlling a continuous transition whose presence depends on matter
content and on the regulator/scheme:

- **gravity**: the Reuter NGFP at `(g*, λ*) ≈ (0.69, 0.14)` in the
  Einstein-Hilbert truncation; existence is regulator-dependent and
  shifts with matter content (`N_s`, `N_v`, `N_D`);
- **3D gauge-Higgs**: the *charged fixed point* (CFP) of SU(N_c) +
  `N_f` complex scalars, governing Coulomb-to-Higgs transitions for
  `N_f > N_f^*`; the threshold is `~ 375` in the 4-ε expansion but
  much smaller in d = 3 on the lattice.

Both fixed points are interacting (non-Gaussian); both organise a
non-trivial RG flow with a *critical surface* of relevant directions;
both have a quantitative existence boundary in the space of matter
contents. The toolkit's
[`CrossAnalogueBridge`](../src/asymsafety/transforms/bridge/cross_analogue.py)
already mediates between the RG / hydraulic / quantum / integral-
transform representations of a fixed point — the AHM analogue
[`GaugeHiggsAnalogue`](../src/asymsafety/transforms/bridge/gauge_higgs.py)
slots into the same vocabulary as a new node.

## Concept dictionary

| Asymptotic safety (gravity)                         | 3D Abelian-Higgs / charged FP                              |
|-----------------------------------------------------|------------------------------------------------------------|
| NGFP `(g*, λ*) ≈ (0.69, 0.14)` (Reuter 1998)        | Charged FP `(α*, u*) = (ε/N, ε/(N+4)) + O(1/N²)` (HLM 1974, Bonati 2025) |
| Critical exponents `θ_i = −eig(∂β/∂g)`              | Correlation-length exponent `ν = 1/θ_r`                    |
| `θ_1, θ_2 ≈ 1.5 ± 3.0 i` (2 relevant directions)    | 1 strongly-relevant direction (mass), `θ_r ≈ 2 − O(1/N)`   |
| Matter content `(N_s, N_v, N_D)` shifts NGFP        | Flavor count `N_f` shifts CFP location and existence       |
| NGFP existence shifts with matter (Eichhorn–Schiffer 2022, Korver 2024) | CFP exists only for `N_f > N_f^*` (Bonati 2025) |
| Regulator/scheme dependence: Litim vs Exponential vs sharp | Scheme dependence: 4-ε vs lattice MC vs large-`N_f` |
| Critical surface = UV-attractive subset of theory space | Critical surface = (`u`, `r`)-plane at `α = α*`         |
| `Tr` over heat kernel on `S^4`                       | `Tr` over lattice operator (transfer matrix on `Z^3`)     |
| Wilson-loop holonomy (gauge sector of matter coupling) | Polyakov / Wilson loop = order parameter (Bonati §6)    |
| RG-improved black-hole horizon at `k = k(r)`        | Correlation length `ξ = 1 / |r|^ν` at the transition       |

## Formal mapping (commutative diagram)

The bridge enforces equality of critical exponents along three
toolkit paths — *direct RG stability analysis*, *transfer matrix
exponents*, and *resolvent poles* — for **any** β-system that fits
into the `BetaFunctionSystem` interface. The AHM analogue therefore
extends the existing commutative diagram

```
   classical RG  ──── Laplace / Mellin ────  integral transforms
        ↕                                              ↕
   transfer matrix  ────  Koopman / resolvent  ────  quantum
```

with a new vertical edge: the *charged-FP* node maps onto the
classical-RG box, then propagates around the diagram. The shipped
test in
[`tests/test_gauge_higgs_analogue.py`](../tests/test_gauge_higgs_analogue.py)
asserts that all three paths agree on the AHM critical exponents to
`tol = 0.2`, exactly as `tests/test_bridge.py` does for the gravity
NGFP.

## Toolkit implementation

The one-loop 4-ε charged-AHM β-system carries three couplings:

- `α` (gauge coupling),
- `u` (scalar quartic),
- `r` (scalar mass-squared, the relevant direction).

In matter-coupling conventions where `1/(48π²)` and `1/(8π²)` loop
factors are absorbed into `α` and `u`:

```
β_α =  −ε α + N α²
β_u =  −ε u + (N + 4) u² − 6 u α + 9 α²
β_r =  (−2 + (N + 2) u − 6 α) r
```

At one loop:

- charged FP at `α* = ε/N`, `u* = (B + √disc)/(2A)` — the root
  continuously connected to Wilson–Fisher as the gauge coupling is
  switched off — with `r* = 0`;
- one strongly-relevant direction (mass), correlation-length exponent
  `ν = 1/(2 − (N+2) u* + 6 α*)`;
- grows with `N_f` toward the large-`N_f` limit `ν → 1` from below.

The one-loop 4-ε values at `ε = 1` track the Bonati lattice MC points
(`ν = 0.64, 0.745, 0.81` at `N_f = 30, 40, 60`) to a few percent and
the published large-`N_f` formula `ν = 1 − 9.727/N_f` (SU(2)) to
≈1.5%. Below the one-loop threshold `N* ≈ 28` the charged FP does not
exist and the builder falls back to the Wilson–Fisher root. The
cross-analogue figure [`nu_vs_nf`](images/nu_vs_nf.caption.md)
overlays the one-loop curve, the large-`N_f` asymptote, and the
lattice points.

## What the validation module does

[`asymsafety.validation.bonati_2025`](../src/asymsafety/validation/bonati_2025.py)
mirrors the convention of `reuter_1998.py`:

- `BONATI_SU2_MC` — lattice MC `(β_c, ν)` at `N_f ∈ {30, 40, 60}` with
  published error bars;
- `BONATI_LARGE_NF` — the SU(2) coefficient `C = 9.727` of the
  `ν = 1 − C/N_f` field-theory prediction;
- `CHARGED_FP_THRESHOLDS` — `N_f^* ≈ 375` in 4-ε, `< 30` in d = 3;
- `validate_charged_fp_existence`, `validate_nu_vs_nf`,
  `large_nf_nu` — helpers consumed by
  [`tests/test_benchmarks_published.py::TestBonati2025GaugeHiggs`](../tests/test_benchmarks_published.py).

The bridge module and the validation module are independently
useful: the bridge runs the toolkit's perturbative AHM β-system and
produces critical exponents; the validation module stores the
published lattice numbers for comparison plots and pedagogy.

## Why this entry exists in a gravity toolkit

The cross-analogue bridge is a load-bearing piece of the toolkit's
architecture (see `transforms/bridge/cross_analogue.py` and the
[hydraulic / quantum / transform](../README.md#physical-computational-analogues)
sections of the README). Every analogue node it ships is a different
*window* on the same RG-flow physics. The AHM charged FP is the
*closest stat-mech analogue* of the gravitational NGFP, and the rigour
of the Bonati review — combining Monte Carlo, large-`N_f` field
theory, and ε-expansion — provides a worked example of cross-method
triangulation that the gravity-side computations have only recently
begun to match.

## See also

- [`tests/test_gauge_higgs_analogue.py`](../tests/test_gauge_higgs_analogue.py) — end-to-end commutativity check.
- [`tests/test_benchmarks_published.py`](../tests/test_benchmarks_published.py) — `TestBonati2025GaugeHiggs` class.
- [`scripts/generate_figures.py`](../scripts/generate_figures.py) — `gen_ahm_*` and `gen_nu_vs_nf` / `gen_charged_fp_boundary`.
- [`scripts/generate_animations.py`](../scripts/generate_animations.py) — `gen_anim_rg_flow_ahm`, `gen_anim_nu_vs_nf_sweep`.
- [`docs/LITERATURE.md`](LITERATURE.md#cross-disciplinary-analogues) — §8 *Cross-Disciplinary Analogues*.
