# Evaluation Report: Mathematical Models of `asymsafety`

**Date:** 2026-06-09 · **Scope:** every quantitative claim in `src/asymsafety/` — FRG beta functions, fixed points, critical exponents, spectral/heat-kernel machinery, RG-improved observables, string-bootstrap comparison, cross-analogue bridges — plus the visual-simulation layer (52 figures, 2 animations) and end-to-end capability demonstrations.
**Method:** blind ground-truth derivation (4 independent agents, no code access, literature fetched from arXiv) → 46 executed audit checks against the code → adversarial skeptic defense of every contested finding → referee adjudication with independent reproduction of every BUG (28/28 spot-checks). In parallel: full test/notebook campaign, figure generation with image-level physics review against computed anchors, and 10 capability demos. Per user decision, all findings adjudicated **BUG (unambiguous mathematics)** were then fixed on branch `evaluation-fixes` and re-verified against their condemning checks.

---

## 1. Executive summary

**Verdict before fixes:** the toolkit was well-engineered software wrapped around a structurally incorrect physics core. The Einstein–Hilbert beta functions — the foundation of every truncation, flow, figure, observable, and "literature benchmark" — counted 6 graviton modes instead of 10 and flipped the sign of the Gaussian-fixed-point cosmological-constant slope. As a result the flagship Reuter fixed point sat at (g\*, λ\*) = (0.694, 0.142) with **real** critical exponents {+0.75, −29.9}, one relevant direction, and was **not UV-attractive** — where every published scheme gives the complex pair θ = 1.475 ± 3.043i with two relevant directions. The validation layer largely certified failures: `reuter_1998` failed 4/5 of its own checks on actual toolkit output, the quadratic and foliated "benchmarks" only ever compared literature dictionaries to themselves (neither truncation possesses the NGFP it advertises), and the test suite hard-pinned the bugged fixed point at rtol 1e-6.

**Audit outcome (46 checks):** 15 BUG · 10 MISLEADING-CLAIM · 9 INTERNAL-INCONSISTENCY · 9 VERIFIED-CORRECT · 2 CORRECT-WITH-DOCUMENTED-APPROXIMATION · 1 DISCREPANT-VS-LITERATURE. Severity: 12 High / 18 Medium / 16 Low. Every BUG verdict carries a minimal reproducer plus a citation or self-contained proof; every contested finding survived (or was acquitted by) an explicit skeptic defense. Full records: `evaluation/audit/findings.json`, `evaluation/audit/audit-summary.md`.

**Verdict after fixes:** with all 15 BUGs (plus the misleading-claim and consistency items) repaired, the toolkit now reproduces the published physics it claims:

| Quantity | Before | After | Literature |
|---|---|---|---|
| EH NGFP (g\*, λ\*) | (0.6937, 0.1423) | **(0.70732, 0.19320)** | ≈(0.707, 0.193) (Litim cutoff) |
| g\*λ\* (scheme-robust) | 0.0987 (28% off) | **0.13665** | ≈0.137 |
| Critical exponents θ | +0.75, −29.9 (real) | **1.4753 ± 3.0432i** | 1.475 ± 3.043i |
| Relevant directions | 1 (not UV-attractive) | **2 (UV-attractive)** | 2 |
| `reuter_1998` self-validation | 4/5 checks fail | **5/5 pass** | — |
| Matter trends vs Donà–Eichhorn–Percacci | scalar sign inverted | **match (per-field weights exact)** | 1311.2898 |
| ν(N_f) vs Bonati 2025 lattice | decreasing toward ½ (wrong root) | **increasing toward 1, 1.5–4% off lattice** | 0.64/0.745/0.81 |
| Virasoro–Shapiro amplitude | identically 1 (massless); 3-pole rational | **genuine Regge tower, ultrasoft falloff** | exact Γ-identities |
| Amplitude CLI IR/GR ratio | 0.245 with `all_passed=True` | **1.000 with honest roll-up** | →1 (Newtonian) |
| Full test suite | 478/480 (pins locked to bugged values) | **568/568** (pins regenerated, assertions re-tightened) | — |

**Top three strengths:** (1) the numerical/engineering layer is genuinely solid — Litim threshold closed forms, the η_N algebraic closure, fixed-point/stability/flow machinery, serialization, batch backends, and the figure pipeline all verified correct; (2) honest epistemic caveats in the scattering/cosmology docstrings (RG-improvement vs first principles); (3) the architecture made the repair tractable — beta builders are isolated, symbolic, and testable.

**Top three defects found (all now fixed or honestly documented):** (1) the EH core mode-counting/sign errors; (2) a self-certifying validation layer (benchmarks that compare literature to itself, a commutativity check comparing an eigendecomposition with itself at 10% tolerance); (3) advertised NGFPs in the quadratic and foliated truncations that the implementations do not possess.

---

## 2. Environment and evidence base

- Python 3.12.3 venv (`.venv`), sympy 1.14, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9, qiskit 2.4.1; jax 0.10.1 + h5py 3.16.0 installed during evaluation. PySide6 absent (GUI exercised only via its matplotlib-importable `visualization_3d` module). ffmpeg present.
- Repository placed under git during evaluation: pristine snapshot `c2b56f6`; evaluation artifacts `c8d1bff`; campaign fixes `91a546e`; adjudicated physics fixes `4c1e281` (branch `evaluation-fixes`).
- All artifacts under `evaluation/`: `audit/` (findings.json, audit-summary.md, ~30 reproducer scripts in four domains + defense + referee), `demos/` (10 capability demos with logs), `figures/` + `figures-after/` (pre/post-fix renders), `animations/`, `logs/` (pytest, nbmake, figure logs, figure verdicts, fix results).

## 3. Mathematical-model audit (Phase 1)

Ground-truth hierarchy: re-derivable closed-form math (sympy/mpmath) > exact representation theory > published scheme-independent universals > same-scheme published values > cross-scheme values > internal consistency. The test suite was treated strictly as a regression lock, never as ground truth — correctly so, since it pinned the toolkit's own (incorrect) outputs.

### Headline defects (all referee-confirmed with reproducers)

1. **EH beta functions structurally wrong** (`beta/einstein_hilbert.py`, BUG, High). Graviton volume trace counted 6 modes instead of 10 while keeping the full ghost −8; η_N numerator/denominator matched no derivation; ∂β_λ/∂g at the Gaussian FP was −8/π instead of +1/(2π) — graviton loops generated *negative* vacuum-energy flow. Direct consequences: λ\* 26% off, real exponents, one relevant direction, and a fixed point that repelled UV flow in one direction.
2. **Matter sector self-contradiction + inverted DEP sign** (`frg/anomalous_dim.py`, `actions/matter.py`, `beta/matter.py`, High). Three different η_N implementations; the `eh_matter` builder's zero-matter limit had *no fixed point at all*; the scalar weight sign was flipped versus the cited Donà–Eichhorn–Percacci paper, inverting the headline matter-bounds physics.
3. **"Virasoro–Shapiro" wasn't** (`scattering/bootstrap.py`, High). With massless kinematics the implemented function was identically 1; at shipped defaults, a 3-pole rational function with a power-law tail — no Regge tower, not ultrasoft. The flagship AS-vs-strings comparison was comparing AS against a rational function, and the `cheung_2025` "ultrasoft" validation passed only via pole-contaminated fit windows.
4. **Quadratic-gravity universals misassigned** (`beta/quadratic.py`, High). λ→0 limits were 67/180 and −329/90 (×1/16π²), contradicting the file's own claimed "exact one-loop universals" (53/45, −196/45), and neither matched the true coefficient-basis values (+5/36, +133/20): the C² invariant was attached to the R² coupling and 3× the Gauss–Bonnet universal contaminated β_α.
5. **Foliated narrative unrealized and misattributed** (High/Medium). The implemented foliated system admits no physical NGFP (η = −2 unreachable for λ > −1/2); the advertised benchmark (0.96, 0.20, 1) is not a root of the system (β = (3.26, 0.18, 0) there); the cited Manrique–Rechenberger–Saueressig paper actually gives (λ\*, g\*) = (0.19, 0.31) and contains no λ_ADM coupling; the λ_ADM = 1 plane is UV-*repulsive* at physical λ (the repo's own test was sign-blind).
6. **Gauge-Higgs wrong quartic root** (`transforms/bridge/gauge_higgs.py:157`, High). `(B−√disc)/(2A)` instead of `(B+√disc)/(2A)`: ν(N_f) decreased toward ½ instead of increasing toward 1, and the Bonati-2025 lattice comparison failed; the test had been loosened to accept "either monotone direction."
7. **Spectral/heat-kernel layer: six independent math errors, all off the beta-function path** (Medium). S⁴ scalar degeneracy `(l+1)²(l+2)²/4` (degree-4 — impossible by the Weyl law) vs the exact `(2l+3)(l+1)(l+2)/6`; transverse-vector degeneracy giving 15 instead of 10 Killing vectors at l=1; `trace_on_S1xS3` crashing on every call via an always-true `hasattr` guard; non-integer S³ TT multiplicities; TT b₂ matching no operator; Ω² bundle-curvature signs flipped in b₄; the exponential-regulator threshold integral returning NaN on every call *and* implementing a wrong ∂ₜR_k.
8. **Self-certifying validation** (High/Medium). `reuter_1998` failed 4/5 of its own checks on real output (the test suite deliberately avoided asserting `all_passed`); the quadratic/foliated "benchmark" tests asserted properties of literature dictionaries, never of computed physics; `korver_2024` misquoted its cited paper's bounds (12, 6 vs the actual wedge N_s + 6.4N_v ≈ 23.1) with a validator boolean that could not fail in its claimed regime; `verify_commutativity` compared an eigendecomposition with itself at a tolerance 11 orders looser than the actual deviation.

### Verified correct (the audit's positive findings)

Litim threshold functions Φᵖₙ, Φ̃ᵖₙ and `QFunctional` exact to ~1e-16 with all identities; η_N algebraic closure exact (η_N(FP) = −2 to 1e-13); Gaussian-FP structure exactly canonical; the Veneziano amplitude exact (1.8e-15 vs mpmath, crossing/poles/Regge verified); the RG-improved graviton amplitude exactly crossing-symmetric with correct IR/GR limit and the safe/unsafe (Knorr) dichotomy matching the analytic −8πg\* plateau to 7 digits; RG-improved Schwarzschild horizon structure and critical mass reproducing the Bonanno–Reuter closed form to ~3e-4; two honestly documented approximations (partial-wave angular cutoff; hand-inserted λ_ADM plane).

## 4. Test and notebook campaign (Phase 2)

Pre-fix: 480 collected, **478 passed**; both failures triaged — one genuine test-isolation defect (persistent `~/.cache/asymsafety` satisfied a disk-cache test from a previous run; now isolated via `ASYMSAFETY_CACHE_DIR` monkeypatch) and one CPU-contention timeout (passes in isolation). All 6 tutorial notebooks passed under nbmake. All 10 sha256 figure baselines passed under matplotlib 3.10.9 — proving the `<3.10` pyproject pin was stale metadata (now bumped to `<3.11`).

Calibration finding: the benchmark suite pinned *toolkit self-consistency*, not literature accuracy — `g*=0.6936584729648413` at rtol 1e-6 alongside comments declining to assert the validation module's own verdict. Post-fix: pins regenerated, loosened assertions re-tightened (sign-aware foliated test, strictly-increasing ν, `all_passed` asserted), suite expanded by the fix clusters' new tests: **568 passed, 0 failed**.

## 5. Visual simulations (Phase 3)

Generation: 50–52/52 figures and 2/2 ffmpeg animations rendered without errors, pre- and post-fix; 5/5 random render-sanity checks OK; both animations show clean monotone progression. Image-level physics review of 15 curated figures against demo-computed anchors:

**Pre-fix: 5 PASS / 10 FAIL.** The failures traced directly to the audit's defects, in two classes. *Honest-but-broken physics:* `running_couplings` showed runaway trajectories with no g\* plateau; the three cosmology figures (`lapse_with_horizons`, `classical_vs_rg_lapse`, `hawking_temperature`) showed lapses falling to −∞ and monotone T_H because integrated trajectories never reach a classical IR; `eh_phase_portrait`'s "UV-critical separatrix" was integrated in the wrong direction and never connected NGFP→GFP, while its caption claimed two relevant directions against the figure's own (then-correct) "1 rel." annotation. *Fabricated reference physics:* `foliated_3d` drew an "NGFP (2 rel.)" star at g≈0.45 that the system does not possess; `quadratic_pairwise` anchored slices at nonexistent FP coordinates without provenance; `partial_wave_unitarity` drew both curves above the |a_l|=1 bound everywhere while its caption claimed a crossing.

**Post-fix re-read of the 10 failures: see §7.** Verdicts: `evaluation/logs/figure-verdicts.json` (pre-fix), `evaluation/logs/figure-verdicts-after.json` (post-fix).

## 6. Capability demonstrations (Phase 4)

| Demo | Result (pre-fix campaign) |
|---|---|
| D1 EH fixed point + stability | Reproduced pins exactly; exposed real exponents {+0.75, −29.9}, 1 relevant, not UV-attractive — the audit's smoking gun |
| D2 Quadratic NGFP | `find_fixed_point` → None; global search: **no non-Gaussian FP exists** (only Gaussian) |
| D3 Foliated NGFP | Collapses to Gaussian; global search: only the degenerate g=0 fixed line; λ_ADM=1 plane verified |
| D4 CLI gravity-matter scan | Worked; two isolated interior fsolve failures (n_s=2,4) — fixed with retry fallbacks |
| D5 Flow + 3D visualization | 6 trajectories integrated, 3D world-line figure rendered |
| D6 RG-improved black hole | Full Bonanno–Reuter structure with a plateau trajectory (0/2 horizons around **M_cr = 1.196 M_Pl**); with the toolkit's own integrated trajectories, G ∝ r² extrapolation gave exactly 1 horizon for every mass — capability gap, root cause in the (then-broken) flow |
| D7 CLI amplitude + string comparison | UV softening genuine (AS plateau ~110 vs GR 3.9e12); **`all_passed=True` printed beside IR/GR ratio 0.245** — Newtonian-recovery flag was excluded from the roll-up (fixed) |
| D8 Cross-analogue commutativity | Three θ paths agree to ≤5e-13 — because they share the eigendecomposition (reframed as a regression check, tol now 1e-9) |
| D9 numpy vs jax backends | float32 silently used: 1.4e-3 deviation; with x64, 2.7e-12. JaxBatchEvaluator now enables x64 |
| D10 Serialization + HDF5 | JSON srepr round-trip exact (0.0 drift); standalone codegen OK; `.h5` output verified |

## 7. Fixes applied (branch `evaluation-fixes`)

**Commit `91a546e` (campaign findings):** cache-test isolation; matplotlib pin `<3.11`; jax x64 in `JaxBatchEvaluator`; amplitude CLI `all_passed` includes Newtonian recovery; scan retry fallbacks.

**Commit `4c1e281` (adjudicated physics fixes, five parallel clusters + validation docs, each verified against its condemning audit script):**

- **EH core:** beta functions rebuilt to the d=4 Litim closed forms (symbolically identical to the blind-derived reference); d-dependence implemented ((d−2+η_N)g, d-dimensional coefficients; d=2 raises). New NGFP (0.70732088, 0.19320051), θ = 1.4753 ± 3.0432i, η_N(FP) = −2 exactly, GFP slope +1/(2π).
- **Matter:** three η_N paths unified (zero-matter `eh_matter` ≡ `gravity_matter` ≡ `eh`, sympy-exact); DEP per-field weights exact (scalar +1/(6π), Dirac +1/(3π), vector −2/(3π)); trends now match DEP verbatim — scalars push λ\* up with g\* nearly stable, fermions raise g\* (FP lost at N_D ≈ 1.4), vectors lower g\*.
- **Bootstrap:** Virasoro–Shapiro corrected to Γ(−aᵢ)/Γ(1+aᵢ) with massless kinematics — genuine Regge tower (poles at every level, residues matching closed form to 1e-9), super-polynomial falloff on every pole-free window; Veneziano residue sign fixed; `cheung_2025` windows pole-free and the higher-spin check made non-circular.
- **Quadratic:** one-loop universals corrected to +5/36 and +133/20 (×1/16π²), exact rationals asserted at rtol 1e-12; the no-interior-NGFP fact documented everywhere users previously saw "benchmark NGFP" advertisements.
- **Spectral/heat-kernel:** Rubin–Ordóñez degeneracies (both code copies); `trace_on_S1xS3` actually runs; integer S³ TT multiplicities; exponential-regulator thresholds finite and exact (Φ¹₁(0) = π²/6 to 1e-10); TT b₂ and Ω² bundle signs fixed — direct mode sums now match b₀/b₂/b₄ per field type (previously a 65× overcount).
- **Gauge-Higgs:** correct quartic root (B+√disc); ν(30/40/60) = 0.666/0.754/0.832 vs lattice 0.64/0.745/0.81; one relevant direction (was tricritical).
- **Validation/docs honest:** `reuter_1998` now passes 5/5 on actual output; `korver_2024` quotes the real wedge (≈23.1) with a falsifiable validator; `manrique_2011` attribution corrected (MRS Eq. (10): λ\*=0.19, g\*=0.31; no λ_ADM in MRS); foliated NGFP claims removed and the λ_ADM=1 plane described as UV-repulsive; FLRW energy-balance approximation documented; amplitude sign convention declared; de-Sitter-core gloss corrected in docstring/notebook/guide.
- **CLI amplitude trajectory** rebuilt to integrate IR→UV onto the now genuinely UV-attractive NGFP (perturb-and-flow-down now spirals into the λ=1/2 pole — itself evidence the fixed physics is right): IR/GR ratio 1.000, every consistency check passes for real; `ScatteringBridge` samples the string amplitude between Regge poles.

**Post-fix figure re-read (10 previous failures):** after the physics fixes alone, 4 of 10 passed outright (`eh_phase_portrait` — spiral topology, correct NGFP, "2 rel." annotation; `running_couplings` — damped-oscillation saturation at g\* = 0.707, the visual signature of the complex pair; `quadratic_pairwise`; `3d_flow` — helical UV convergence onto the NGFP). The remaining 6 failed for *figure-layer* reasons with precise diagnoses, all then fixed and image-verified: the cosmology trajectory builder ported to the IR-upward construction (lapse figures now show the genuine two-horizon/critical-mass structure with M_cr ≈ 5.7 at G_N = 0.02, naked-remnant/merged/separated regimes at M = 3/6/9; Hawking T_H now peaks and falls toward the extremal limit), `matter_continuation` regenerated with post-fix guesses (all four panels, λ\* rising / g\* falling, Korver wedge shading at ≈23), `foliated_3d`'s `lambda_adm` key bug fixed and its Gaussian FP labeled honestly, `partial_wave_unitarity` renormalized (GR crosses |a₀|=1 at √s ≈ 0.5 M_Pl; the AS plateau sits above the literal bound for documented cutoff reasons — stated in the caption rather than hidden), and stale pre-fix captions (0.69/0.14) updated everywhere. Verdicts: `evaluation/logs/figure-verdicts-after.json`; final renders in `evaluation/figures-after/`.

## 8. Remaining limitations (documented, not fixed)

- **Quadratic and foliated truncations have no NGFP** in their implemented one-loop/schematic forms — now stated honestly in docstrings, captions, and validation modules. Realizing those fixed points requires genuinely new derivations (FRG-improved quadratic betas; full ADM-decomposed foliated hessians), which is feature work, not bug fixing.
- **TT heat-kernel b₄ on general backgrounds** retains a documented approximation (full Lichnerowicz endomorphism trace); sphere-constrained values are exact.
- **RG-improvement epistemic status** (cosmology and scattering observables are improvements of solutions, not first-principles form factors) — disclosed in docstrings, consistent with Knorr 2602.21285.
- **Partial-wave unitarity** uses a documented angular cutoff (massless t-pole); forward positivity deliberately not enforced.
- **GUI** (PySide6) not evaluated — no display; its matplotlib layer was exercised. Quantum modules exercised through their unit tests (qiskit present, all green).
- The deep-IR fine-tuning of trajectories (the cosmological-constant problem in miniature) limits how far integrated flows can extend; the IR-upward construction sidesteps this for observables, and `make_as_trajectory` remains the canonical idealized pattern.

## 9. Recommendations

1. **Treat `evaluation/audit/findings.json` as the canonical defect record** — each entry has a reproducer; re-run them after any beta-function refactor.
2. Keep benchmark tests pointed at *computed* physics: the new suite asserts `validate_eh_fixed_point()["all_passed"]` and computed ν(N_f) against lattice values — preserve that pattern when adding truncations.
3. If the quadratic/foliated NGFPs matter to the project's goals, implement the real derivations (CPR 2009 full system; MRS 1102.5012 ADM hessians) — the corrected EH core now provides trustworthy building blocks (thresholds, heat kernels, spectra all verified).
4. Consider porting the CLI's IR-upward trajectory construction to the cosmology examples (notebook 04) so RG-improved black holes use plateau trajectories by default.
5. Add a CI run of `pytest` + the 10 baseline hashes + a handful of audit reproducers; the suite runs in ~35 s.

---

*Generated by an ultracode multi-agent evaluation: 4 blind-derivation agents, 4 domain auditors, 24 skeptic defenses, 1 referee (Phase 1); full test/notebook/figure campaigns (Phases 2–3); 10 capability demos (Phase 4); 6 fix clusters with per-finding re-verification (Phase 5). ~70 subagents, ~3.9M subagent tokens.*
