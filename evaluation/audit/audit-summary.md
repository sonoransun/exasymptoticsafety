# asymsafety mathematical audit — referee summary

**Date:** 2026-06-09 · **Scope:** 46 checks across four domains (spectral, frg-beta, amplitudes, observables), each contested finding subjected to a skeptic defense and referee adjudication. All BUG reproducers were re-run independently by the referee (28/28 spot-checks confirm the audited facts; see `evaluation/audit/referee/referee_spotchecks.py`). Final per-finding records: `evaluation/audit/findings.json`.

## Counts

| Classification | Count |
|---|---|
| BUG | 15 |
| MISLEADING-CLAIM | 10 |
| INTERNAL-INCONSISTENCY | 9 |
| VERIFIED-CORRECT | 9 |
| CORRECT-WITH-DOCUMENTED-APPROXIMATION | 2 |
| DISCREPANT-VS-LITERATURE | 1 |

| Severity | Count |
|---|---|
| High | 12 |
| Medium | 18 |
| Low | 16 |

Referee adjudication changes vs the raw audits: **HV-14f** upgraded MISLEADING-CLAIM → VERIFIED-CORRECT (defense succeeded: crossing-by-construction is a disclosed, real scheme choice, and the check does fail crossing-violating scale identifications). **HV-10d** reclassified INTERNAL-INCONSISTENCY → MISLEADING-CLAIM, High → Medium (the FP-free one-loop higher-derivative structure is literature-standard and documented in tests; the residual sin is docstrings/GUI advertising a "benchmark NGFP"). All other defenses failed; classifications confirmed.

## Headline defects (most severe first)

1. **HV-8 / HV-11b — Einstein–Hilbert beta functions are structurally wrong** (`src/asymsafety/beta/einstein_hilbert.py:99-122`; BUG, High). The graviton volume trace counts 6 modes (3A) instead of 10 (5A) while keeping the full ghost −8, the η-numerator/denominator match no derivation, and the IR sign of dβ_λ/dg at the Gaussian FP is flipped (−8/π vs +1/(2π)) — graviton loops generate *negative* vacuum-energy flow. Consequences: λ\*=0.142 vs 0.193, θ = {+0.75, −29.9} both **real** with **one** relevant direction vs the scheme-robust complex pair 1.475±3.043i with two. This is the core truncation every CLI/GUI/figure/cosmology/scattering path inherits. Reproduce: `evaluation/audit/frg-beta/eh_litim_reference.py`, `hv11_gaussian.py`.

2. **HV-12 / HV-12b — gravity+matter sector contradicts itself and its cited source** (`frg/anomalous_dim.py:80-93`, `actions/matter.py:192`, `beta/matter.py:271`; INTERNAL-INCONSISTENCY + BUG, High). The `eh_matter` builder's η_N differs symbolically from the EH builder's, so the zero-matter limit has **no NGFP at all** (`scan_matter_content` returns fp_exists=False even at N=0). The scalar matter weight has the **wrong sign** vs Donà–Eichhorn–Percacci (the cited reference): the toolkit predicts scalars *stabilize* the fixed point (g\* 0.694→0.535 at N_s=4) — the opposite trend. Reproduce: `evaluation/audit/frg-beta/hv12_matter.py`.

3. **HV-13c / HV-13d / HV-13e — the "Virasoro–Shapiro" amplitude is not Virasoro–Shapiro** (`scattering/bootstrap.py:83,131`, `validation/cheung_2025.py:46`; BUG + INTERNAL-INCONSISTENCY + MISLEADING-CLAIM, High). The denominator Γ(a_i+a_j) form is paired with the wrong numerators and used off its constraint surface: with massless kinematics (α₀=0) the function is **identically 1**; with the shipped defaults it is a 3-pole rational function with a pure s⁻⁹ tail — no Regge tower, not ultrasoft. The cheung_2025 "ultrasoft" validation passes only via pole-contaminated fit windows, so the headline AS-vs-strings bridge compares AS against a rational function. Reproduce: `evaluation/audit/amplitudes/hv13_bootstrap.py`.

4. **HV-10 / HV-10b — quadratic-gravity one-loop universals are misassigned** (`beta/quadratic.py:91-113` vs `validation/codello_2009.py:25-31`; INTERNAL-INCONSISTENCY + BUG, High). The λ→0 limits are exactly 67/180 and −329/90 (×1/16π²), contradicting the file's own "exact one-loop universal" comments (53/45, −196/45), and neither set matches the true coefficient-basis universals (+5/36, +133/20): the C² invariant (133/10) is attached to the R² coupling and 3× the Gauss-Bonnet universal contaminates β_α. Every quadratic-sector number is wrong; the documented "asymptotic freedom" sign (HV-10c) is inverted in the code's own action convention. Reproduce: `evaluation/audit/frg-beta/hv10_quadratic.py`.

5. **HV-15c-b / HV-15c-c / HV-15c-d — the foliated NGFP narrative is unrealized and misattributed** (`beta/foliated.py:52,143`, `validation/manrique_2011.py:29`; INTERNAL-INCONSISTENCY ×2 + MISLEADING-CLAIM, High/Medium). The implemented foliated system admits **no physical NGFP** (η = −2 unreachable for λ > −1/2 under any admissible regulator); the advertised benchmark (0.96, 0.20, 1) is not a root (β = (3.26, 0.18, 0)); the values are attributed to Manrique–Rechenberger–Saueressig PRL 106, 251302, whose actual Eq. (10) gives (0.19, 0.31) and which contains no λ_ADM coupling; and the λ_ADM=1 plane is UV-*repulsive* at physical λ, contradicting the "full-Diff restoration" story and the repo's own (sign-blind) test. Reproduce: `evaluation/audit/observables/hv15c_foliated.py`, `hv15c_supplement.py`.

6. **HV-15d-c — gauge-Higgs analogue picks the wrong quartic root** (`transforms/bridge/gauge_higgs.py:157`; BUG, High). `charged_fp_guess` returns u₋ = (B−√disc)/(2A); the Wilson–Fisher-connected charged FP — by the code's *own* stated criterion — is u₊. Consequence: ν(N_f) decreases toward 1/2 instead of increasing toward 1, the FP is tricritical (2 relevant directions), and the Bonati-2025 comparison fails; with u₊ it agrees to ~1.5%. One-character-class fix. Reproduce: `evaluation/audit/observables/hv15d_bridge.py` (section D).

7. **HV-9 / HV-9b / HV-12c — validation modules certify failures** (`validation/reuter_1998.py:36-44`, `beta/einstein_hilbert.py:140-154`, `validation/korver_2024.py:23-29`; MISLEADING-CLAIM, High/Medium/Low). The Reuter validation fails 4/5 of its own checks on actual toolkit output (the test suite avoids asserting `all_passed`); `eh_fixed_point_litim_d4` labels the toolkit's own anomalous FP "known" literature values; the Korver-2024 matter bounds (12, 6) misquote the cited paper (≈23, ≈3–4) and the validator's boolean logic cannot fail in the regime it claims to test.

8. **Spectral/heat-kernel layer: five independent math errors, all off the beta-function path** (BUG ×5, Medium). S⁴ scalar degeneracy `(l+1)²(l+2)²/4` vs exact `(2l+3)(l+1)(l+2)/6` (HV-3, `geometry/decomposition.py:80`); transverse-vector degeneracy 3× scalar tower, l=1 gives 15 vs 10 Killing vectors (HV-4, `:106`); `trace_on_S1xS3` crashes on every call via an always-true `hasattr` guard, with a non-integer S³ TT fallback (HV-4c, `frg/spectral.py:103,132`); TT b₂ = +5R/3 matches no operator (HV-5, `frg/heat_kernel.py:211`); vector/TT b₄ have the Ω² bundle-curvature sign flipped (HV-6/HV-6b, `:247,:255`). Also the exponential-regulator threshold integral returns NaN on every call and implements a wrong numerator (HV-2, `frg/threshold.py:137`). None of these feed a beta function or fixed point — the Litim closed forms used by all builders are verified correct (HV-1).

## Verified-correct highlights

- **HV-1**: Litim threshold functions Φᵖₙ, Φ̃ᵖₙ and the QFunctional match definitional quadrature to ~1e-16, with all derivative/recursion identities exact — the closed forms every beta builder actually consumes are sound.
- **HV-7 / HV-11**: the η_N algebraic closure is exact (η_N = −2 at the NGFP to 7.6e-13) and the Gaussian-FP structure is exactly canonical ({+2, −2}, β_λ(0,λ) = −2λ).
- **HV-13a**: the Veneziano amplitude is implemented exactly (1.8e-15 vs mpmath; crossing, poles, Regge all verified).
- **HV-14a/c/d/f**: the RG-improved graviton amplitude is exactly crossing symmetric, reduces to GR in the IR, exhibits the safe/unsafe fixed-angle dichotomy matching the analytic −8πg\* plateau to 7 digits, and the crossing consistency check is a disclosed by-construction pass that genuinely discriminates against crossing-violating scale identifications (successful defense).
- **HV-15a**: the RG-improved Schwarzschild horizon structure and critical mass reproduce the Bonanno–Reuter closed form to ~3e-4.
- **HV-14e / HV-15c**: documented approximations (angular cutoff in the unitarity check; hand-inserted λ_ADM fixed plane) are honest and conclusion-stable.

## Fix shortlist

### Source fixes (BUG — unambiguous mathematics, reproducers confirmed by referee)

Priority order:

1. **HV-8 + HV-11b** — rederive EH trace coefficients (10-mode graviton trace, η numerator/denominator, 1/(2π) volume normalization); regenerate the pinned FP values in `validation/` and `tests/test_benchmarks_published.py`.
2. **HV-12b** — correct matter weights to DEP (scalar +1/(6π), Dirac +1/(3π), vector −2/(3π)) in `actions/matter.py:192` and the inline duplicate `beta/matter.py:271`.
3. **HV-12** — unify the three η_N implementations so `build_eh_matter_beta_system(MatterContent())` ≡ `build_eh_beta_system` (fix `frg/anomalous_dim.py:80-93`).
4. **HV-13c + HV-13d** — Virasoro–Shapiro: use Γ(−a_i)/Γ(1+a_i) denominators with α₀=0 massless kinematics (sum a_i = 0); then re-run cheung_2025 (resolves HV-13e for the VS side).
5. **HV-13b** — `veneziano_residue` sign: drop the (−1)ⁿ alternation (Res = −(1/n!)·∏(α(t)+k)).
6. **HV-10b** — quadratic one-loop constants: β_α → +5/36, β_β → +133/20 (×1/16π²) in the coefficient basis; fix `codello_2009.py` dict accordingly (resolves HV-10 and the HV-10c sign story).
7. **HV-15d-c** — `gauge_higgs.py:157`: `(B - sqrt(disc))` → `(B + sqrt(disc))`; then re-tighten the loosened monotonicity tests.
8. **HV-11c** — `build_eh_beta_system`: implement d-dependence ((d−2+η)g etc.) or raise `NotImplementedError` for d≠4 (CLI/GUI reachable).
9. **HV-2** — exponential-regulator Φ: numerator −z·y·r′ per the module's Reuter convention + overflow guard (currently NaN on every call).
10. **HV-3, HV-4** — S⁴ scalar/vector degeneracies → Rubin–Ordóñez formulas (also the verbatim duplicates in `compute/batch/spectral.py:109,111`).
11. **HV-4c** — fix the always-true `hasattr` guard in `trace_on_S1xS3` and the S³ TT multiplicity → 2(l−1)(l+3) (both copies).
12. **HV-5, HV-6, HV-6b** — heat kernel: TT coefficients via the constrained spectral route (b₂ = −(5/6)R for −D², −(25/6)R for Lichnerowicz); flip the Ω² bundle signs (vector −11/180 Riem²; Sym² −(d+2)/12).

### Documentation / data fixes (MISLEADING-CLAIM or INTERNAL-INCONSISTENCY residing in docs, dicts, or validation framing)

- **HV-9 / HV-9b** — stop presenting reuter_1998 as a passing validation (or fix HV-8 so it passes); delete or relabel `eh_fixed_point_litim_d4`'s "known values" framing (include θ₂).
- **HV-10d** — remove "Benchmark NGFP coordinates"/"four-coupling NGFP search" advertisements (`quadratic.py:46-48`, `phase_portrait.py:1013-1017`); document the FP-free one-loop structure where users see it.
- **HV-10c** — state the AF criterion in the code's own coefficient convention (or fix with HV-10b).
- **HV-12c** — correct `korver_2024.py` bounds to the KSW wedge (N_s+6.4N_v ≈ 23.1) and fix the can't-fail validator boolean.
- **HV-15a-b** — remove/correct the "de Sitter core / no curvature singularity" gloss (docstring, notebook 04, visualization guide), or implement BR's d(r) ~ r^{3/2} softened scale (that would be a source feature).
- **HV-15b** — document that integrate() imposes standard conservation (energy exchange with the G,Λ sector dropped), or implement the modified continuity equation + the BR constraint fixing ξ.
- **HV-15c-b / HV-15c-c / HV-15c-d** — fix the manrique_2011 attribution/values (MRS Eq. (10): 0.19/0.31), drop the unrealized "NGFP exists"/benchmark/restoration claims from `beta/foliated.py`, and make the foliated test sign-aware (or fix the schematic coefficients — source work).
- **HV-13e** — after the HV-13c source fix, recompute the cheung_2025 windows pole-free; document the circularity of the higher-spin check.
- **HV-15d / HV-15d-b** — reframe `verify_commutativity` as a linear-algebra regression check (tighten tol; note the resolvent path is definitionally identical) and either implement a genuine EDMD comparison in `compare_with_stability` or rename it; fix README/figure-caption claims.
- **HV-7b** — one-sign docstring fix in `conventions.py:29` (and `anomalous_dim.py:3`): η_N = +∂_t ln G = −∂_t ln Z_N.
- **HV-2b** — normalize `_phi_symbolic` to the module's stated convention (factor 1/2) or document the dtR_k normalization; make it return an Integral.
- **HV-4b** — `excluded_modes['vector']` → [1] per its own docstring (dead code; lowest priority).
- **HV-14b** — declare the amplitude sign convention (or flip to +8πG per DKRS); currently no consumer is numerically affected.

### No action required

- No DISCREPANT finding survived with a successful scheme defense in a form requiring zero action: **HV-14b** (the only DISCREPANT-VS-LITERATURE) needs a one-line convention declaration (filed under documentation above). **HV-14f** was acquitted entirely (VERIFIED-CORRECT, no action). The 9 VERIFIED-CORRECT and 2 CORRECT-WITH-DOCUMENTED-APPROXIMATION findings need no action.

## Caveat for maintainers

`tests/test_benchmarks_published.py` hard-pins the toolkit's current (incorrect) EH fixed point at rtol 1e-6, and several tests were written to tolerate the defects (sign-blind foliated assertion, either-direction monotonicity in gauge-Higgs, neutered `all_passed`). Any fix to HV-8/HV-12/HV-15d-c **must** regenerate those pins and re-tighten those assertions, otherwise the suite will report regressions against bugged baselines.
