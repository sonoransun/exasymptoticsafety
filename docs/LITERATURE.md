# Recent Literature: Asymptotic Safety and FRG Computations (2020–2026)

A structured survey of recent scientific papers expanding on the physics and computational methods implemented in this project. Organized by the project's key areas.

---

<a id="lorentzian-foliated"></a>
## 1. Lorentzian Signature & Foliated Gravity

These papers directly extend the foliated ADM formulation in `beta/foliated.py` and the Manrique et al. (2011) benchmarks in `validation/`.

### D'Angelo, Drago, Pinamonti & Rejzner (2024)

**"Asymptotic safety in Lorentzian quantum gravity"**
Phys. Rev. D **109**, 066012 [[2310.20603](https://arxiv.org/abs/2310.20603)]

First covariant Lorentzian Wetterich-type functional renormalization group equation formulated directly in Lorentzian signature. Shows that universal terms in the flow determine a Reuter-type fixed point for Lorentzian quantum gravity, providing the first background-independent, non-trivial fixed point evidence in Lorentzian signature in the Einstein–Hilbert truncation. **Key result**: the Euclidean asymptotic safety mechanism carries over to Lorentzian spacetimes.

**Relevance**: Validates the project's Euclidean EH results; opens the path to implementing Lorentzian FRG flow equations directly.

---

### Saueressig, Knorr & collaborators (2025)

**"Foliated Asymptotically Safe Gravity: Lorentzian Signature Fluctuations from the Wick Rotation"**
Phys. Rev. D **111**, 106007 [[2501.03752](https://arxiv.org/abs/2501.03752)]

Studies the Wick rotation from Euclidean to Lorentzian signature spacetimes within the ADM-based FRG framework. Computes the RG flow of the graviton two-point function using vertices from the Einstein–Hilbert action. Establishes that the Lorentzian two-point function resulting from analytic continuation of the lapse has the causal structure of the Feynman propagator. UV and IR completions identified in the Euclidean case are robust when changing spacetime signature.

**Relevance**: Directly extends `beta/foliated.py`. Confirms $\lambda_{\mathrm{ADM}} \to 1$ (full-Diff restoration) under Wick rotation. New Lorentzian coefficients could be added to the existing foliated beta functions.

**Figures**: [`foliated_3d`](images/foliated_3d.caption.md), [`fp_stability_3d`](images/fp_stability_3d.caption.md).

---

### Korver, Saueressig & Wang (2024)

**"Global Flows of Foliated Gravity-Matter Systems"**
Phys. Lett. B **855**, 138789 [[2402.01260](https://arxiv.org/abs/2402.01260)]

Derives bounds on the number of matter fields compatible with asymptotic safety in the foliated setting. The flow is driven by 3- and 4-point vertices from the foliated Einstein–Hilbert action supplemented by minimally coupled scalar and vector fields. An intriguing feature is the presence of an IR fixed point for the graviton mass that prevents the squared mass from taking negative values — this persists for any number of matter fields.

**Relevance**: Extends `beta/matter.py` with foliated matter bounds. New validation benchmarks for the number of allowed matter fields.

**Figures**: [`matter_continuation`](images/matter_continuation.caption.md).

---

### Knorr, Ripken & Saueressig (2023)

**"Foliated asymptotically safe gravity in the fluctuation approach"**
JHEP **09**, 064 [[2306.10408](https://arxiv.org/abs/2306.10408)]

Applies the fluctuation approach (background-field split with independent dynamical and background couplings) to foliated quantum gravity. Background-independent beta functions are derived, going beyond the single-metric approximation.

**Relevance**: The project's `beta/foliated.py` currently uses a single-metric approximation. This paper provides the path to bi-metric extensions.

---

<a id="vertex-expansions"></a>
## 2. Vertex Expansions & Momentum-Dependent Couplings

These papers extend the trace evaluation and anomalous dimension methods in `frg/traces.py` and `frg/anomalous_dim.py`.

### Pawlowski & Reichert (2023)

**"Quantum Gravity from dynamical metric fluctuations"**
[[2309.10785](https://arxiv.org/abs/2309.10785)]

Develops a systematic vertex expansion for the functional RG applied to gravity, with flow equations derived for both Euclidean and Lorentzian signatures. Disentangles dynamical metric fluctuations from the background metric. Discusses convergence of the expansion scheme for the dynamical graviton propagator.

**Relevance**: Could improve anomalous dimension computation in `frg/anomalous_dim.py` beyond the single-metric approach. The vertex expansion provides a systematic improvement over the threshold-function approach.

---

### Knorr, Ripken & Saueressig (2022)

**"Form factors in asymptotically safe quantum gravity"**
In *Handbook of Quantum Gravity*, Springer [[2210.16072](https://arxiv.org/abs/2210.16072)]

Introduces momentum-dependent form factors that modify the UV graviton propagator. Derives conditions under which the resulting scattering amplitudes are UV-finite, unitary, and causal. Form factors can be constructed that lead to scale-free amplitudes at trans-Planckian energies without introducing ghost poles.

**Relevance**: Relevant for extending batch evaluators to momentum-dependent couplings. Addresses whether the project's computed fixed points lead to consistent scattering.

---

### Fehre, Litim, Sherrill & Sherrill (2023)

**"Momentum-dependent field redefinitions in Asymptotic Safety"**
[[2311.12097](https://arxiv.org/abs/2311.12097)]

Shows that momentum-dependent field redefinitions can eliminate ghost poles that appear in higher-derivative truncations. Evidence that the asymptotically safe fixed point may not feature extra ghost degrees of freedom.

<a id="quadratic"></a>

**Relevance**: Directly addresses unitarity concerns with the quadratic gravity truncation in `beta/quadratic.py`. Suggests the ghost poles in the 4th-order propagator may be artefacts of the truncation.

**Figures**: [`quadratic_pairwise`](images/quadratic_pairwise.caption.md).

### Draper, Knorr, Ripken & Saueressig (2020)

**"Graviton-Mediated Scattering Amplitudes from the Quantum Effective Action"**
Phys. Rev. Lett. **125**, 181301 [[2007.04396](https://arxiv.org/abs/2007.04396)]

Constructs gauge-invariant graviton-mediated scattering amplitudes for non-minimally coupled scalars from a curvature/form-factor expansion of the effective action. The amplitude reduces to general relativity in the infrared and, once the asymptotically safe scale-invariant regime is reached, approaches a constant in the ultraviolet — UV-finite, unitary and causal without extra degrees of freedom.

**Relevance**: The physics target of `scattering/` (the `GravitonMediatedAmplitude` IR→GR / UV-constant limits and the no-ghost causality check). Validated by `validation/draper_2020.py`.

### Knorr (2026)

**"Asymptotically (un)safe scattering amplitudes from scratch: a deep dive into the IR jungle"**
[[2602.21285](https://arxiv.org/abs/2602.21285)]

Computes leading quantum-gravity contributions to a scalar scattering amplitude in asymptotic safety and shows that the existence of a fixed point does **not** by itself guarantee a bounded amplitude (the "safe vs unsafe" dichotomy); naive RG-improvement / derivative expansion can fail quantitatively, so genuine momentum-dependent form factors are required.

**Relevance**: The central caveat documented throughout `scattering/`; realised as the safe-vs-unsafe diagnostic in `validation/knorr_2026.py`.

---

<a id="matter-coupling"></a>
## 3. Matter Coupling & Phenomenology

These papers extend the matter coupling computations in `beta/matter.py`.

### Eichhorn & Schiffer (2022)

**"Asymptotic safety of gravity with matter"**
In *Handbook of Quantum Gravity*, Springer [[2212.07456](https://arxiv.org/abs/2212.07456)]

Comprehensive review of gravity-matter fixed points. Provides updated bounds on the matter content $(N_s, N_D, N_v)$ compatible with asymptotic safety in various truncation schemes. Discusses the interplay between gravitational and matter beta functions.

**Relevance**: Directly updates the matter bounds in `beta/matter.py` and provides new reference values for validation.

**Figures**: [`matter_continuation`](images/matter_continuation.caption.md).

---

### Buccio, Percacci & collaborators (2025)

**"Asymptotic safety meets tensor field theory: Toward a new class of gravity-matter systems"**
Phys. Rev. D **111**, 085030 [[2501.10307](https://arxiv.org/abs/2501.10307)]

First example of a theory with gravity and scalar fields in four dimensions that may realize asymptotic safety at a non-vanishing value of the scalar quartic coupling. Demonstrates that quantum fluctuations of gravity generically screen quartic couplings in multi-scalar models. The fixed point originates from competition between antiscreening matter self-interactions and screening gravitational effects.

**Relevance**: Novel gravity-matter fixed point type not yet implemented in the project. Could motivate extending `beta/matter.py` to include scalar self-interactions.

---

### Draper, Knorr, Ripken & Saueressig (2025)

**"$e^+e^- \to \mu^+\mu^-$ in the Asymptotically Safe Standard Model"**
Phys. Rev. D **111**, 106005 [[2412.13800](https://arxiv.org/abs/2412.13800)]

First computation of a scattering cross section in the asymptotically safe Standard Model. Graviton contributions to the scattering amplitude are computed from momentum-dependent timelike one-particle-irreducible correlation functions. The full asymptotically safe quantum cross section decreases in the UV with center-of-mass energy and is compatible with unitarity bounds.

**Relevance**: Demonstrates observational testability of asymptotic safety. Shows that the running couplings computed by this project can be connected to physical scattering processes.

---

### Eichhorn, Held & collaborators (2025)

**"Unearthing the intersections: positivity bounds, weak gravity conjecture, and asymptotic safety landscapes from photon-graviton flows"**
JHEP **03**, 003

Connects asymptotic safety to swampland conjectures and positivity bounds derived from photon-graviton flows. Explores the landscape of asymptotically safe gravity-matter theories constrained by consistency conditions from quantum gravity.

**Relevance**: Theoretical consistency constraints that could be used to cross-validate the project's computed fixed points.

---

<a id="black-holes-cosmology"></a>
## 4. Black Holes & Cosmological Applications

These papers represent new application directions for the running couplings computed by this project.

### Platania (2023/2025)

**"Black Holes in Asymptotically Safe Gravity"**
In *Handbook of Quantum Gravity*, Springer [[2302.04272](https://arxiv.org/abs/2302.04272)]

**"Some thoughts about black holes in asymptotic safety"** (2025)
Gen. Rel. Grav.

Comprehensive review of quantum-corrected black hole solutions constructed via RG improvement of classical metrics. Key features: (i) singularity resolution, (ii) more compact event horizons and photon spheres, (iii) second (inner) horizon even at vanishing spin, (iv) cold remnant as a possible Hawking evaporation endpoint. Discusses quasinormal modes, grey-body factors, and EHT shadow constraints.

**Relevance**: The running couplings $g(k)$, $\lambda(k)$ computed by this project can be directly inserted into the RG improvement procedure to generate quantum-corrected black hole metrics.

**Figures**: [`running_newton_constant`](images/running_newton_constant.caption.md), [`lapse_with_horizons`](images/lapse_with_horizons.caption.md), [`classical_vs_rg_lapse`](images/classical_vs_rg_lapse.caption.md), [`hawking_temperature`](images/hawking_temperature.caption.md).

---

### Platania (2025)

**"Cosmic Acceleration from Quantum Gravity: Emergent Inflation and Dynamical Dark Energy"**
[[2512.11712](https://arxiv.org/abs/2512.11712)]

Shows that emergent inflation and dynamical dark energy can arise from the asymptotically safe effective action. The $R^2$ terms from the quadratic truncation drive early-universe acceleration, while late-time acceleration emerges from quantum gravity effects.

**Relevance**: Connects the $R^2$ coupling $\alpha$ from `beta/quadratic.py` to inflationary phenomenology. The Starobinsky model $R + \alpha R^2$ naturally arises from the quadratic truncation.

**Figures**: [`flrw_evolution`](images/flrw_evolution.caption.md), [`quadratic_pairwise`](images/quadratic_pairwise.caption.md).

---

### Bonanno, Koch & Platania (2025)

**"Black Holes in Asymptotic Safety: A Review of Solutions and Phenomenology"**

Comprehensive review including constraints from the Event Horizon Telescope (EHT) and X-ray observations. For slowly-spinning black holes, current observations constrain quantum-gravity scales far above the Planck length, but near-critical spin may produce signatures detectable by next-generation EHT.

**Relevance**: Observable predictions from the project's computed running couplings. Potential basis for a new `applications/` module.

---

<a id="lattice-continuum"></a>
## 5. Lattice–Continuum Connection

These papers provide independent verification of the FRG approach used in this project.

### Ambjørn, Gizbert-Studnicki, Görlich, Jurkiewicz & Németh (2024)

**"Is Lattice Quantum Gravity Asymptotically Safe? Making contact between Causal Dynamical Triangulations and the Functional Renormalization Group"**
[[2408.07808](https://arxiv.org/abs/2408.07808)]

Direct comparison between Causal Dynamical Triangulation (CDT) Monte Carlo simulations and FRG calculations near critical phase transition lines. Results are compatible with the existence of a UV fixed point, although data precision does not yet provide definitive proof.

**Relevance**: Cross-validates the FRG approach used throughout this project from a completely independent lattice perspective.

---

### Ambjørn, Görlich, Jurkiewicz & Loll (2024)

**"Causal Dynamical Triangulations: Gateway to Nonperturbative Quantum Gravity"**
[[2401.09399](https://arxiv.org/abs/2401.09399)]

Review of CDT status including the spectral dimension reduction from $d_s \approx 4$ at large scales to $d_s \approx 3/2$ at small scales (dynamical dimensional reduction). Phase structure analysis near the A-CdS transition line with critical exponent $\delta = 0.54 \pm 0.04$.

**Relevance**: Dynamical dimensional reduction is a prediction of asymptotic safety that can be compared with the spectral methods in `frg/spectral.py`.

---

<a id="reviews"></a>
## 6. Systematic Reviews & Conceptual Assessments

### Bonanno, Eichhorn, Gies, Pawlowski, Percacci, Reuter, Saueressig & Vacca (2020)

**"Critical reflections on asymptotically safe gravity"**
Front. Phys. **8**, 269 [[2004.06810](https://arxiv.org/abs/2004.06810)]

Multi-author critical assessment of the asymptotic safety program by leading researchers. Addresses: convergence of truncation schemes, scheme/gauge dependence, the role of essential vs. inessential couplings, unitarity, and predictivity. Identifies the key open questions for the program.

**Relevance**: Essential context document. The open questions identified here (scheme dependence, convergence, unitarity) directly motivate the validation benchmarks and multiple-regulator approach implemented in this project.

**Figures**: [`asymptotic_safety_concept`](images/asymptotic_safety_concept.caption.md), [`eh_phase_portrait`](images/eh_phase_portrait.caption.md), [`comparison_table`](images/comparison_table.caption.md), [`pseudospectrum`](images/pseudospectrum.caption.md).

---

### Saueressig (2023)

**"The Functional Renormalization Group in Quantum Gravity"**
In *Handbook of Quantum Gravity*, Springer [[2302.14152](https://arxiv.org/abs/2302.14152)]

The most comprehensive and up-to-date review of FRG methods for quantum gravity. Covers: the Wetterich equation, regulator choices, heat kernel and spectral sum techniques, threshold functions, the Einstein–Hilbert truncation, higher-derivative extensions, the fluctuation approach, and systematic vertex expansions.

**Relevance**: Most current review of the exact methods implemented in `frg/`. Serves as the definitive reference for all computational techniques used in this project.

---

### Reuter & Saueressig (2019)

**"Quantum Gravity and the Functional Renormalization Group: The Road towards Asymptotic Safety"**
Cambridge University Press

Textbook treatment of the entire asymptotic safety program. Covers the Wetterich equation, regulators, truncation strategies, the Einstein–Hilbert truncation, higher-derivative gravity, the role of topology, and connections to other approaches.

**Relevance**: Canonical reference for all computations in this project. Provides the textbook derivations underlying `beta/`, `frg/`, and `analysis/`.

---

## Summary: How Recent Results Relate to This Project

| Project Module | Recent Extension | Key Papers |
|---|---|---|
| `beta/einstein_hilbert.py` | Lorentzian FP confirmation | D'Angelo et al. (2024) |
| `beta/foliated.py` | Wick rotation, fluctuation approach | Saueressig et al. (2025), Knorr et al. (2023) |
| `beta/matter.py` | Updated matter bounds, tensor field theory | Korver et al. (2024), Eichhorn & Schiffer (2022), Buccio et al. (2025) |
| `beta/quadratic.py` | Ghost-freedom via field redefinitions | Fehre et al. (2023) |
| `frg/traces.py` | Vertex expansion, form factors | Pawlowski & Reichert (2023), Knorr et al. (2022) |
| `frg/anomalous_dim.py` | Dynamical graviton propagator | Pawlowski & Reichert (2023) |
| `frg/spectral.py` | CDT spectral dimension comparison | Ambjørn et al. (2024) |
| `validation/` | Lorentzian benchmarks, foliated matter bounds | D'Angelo et al. (2024), Korver et al. (2024) |
| `transforms/bridge/gauge_higgs.py` | Charged-FP analogue for 3D AHM | Bonati, Pelissetto & Vicari (2025) |
| (future) `applications/` | RG-improved black holes, AS cosmology | Platania (2023/2025), Bonanno et al. (2025) |

---

<a id="cross-disciplinary-analogues"></a>
## 8. Cross-Disciplinary Analogues

Papers from outside quantum gravity whose rigour and methodology can be carried over via the toolkit's `transforms/bridge/` machinery (the *cross-analogue bridge*, which enforces a commutative diagram across classical-RG / hydraulic / quantum / integral-transform representations).

### Bonati, Pelissetto & Vicari (2025)

**"Three-dimensional Abelian and non-Abelian gauge-Higgs theories"**
Phys. Rep. [[2410.05823](https://arxiv.org/abs/2410.05823)]

Comprehensive Physics Reports review consolidating lattice Monte-Carlo and field-theory results for 3D gauge-Higgs systems (U(1) and SU(N_c) gauge groups, N_f complex scalar flavors in the fundamental representation). The headline result is the rigorous identification of a *charged fixed point* (CFP) controlling Coulomb-to-Higgs transitions when the matter content is large enough (`N_f > N_f^*`). The threshold is regulator-dependent — `N_f^* ≈ 375` in the 4-ε ε-expansion for SU(2) but well below 30 on the d=3 lattice — and the correlation-length exponent follows the large-N_f law `ν = 1 − 9.727/N_f` (SU(2)). Reference MC values at `N_f ∈ {30, 40, 60}` are tabulated.

**Relevance**: Mirrors the asymptotic-safety NGFP from a 3D stat-mech vantage point. Both are interacting UV fixed points whose existence depends on matter content and on the regulator/scheme; the paper documents exactly the kind of scheme-dependence debate that pervades gravity-side FRG. Implemented in `transforms/bridge/gauge_higgs.py` (toolkit-side AHM β-system + `GaugeHiggsAnalogue` wrapper for `CrossAnalogueBridge`) and `validation/bonati_2025.py` (MC reference data, large-N_f coefficient, existence-threshold helper). See [`docs/cross-analogue-gauge-higgs.md`](cross-analogue-gauge-higgs.md) for the full dictionary mapping AS ↔ AHM concepts.

**Figures**: [`ahm_phase_diagram`](images/ahm_phase_diagram.caption.md), [`charged_fp_boundary`](images/charged_fp_boundary.caption.md), [`nu_vs_nf`](images/nu_vs_nf.caption.md), [`ahm_as_bridge`](images/ahm_as_bridge.caption.md).

**Animations**: [`rg_flow_ahm`](animations/rg_flow_ahm.caption.md), [`nu_vs_nf_sweep`](animations/nu_vs_nf_sweep.caption.md).

### Cheung, Remmen, Sciotti & Tarquini (2025)

**"Strings from Almost Nothing"**
Phys. Rev. Lett. (DOI `cw4p-cqh7`) [[2508.09246](https://arxiv.org/abs/2508.09246)]

A physical-scattering bootstrap: from analyticity, crossing, ultrasoft (faster-than-power-law) high-energy behaviour, and an infinite sequence of momentum-transfer values at which higher-spin exchanges cancel, the space of minimally consistent four-point amplitudes collapses onto the Veneziano and Virasoro–Shapiro string amplitudes — with the Regge mass spectrum as an output of the bootstrap.

**Relevance**: The "physical scattering" half of the combined analysis. The toolkit implements the bootstrap amplitudes and their defining properties (`scattering/bootstrap.py`) and runs the *same* physical-consistency battery on both the asymptotically-safe and the string amplitudes via `scattering/bridge.py` (`ScatteringBridge`), establishing that asymptotic safety and strings are distinct, mutually consistent points in the space of physical amplitudes. Validated by `validation/cheung_2025.py`. See [`docs/scattering-amplitudes.md`](scattering-amplitudes.md).

**Figures**: [`regge_trajectory`](images/regge_trajectory.caption.md), [`as_vs_string`](images/as_vs_string.caption.md), [`amplitude_vs_energy`](images/amplitude_vs_energy.caption.md), [`partial_wave_unitarity`](images/partial_wave_unitarity.caption.md), [`graviton_form_factor`](images/graviton_form_factor.caption.md).

