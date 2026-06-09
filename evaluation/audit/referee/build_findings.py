# Referee: assemble final adjudicated findings.json
import json

F = []


def add(id, domain, cls, sev, file_line, claim, gt, obs, cmd, blast, defense):
    F.append({
        "id": id, "domain": domain, "classification": cls, "severity": sev,
        "file_line": file_line, "claim": claim, "ground_truth": gt,
        "observed_vs_expected": obs, "command": cmd, "blast_radius": blast,
        "defense_summary": defense,
    })


REF = "evaluation/audit/referee/referee_spotchecks.py"

# ---------------------------------------------------------------- spectral
add("HV-1", "spectral", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/frg/threshold.py:77-122,199",
    "Litim closed forms Phi^p_n=1/[Gamma(n+1)(1+w)^p], Phi_tilde=1/[Gamma(n+2)(1+w)^p]; QFunctional Q_n=2k^{2n+2}/[Gamma(n+1)(k^2+m^2)^p].",
    "Reuter PRD57(1998)971 definitions; Litim PRD64(2001)105007; Q_n=2k^{2(n-p+1)}Phi^p_n.",
    "All 18 (p,n,w) cases match definitional mpmath quadrature to 2.3e-16; derivative and Phi_tilde identities hold symbolically; QFunctional-2*Phi difference simplifies to 0.",
    "evaluation/audit/spectral/hv1_litim_threshold.py",
    "All beta builders consume the Litim closed forms - verified correct; factor-2 bookkeeping consistent on the whole Litim path.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-2", "spectral", "BUG", "Medium",
    "src/asymsafety/frg/threshold.py:137 (numerator), :161-171 (overflow)",
    "evaluate_numerical('Phi',...) computes the exponential-regulator threshold function per the module's own definition (numerator R_k - z R_k').",
    "Reuter-convention numerator is R0-yR0'=dtR_k/(2k^2); exact anchors Phi^1_1(0)=pi^2/6, Phi^2_1(0)=1, Phi^1_2(0)=2zeta(3) (Bose integrals).",
    "Code numerator is 2R_k+dtR_k, integrating to exactly 2Phi+2Phi_tilde (w-dependent ratio 2.6-3.4 to truth); worse, the scipy path returns NaN for every exponential Phi call (exp overflow at y>709). Phi_tilde path is correct.",
    "evaluation/audit/spectral/hv2_exponential.py + hv2b_code_integrand.py; referee: " + REF,
    "No production consumer (all 5 beta builders use Litim closed forms), but it is the only advertised numerical path for non-Litim/scheme-dependence studies.",
    "Defense (factor-2 dtR_k convention) FAILED: the integrand equals 2Phi+2Phi_tilde, not 2Phi - no constant normalization absorbs a w-dependent ratio; the same method's Litim branch and the module docstring use the Reuter norm; and every exponential Phi call returns NaN anyway. Referee reproduced NaN at (1,1,0) and Phi_tilde=1.0 exactly. BUG confirmed.")

add("HV-2b", "spectral", "INTERNAL-INCONSISTENCY", "Low",
    "src/asymsafety/frg/threshold.py:87-94",
    "_phi_symbolic returns the same-normalized Phi for non-Litim regulators as the Litim closed-form branch of Phi().",
    "Module docstring (threshold.py:7) defines the numerator as [R_k - z R_k'] = dtR_k/2; the Litim branch implements that.",
    "_phi_symbolic uses the full dtR_k numerator: numerically exactly 2.0x the Reuter-normalized Phi for (1,1,0) and (2,1,0.5) - one ThresholdFunctions object, two normalizations differing by 2. Also returns a bare integrand, not an Integral.",
    "evaluation/audit/spectral/hv2_exponential.py",
    "No callers of the symbolic non-Litim branch in src/tests; latent trap for regulator-comparison studies.",
    "Defense (legitimate Codello-style dtR_k convention) FAILED: the identical Litim R_k routed through _phi_symbolic integrates to exactly 2x the Litim branch for all (p,n,w) tried; the repo assigns the dtR_k convention to the separate QFunctional with its factor 2 explicit. INTERNAL-INCONSISTENCY confirmed.")

add("HV-3", "spectral", "BUG", "Medium",
    "src/asymsafety/geometry/decomposition.py:80",
    "S^4 scalar Laplacian degeneracy is (l+1)^2(l+2)^2/4, claimed to equal dim of the (l,0) irrep of SO(5).",
    "Rubin-Ordonez JMP 25(1984)2888: D_l=(2l+3)(l+1)(l+2)/6; anchors D_1=5, D_2=14.",
    "Code gives 1,9,36,100,225,441,784 vs exact 1,5,14,30,55,91,140; degree-4 growth violates the Weyl law on a 4-manifold; trace_on_sphere overcounts x65 at l_max=128, growing linearly in l_max; no SO(5) irrep has the code dimensions.",
    "evaluation/audit/spectral/hv34_spectrum.py; referee: " + REF,
    "trace_on_sphere -> spectral-sum convergence figure; duplicated verbatim in compute/batch/spectral.py:109; quantum/thermal LaplacianHamiltonian and tests/test_thermal.py. NOT consumed by any beta builder or fixed-point path.",
    "Defenses (cumulative Weyl counting, shifted labeling, other SO(5) irrep) all FAILED numerically; degeneracies are rep-theory integers with zero scheme freedom; the docstring even quotes the correct formula in a comment. Referee reproduced code=[1,9,36,100,225] vs exact=[1,5,14,30,55]. BUG confirmed.")

add("HV-4", "spectral", "BUG", "Medium",
    "src/asymsafety/geometry/decomposition.py:106 (cf. wrong cited formula at :100)",
    "S^4 transverse-vector degeneracy is (2l+3)(l+2)(l+1)/2, presented as the exact Rubin-Ordonez multiplicity; eigenvalue shifts {0,1,2} and a^2 handling.",
    "Rubin-Ordonez transverse vector D_l=l(l+3)(2l+3)/2 in d=4; l=1 count 10 = dim SO(5) (Killing vectors).",
    "Code l=1 gives 15, must be 10; code formula = 3 x (true scalar degeneracy). Eigenvalues (l(l+3)-{0,1,2})/a^2 and the TT degeneracy 5(l-1)(l+4)(2l+3)/6 are VERIFIED correct.",
    "evaluation/audit/spectral/hv34_spectrum.py; referee: " + REF,
    "Same consumers as HV-3; vector trace error subleading (~0.1%). No beta builder affected.",
    "Defense (harmless 'd-1 components x scalar tower' approximation) FAILED: the docstring claims the exact Rubin-Ordonez formula (false citation), no approximation is declared, and l=1 gives 15 vs 10 = dim SO(5), contradicting the file's own excluded_modes docstring. Referee reproduced 15 vs 10 and verified the TT formula exact. BUG confirmed.")

add("HV-4b", "spectral", "INTERNAL-INCONSISTENCY", "Low",
    "src/asymsafety/geometry/decomposition.py:165-171",
    "YorkDecomposition.excluded_modes: docstring says vector l=1 Killing vectors are excluded; sigma l=0,1 excluded.",
    "Excluded modes on S^4: xi l=1 (10 KVs), sigma l=0 (1), sigma l=1 (5); Lauscher-Reuter PRD65 025013.",
    "Returned dict has 'vector': [0] - l=1 NOT excluded, contradicting the docstring four lines above; 'scalar_sigma': [0,1] is correct. dof bookkeeping (5+3+2=10) verified correct.",
    "evaluation/audit/spectral/hv34_spectrum.py; referee: " + REF,
    "Dead code - zero consumers of excluded_modes/YorkDecomposition outside the defining module.",
    "Defense (alternative 'modes below min_l' semantics) FAILED: no single semantics fits the whole dict (scalar_sigma:[0,1] needs the KV-exclusion semantics) and the docstring states the KV semantics. Referee reproduced excluded_modes('vector')=[0]. INTERNAL-INCONSISTENCY confirmed; Low (dead code).")

add("HV-4c", "spectral", "BUG", "Medium",
    "src/asymsafety/frg/spectral.py:103 (crash), :132 (TT formula); compute/batch/spectral.py:79",
    "SpectralSumEvaluator.trace_on_S1xS3 evaluates foliated traces, with an S^3-multiplicity fallback.",
    "S^3 TT degeneracy 2(l-1)(l+3) (Rubin-Ordonez d=3); a non-integer degeneracy is self-contained proof of error.",
    "Any call raises NotImplementedError: hasattr(spec_S3,'_scalar_mult_S4') is always True (staticmethod on the class) so ModeSpectrum(d=3).multiplicity is hit, which raises for d!=4. The unreachable TT fallback (l-1)(l+3)(2l+1)/3 is non-integer at l=2 (25/3) and grows l^3 vs Weyl-law l^2; same wrong formula sits silently in compute/batch/spectral.py:79. S^3 scalar/vector formulas correct.",
    "evaluation/audit/spectral/hv34_spectrum.py; referee: " + REF,
    "Zero callers in src/tests (foliated betas use Litim closed forms), but the advertised foliated-trace engine interface is unusable.",
    "Defense (dead code) mitigates blast radius only, not correctness: referee reproduced the NotImplementedError crash on a direct call; degeneracies are SO(4) rep dimensions with no labeling freedom. BUG confirmed.")

add("HV-5", "spectral", "BUG", "Medium",
    "src/asymsafety/frg/heat_kernel.py:211 (TT tr E), :110 (b2)",
    "SeeleyDeWittCoefficients TT_tensor b2 = +5R/3, modeling TT as 5 scalar towers shifted by -R/6.",
    "TT is a constrained bundle: exact S^4 coefficients come from the spectrum (Lauscher-Reuter PRD65 025013 use the spectral route). Exact mode sums: -D^2 gives b2=-(5/6)R; Lichnerowicz gives -(25/6)R.",
    "Code returns b2=+5R/3 and b4=155R^2/432: wrong sign and magnitude under ANY operator convention (degeneracy of the 5-tower model contradicts SO(5) rep theory: 180 vs 35 modes at l=2). Scalar row and vector b2 (for its documented E=-Ric) verified correct.",
    "evaluation/audit/spectral/hv56_heatkernel.py; referee: " + REF,
    "HeatKernelTraceEvaluator (dead), plot_heat_kernel_coefficients figure (displays wrong TT values), tests pin scalar+vector only. b2/b4 do NOT feed any beta function or fixed point.",
    "Defense (self-consistent 5-shifted-scalar-tower scheme) FAILED: the scheme's degeneracies contradict SO(5) representation theory - facts with zero convention freedom encoded correctly in the repo's own decomposition.py; Richardson-extrapolated mode sums show +5R/3 matches no legitimate operator. Referee reproduced b2_TT=+5R/3. BUG confirmed.")

add("HV-6", "spectral", "BUG", "Medium",
    "src/asymsafety/frg/heat_kernel.py:247 (and comment :140)",
    "Vector bundle-curvature term adds +1/12 to c_Riem2 with comment 'tr(Omega^2) = R_munurhosigma R^munurhosigma'.",
    "Vassilevich Phys.Rept.388(2003)279 a4 master formula; Christensen-Duff NPB154(1979)301: spin-1 Riem^2 coefficient -11/180; tr(Omega^2) = -Riem^2 by antisymmetry.",
    "Code total vector Riem^2 coefficient +19/180 vs truth -11/180; on-sphere code b4_vector=97/270 R^2 vs exact 179/540 R^2 (verified by mode sum AND independent Vassilevich evaluation); difference exactly +R^2/36 = the Omega^2 sign flip.",
    "evaluation/audit/spectral/hv56_heatkernel.py; referee: " + REF,
    "Figures and dead HeatKernelTraceEvaluator; ghost_vector shares the bug; no beta builder consumes b4.",
    "Defense (alternative Omega/Riemann sign conventions or operator reinterpretation) FAILED: tr(Omega^2) is quadratic in Omega so its sign survives every convention; the code's own b0/b2 pin the endomorphism, leaving b4=179/540 R^2 with no freedom. Referee reproduced code b4_on_sphere('vector')=97/270 R^2. BUG confirmed.")

add("HV-6b", "spectral", "BUG", "Medium",
    "src/asymsafety/frg/heat_kernel.py:255 (bundle), :153 (Rough E2), :226-230 (TT trE^2)",
    "TT bundle term Riem2 += (1/12)*2*(d-1) = +1/2 in d=4 ('scales with number of components').",
    "Dynkin-index argument: tr(Omega^2)|_Sym2 = -(d+2)Riem^2 -> -1/2 in d=4; exact TT-on-S^4 b4 requires constrained-trace bookkeeping. Exact values: -R^2/432 (-D^2) or 719R^2/432 (Lichnerowicz).",
    "Code adds +1/2 (sign flipped); net TT b4_on_sphere = 155R^2/432, wrong by orders of magnitude vs either operator (brute-force matrix computation gives tr(Omega^2)=-144 vs code's implied +144 on unit S^4). The 'Rough' E2 approximation is not the cause (default path bypasses it, and it is Schur-exact on max-sym backgrounds).",
    "evaluation/audit/spectral/hv56_heatkernel.py; referee: " + REF,
    "Same as HV-5/HV-6: one plotting function; no beta builder, fixed point, or benchmark consumes TT b4.",
    "Defense (d=4 magnitude coincidence 2(d-1)=(d+2)=6; Schur-exact E2) FAILED on the sign: tr(Omega^2) is quadratic, basis/convention-independent; no constant endomorphism reproduces 155/432. Referee reproduced code TT b4_on_sphere=155/432 R^2. BUG confirmed.")

# ---------------------------------------------------------------- frg-beta
add("HV-7", "frg-beta", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/beta/einstein_hilbert.py:102-122",
    "eta_N=-2 at the NGFP (forced by beta_g=(2+eta_N)g=0) and beta_lambda contains the canonical -(2-eta_N)*lambda term.",
    "eta_N(FP)=-2 exactly at any NGFP with g*!=0; beta_lambda = -(2-eta_N)lam + traces (Reuter hep-th/9605030).",
    "eta_N at pinned FP = -2.000000000001 (dev 7.6e-13); beta_lambda(g=0)=-2*lambda exactly; structure matches literature.",
    "evaluation/audit/frg-beta/hv7_hv8_hv9_toolkit.py",
    "n/a (check passed). The algebraic eta_N closure is implemented correctly, independent of the wrong trace coefficients.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-7b", "frg-beta", "INTERNAL-INCONSISTENCY", "Low",
    "src/asymsafety/utils/conventions.py:29 (also frg/anomalous_dim.py:3)",
    "conventions.py states eta_N = -d_t ln G; beta/einstein_hilbert.py states eta_N = -d_t ln Z_N. Since G ~ 1/Z_N these differ by a sign.",
    "With g=G k^{d-2} and beta_g=(d-2+eta_N)g, consistency forces eta_N = +d_t ln G = -d_t ln Z_N; at FP eta_N=-2 gives G~k^{-2}.",
    "Implemented system is consistent only with eta_N=-d_t ln Z_N (einstein_hilbert.py correct); conventions.py has the opposite sign and contradicts its own cited Reuter-Saueressig convention.",
    "evaluation/audit/frg-beta/hv7_hv8_hv9_toolkit.py",
    "Docs-only (only conformal_coupling() is imported from conventions.py), but CLAUDE.md calls conventions.py the canonical RG sign reference.",
    "Defense (opposite eta convention exists in some texts) FAILED: the code's beta_g=(2+eta)g with eta(FP)=-2 forces eta=+d_t ln G by the chain rule; the alternative convention would require beta_g=(2-eta)g. One-sign docstring fix. INTERNAL-INCONSISTENCY confirmed.")

add("HV-8", "frg-beta", "BUG", "High",
    "src/asymsafety/beta/einstein_hilbert.py:99-122",
    "Docstring: coefficients correspond to the single-metric EH truncation with Type I Litim cutoff and de Donder gauge (citing Reuter/Lauscher-Reuter/Litim).",
    "That exact scheme gives lam*=0.193201, g*=0.707321, theta=1.475302+-3.043206i, 2 relevant directions (CPR 0805.2909, reproduced to 6 digits by the audit's independent re-derivation).",
    "Toolkit: g*=0.693658, lam*=0.142289 (-26.4%), theta={+0.750, -29.856} both REAL, 1 relevant direction. lam*/theta are invariant under g-rescaling, so gaps are structural: graviton volume term counts 6 modes (3A) vs 10 (5A), eta numerator/denominator wrong, lam=0 bracket sign flipped.",
    "evaluation/audit/frg-beta/eh_litim_reference.py ; evaluation/audit/frg-beta/hv7_hv8_hv9_toolkit.py; referee: " + REF,
    "CORE: CLI 'eh', quadratic and gravity_matter builders inherit the g/lambda sector, all phase portraits/GUI/figures, RGTrajectory consumers (cosmology, scattering), ~12 test files with pinned regression values.",
    "Defense (Landau-limit physical-mode counting + threshold normalization absorbing 1/pi) FAILED: ghosts kept at full -8 while the gauge modes that cancel them are dropped; the eta-term implies an impossible ghost Z_N insertion; N(lam)=4*D(lam) identically matches no heat-kernel derivation; theta={+0.75,-29.9} with one relevant direction lies outside ALL published gauge/cutoff variation. Referee reproduced FP=(0.693658,0.142289) and real exponents {0.750,-29.856}. BUG confirmed, High.")

add("HV-9", "frg-beta", "MISLEADING-CLAIM", "High",
    "src/asymsafety/validation/reuter_1998.py:36-44",
    "validate_eh_fixed_point tolerances (15% g*/lam*, 30% theta, 10% product) with docstrings implying the toolkit validates against Reuter benchmark values.",
    "REUTER_FP literature values are themselves correct (lam*=0.193, theta=1.47+-3.04i, n_relevant=2 per CPR 0805.2909).",
    "On actual toolkit output: lambda_star FAILS (26.3%), theta_real FAILS (49%), theta_imag FAILS (100%), product FAILS (27.4%); all_passed=False. tests/test_benchmarks_published.py avoids asserting all_passed, attributing gaps to 'threshold normalization and gauge' - which cannot move lam*/theta.",
    "evaluation/audit/frg-beta/hv7_hv8_hv9_toolkit.py",
    "validate_eh_fixed_point consumed only by the test that neuters it; docs advertise validation that fails 4/5 checks.",
    "Defense (legitimate scheme spread + honest regression pins) FAILED: uniform threshold renormalization leaves lambda* and theta bit-for-bit invariant; n_relevant 1 vs 2 is discrete and convention-free; the docstrings pin the exact scheme being failed. MISLEADING-CLAIM confirmed, High.")

add("HV-9b", "frg-beta", "MISLEADING-CLAIM", "Low",
    "src/asymsafety/beta/einstein_hilbert.py:140-154",
    "eh_fixed_point_litim_d4() returns {'g':0.69,'lambda':0.14,'theta_real':0.75,'theta_imag':0.0} described as 'Known approximate fixed point values ... The Reuter fixed point'.",
    "Known published EH+Litim d=4 values: g*=0.707, lam*=0.193, theta=1.475+-3.043i; theta=0.75+0i appears in no published EH study.",
    "Returned values match the TOOLKIT's own output (and silently omit theta2=-29.856), not literature; the same package's reuter_1998 holds the correct literature values - no single source matches both.",
    "evaluation/audit/frg-beta/hv7_hv8_hv9_toolkit.py",
    "Dead code: no consumers outside its own module.",
    "Defense (honest cache of toolkit FP with scheme hedge) FAILED: the docstring pins the exact scheme (Type I Litim, de Donder) with known different published values, and no published Euclidean EH scheme yields real theta=0.75. MISLEADING-CLAIM confirmed, Low (dead code).")

add("HV-10", "frg-beta", "INTERNAL-INCONSISTENCY", "High",
    "src/asymsafety/beta/quadratic.py:91-113 vs src/asymsafety/validation/codello_2009.py:25-31",
    "quadratic.py comments claim 'exact one-loop universal' beta_alpha = (1/16pi^2)*53/45 and beta_beta = (1/16pi^2)*(-196/45), matching codello_2009.ONE_LOOP_UNIVERSAL.",
    "The lambda->0 limit of the code's own expressions must equal the declared universals (exact sympy rationals).",
    "Code yields 16pi^2*beta_alpha(0) = 67/180 (not 53/45) and 16pi^2*beta_beta(0) = -329/90 (not -196/45): code contradicts its own validation dict for both couplings.",
    "evaluation/audit/frg-beta/hv10_quadratic.py; referee: " + REF,
    "build_quadratic_beta_system: CLI 'quadratic', GUI, phase portraits, figures, tests (whose order-of-magnitude check happens to pass).",
    "Defense ('schematic' label; comments as mere literature context) FAILED: the file says 'exact one-loop universal (scheme-independent)' and introduces the expressions as the Litim completion; universality forecloses every regulator/gauge defense. Referee reproduced the exact rationals 67/180 and -329/90. INTERNAL-INCONSISTENCY confirmed, High.")

add("HV-10b", "frg-beta", "BUG", "High",
    "src/asymsafety/beta/quadratic.py:98-113",
    "The rationals in beta_alpha/beta_beta implement Codello-Percacci-Rahmede one-loop higher-derivative results.",
    "Scheme-independent one-loop universals in the coefficient basis (alpha*R^2+beta*C^2 per actions/quadratic.py:101): d_t f_R2 = +5/36, d_t f_C2 = +133/20, d_t f_E = +196/45, all /(16pi^2) (Codello-Percacci hep-th/0607128; Fradkin-Tseytlin 1982; Avramidi-Barvinsky 1985).",
    "133/10 (a C^2-normalization constant) is attached to beta_ALPHA; 196/15 = 3x the Gauss-Bonnet universal also lands in beta_alpha; net constants 67/180 and -329/90 match NO legitimate universal in any basis. The validation-dict values are themselves misassigned literature (53/45 is the pure-EH Euler coefficient).",
    "evaluation/audit/frg-beta/hv10_quadratic.py; referee: " + REF,
    "All quadratic-truncation outputs (CLI, GUI, figures, quadratic_stability_3d); the one-loop universals are THE cutoff-independent content of this truncation.",
    "Defense (different invariant basis or coupling normalization) FAILED: the code's own action and Hessians pin alpha,beta as direct R^2/C^2 coefficients, so the lambda=0 limits must be +5/36 and +133/20; no basis/sign/rescale matches 67/180 and -329/90 for both couplings; one-loop 4-derivative runnings are scheme- and gauge-independent. BUG confirmed, High.")

add("HV-10c", "frg-beta", "MISLEADING-CLAIM", "Medium",
    "src/asymsafety/beta/quadratic.py:104-109; src/asymsafety/actions/quadratic.py:101",
    "quadratic.py and codello_2009.py claim 'beta_beta < 0 = asymptotically free' for the Weyl^2 coupling, 'analogous to non-abelian gauge coupling'.",
    "AF means the C^2 COEFFICIENT f_C2=1/(2lam_C) -> +infinity: d_t f_C2 = +133/20/(16pi^2) > 0. With beta = coefficient of C^2 (the code's action convention), AF requires beta_beta > 0.",
    "Toolkit beta_beta(0) < 0: the C^2 coefficient runs to -infinity, the inverse coupling GROWS - the opposite of asymptotic freedom in the code's own convention.",
    "evaluation/audit/frg-beta/hv10_quadratic.py",
    "tests/test_quadratic_beta.py and test_benchmarks_published.py assert beta_beta<0 as 'AF', locking in the inverted sign; docs propagate it.",
    "Defense (beta is CPR's inverse coupling lam_C, for which beta<0 is correct AF) FAILED on three code anchors: actions/quadratic.py:101 (+beta*C^2), the TT Hessian, and conventions.py:26 all define beta as the direct coefficient; and beta_beta is a beta-independent constant, not a marginal-coupling running. MISLEADING-CLAIM confirmed.")

add("HV-10d", "frg-beta", "MISLEADING-CLAIM", "Medium",
    "src/asymsafety/beta/quadratic.py:46-48,98-113 vs src/asymsafety/validation/codello_2009.py:16-22; visualization/phase_portrait.py:1013-1017",
    "codello_2009.QUADRATIC_FP asserts an NGFP of the 4-coupling system at (0.97,0.14,0.006,0.002) with n_relevant=4; quadratic.py See-Also calls it 'Benchmark NGFP coordinates'.",
    "A fixed point requires all four betas to vanish simultaneously; beta_alpha and beta_beta depend only on lambda.",
    "beta_alpha=0 only at lambda=-67/4654, beta_beta=0 only at lambda=47/112; intersection empty (referee confirmed) - the system as built admits NO complete fixed point, and the alpha/beta Jacobian columns are identically zero (n_relevant=4 unrealizable).",
    "evaluation/audit/frg-beta/hv10b_quadratic_fp.py; referee: " + REF,
    "FixedPointFinder on 'quadratic' (CLI scan, GUI, quadratic_stability_3d) cannot converge to a genuine 4-coupling root.",
    "Defense PARTIALLY SUCCEEDED: alpha/beta-independent one-loop betas with no finite higher-derivative FP is the literature-standard structure (Codello-Percacci 2006), the repo's tests document 'no isolated NGFP in (alpha,beta)', and QUADRATIC_FP is never asserted as a root - defeating strict internal-inconsistency. Residual defect: docstrings and phase_portrait.py advertise a 'benchmark NGFP'/'four-coupling NGFP search' the system provably lacks, and n_relevant=4 is unrealizable. Referee adjudication: reclassified INTERNAL-INCONSISTENCY -> MISLEADING-CLAIM, severity High -> Medium.")

add("HV-11", "frg-beta", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/beta/einstein_hilbert.py:105,122",
    "EH Gaussian FP: stability eigenvalues exactly {+2,-2}; beta_lambda(0,lam)=-2*lambda identically; eta_N(0,lam)=0.",
    "GFP exponents are canonical dimensions; all quantum traces proportional to g.",
    "Symbolic Jacobian at (0,0) has eigenvalues exactly {2,-2}; both identities hold symbolically.",
    "evaluation/audit/frg-beta/hv11_gaussian.py",
    "n/a (check passed).",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-11b", "frg-beta", "BUG", "High",
    "src/asymsafety/beta/einstein_hilbert.py:119-122",
    "Off-diagonal GFP slope: toolkit volume term 8g/pi*(3/(1-2lam)-4) gives d(beta_lambda)/dg at the GFP.",
    "Literature (same claimed scheme): d(beta_lambda)/dg|_GFP = +1/(2pi) = +0.159 > 0; graviton loops generate POSITIVE vacuum energy. Sign is invariant under g-rescaling.",
    "Toolkit: d(beta_lambda)/dg|_GFP = -8/pi = -2.546 < 0 - opposite sign. Cause: graviton volume counts 6 modes vs 10; 6-8<0 while 10-8>0.",
    "evaluation/audit/frg-beta/hv11_gaussian.py; referee: " + REF,
    "Sign of gravity-induced cosmological-constant flow near the GFP is flipped: every IR flow trajectory, phase portrait, separatrix, and the cosmology/scattering RG-improvement inherit it.",
    "Defense (Landau-limit alpha->0 mode counting) FAILED: even at alpha->0 the 4 gauge modes contribute +2 to the bracket, giving a positive slope; the integer mode count d(d+1)/2-2d=+2 in d=4 is invariant under gauge, cutoff, parametrization and rescalings; dropping 4 bosonic modes while keeping 8 ghosts is BRST-inconsistent. Referee reproduced slope=-8/pi symbolically. BUG confirmed, High.")

add("HV-11c", "frg-beta", "BUG", "Medium",
    "src/asymsafety/beta/einstein_hilbert.py:37-137",
    "build_eh_beta_system(d=...) advertises 'Args: d: Spacetime dimension' and is called with d by build_quadratic_beta_system.",
    "beta_g canonical term must be (d-2+eta_N)g; the matter builder (beta/matter.py:90) correctly uses d-2.",
    "build_eh_beta_system(d=3) returns expressions bit-identical to d=4 (referee confirmed symbolically); d (and gauge) silently ignored; threshold d-dependence also frozen at d=4.",
    "evaluation/audit/frg-beta/hv11_gaussian.py; referee: " + REF,
    "Any d!=4 use via build_eh_beta_system or build_quadratic_beta_system(d) silently returns d=4 physics; d!=4 is user-reachable via CLI --param d and the GUI dimension spinbox (range 3-6).",
    "Defense (d=4 placeholder disclosed in CLAUDE.md) FAILED: the canonical (d-2) coefficient follows from the module's own g=Gk^{d-2} convention with zero freedom; d!=4 is user-reachable; CLAUDE.md is developer-facing and understates the freeze (even the quantum parts are d-dependent in every scheme). BUG confirmed.")

add("HV-12", "frg-beta", "INTERNAL-INCONSISTENCY", "High",
    "src/asymsafety/frg/anomalous_dim.py:80-93; src/asymsafety/beta/matter.py:79-119",
    "build_eh_matter_beta_system(MatterContent()) (zero matter) should reduce to the pure-gravity EH system and possess the NGFP.",
    "Zero-matter limit must equal pure gravity (same truncation); a pure-gravity NGFP exists in every published EH scheme.",
    "Expressions differ symbolically from build_eh_beta_system in BOTH eta_N (sign of the quadratic term flipped vs literature) and the volume term; consequence: the matter path has NO NGFP with g>0,|lam|<0.5 even at zero matter (eta_N>0 for all g>0), and scan_matter_content reports fp_exists=False for ALL N including N=0, while build_gravity_matter_fp_system in the same file duplicates the EH builder inline and DOES find the toolkit NGFP.",
    "evaluation/audit/frg-beta/hv12_matter.py ; evaluation/audit/frg-beta/hv12b_matter_ngfp_grid.py; referee: " + REF,
    "CLI truncation 'eh_matter', GUI panels, matter-continuation figure, tests/test_matter.py; three mutually inconsistent eta_N implementations coexist.",
    "Defense (scheme dependence; an exotic root at (16.28,-0.67)) FAILED: the sign-flipped TT term is gauge-invariant in single-metric EH; no threshold relabeling flips one sign selectively; eta_N>0 strictly for g>0,|lam|<1/2 so the Reuter FP is structurally absent; the exotic root has g*lam*=-10.9 vs universal +0.12-0.14. Referee confirmed zero-matter expressions differ symbolically from pure EH. INTERNAL-INCONSISTENCY confirmed, High.")

add("HV-12b", "frg-beta", "BUG", "High",
    "src/asymsafety/actions/matter.py:192; src/asymsafety/beta/matter.py:271",
    "Matter weights in eta_N: scalar A_s=-1/(12pi), Dirac +1/(6pi), vector -(d-2)/(6pi), citing Dona-Eichhorn-Percacci.",
    "DEP 1311.2898 eq.(35)/(38): per-field A: scalar +1/(6pi), Dirac +1/(3pi), vector -2/(3pi). Scalars and fermions DEstabilize (dg*/dN>0); minimally-coupled-scalar b2=+R/6 is scheme-independent.",
    "Toolkit scalar weight has the WRONG SIGN (and half magnitude); Dirac/vector right sign, half magnitude. Numeric scan: g* DECREASES with N_s (0.6937->0.4812 at N_s=8) - scalars 'stabilize', contradicting the cited source; referee confirmed 0.6937->0.5351 at N_s=4.",
    "evaluation/audit/frg-beta/hv12_matter.py; referee: " + REF,
    "build_gravity_matter_fp_system (CLI 'gravity_matter', pinned benchmark tests, notebook 05, figures): all scalar-matter trends inverted.",
    "Defense (type-I/II cutoff scheme freedom + eta sign conventions) FAILED: scheme freedom exists only for fermions; the code's own Dirac/vector signs pin the convention; the toolkit's own ScalarFieldAction.gravity_beta_contribution yields the correct sign, contradicting the inlined constant. BUG confirmed, High.")

add("HV-12c", "frg-beta", "MISLEADING-CLAIM", "Medium",
    "src/asymsafety/validation/korver_2024.py:23-29 (validator logic :65)",
    "FOLIATED_MATTER_BOUNDS max_N_s=12, max_N_v=6, attributed to Korver-Saueressig-Wang PLB 855 (2024) 138789 [2402.01260].",
    "KSW NGFP annihilation lines: N_s+6.4N_v=23.1 (p0), N_s+4.7N_v=22.4 (p-vec): max scalars alone ~22-23, max vectors alone ~3-4.",
    "Toolkit claims 12 (paper ~23, 2x low) and 6 (paper ~3-4, 2x HIGH); the rectangle neither contains nor is contained in the paper's wedge; the integers appear in no version of the cited sources. Also validate_foliated_matter_bounds's '(within_bounds == ngfp_exists) or within_bounds' cannot fail in the regime it claims to test.",
    "evaluation/audit/frg-beta/hv12_matter.py",
    "fixed_point_plot.py shades the 'Korver 2024 bound' from this dict in published-figure output; tests only check key presence.",
    "Defense (approximate or conservative inner bounds) FAILED on every branch: N_v=6 lies OUTSIDE the cited wedge so it is not conservative; the toolkit's own gravity-matter NGFP persists past N_s=30; field counts admit no convention freedom. MISLEADING-CLAIM confirmed.")

# ---------------------------------------------------------------- amplitudes
add("HV-13a", "amplitudes", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/scattering/bootstrap.py:57",
    "veneziano implements A(s,t)=Gamma(-a_s)Gamma(-a_t)/Gamma(-a_s-a_t): crossing symmetric, correct pole tower, Regge behavior.",
    "Euler Beta B(-a_s,-a_t); reference values match mpmath to 16 digits.",
    "Worst rtol vs mpmath(dps=40) = 1.8e-15; crossing asymmetry 0.0; all 6 mass_spectrum poles confirmed; Regge ratio 0.99999999 off-axis. Only limitation: real-only domain (float cast), undocumented but not a correctness bug.",
    "evaluation/audit/amplitudes/hv13_bootstrap.py",
    "cheung_2025 validation, StringAmplitude, bridge, tests - formula correct.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-13b", "amplitudes", "BUG", "Medium",
    "src/asymsafety/scattering/bootstrap.py:101",
    "veneziano_residue: 'Res = -(-1)^n/n! * prod_{k=1..n}(alpha(t)+k)' at the alpha(s)=n pole.",
    "Gamma(-a_s)~(-1)^{n+1}/[n!(a_s-n)] gives Res_{a_s=n}A = -(1/n!)prod(a_t+k) for ALL n (sign uniform in n in every genuine convention).",
    "Numeric residue of the code's own veneziano: n=1,t=-2.7 gives +0.700000; code returns -0.700000 (n=3 likewise flipped; n=0,2 agree). The alternating -(-1)^n matches no convention.",
    "evaluation/audit/amplitudes/hv13_bootstrap.py; referee: " + REF,
    "cheung_2025 higher_spin_cancellation and tests use it only at its zeros - latent: no downstream numeric output flips, but the public API contradicts the amplitude it describes.",
    "Defense (u-channel variable under the tachyonic constraint, where the identity holds) FAILED: the argument is named and consumed as t everywhere, and the module's own kinematics are massless where the identity breaks. Referee reproduced numeric +0.70 vs code -0.70 at n=1. BUG confirmed.")

add("HV-13c", "amplitudes", "BUG", "High",
    "src/asymsafety/scattering/bootstrap.py:83",
    "virasoro_shapiro implements the closed-string VS amplitude with denominators Gamma(a_s+a_t)Gamma(a_t+a_u)Gamma(a_u+a_s) ('symmetric closed-string convention').",
    "A_VS = prod Gamma(-a_i)/prod Gamma(1+a_i); the positive-sum denominator form is equivalent ONLY on the constraint surface sum a_i = -1 (Virasoro 1969; Polchinski 6.2.39-40; arXiv 2508.09246).",
    "With massless kinematics (alpha0=0) the code is IDENTICALLY 1; with defaults (sum a_i=3) it equals the rational function 1/prod[(-a)(1-a)(2-a)]: 3 poles per channel, no Regge tower, pure s^-9 fixed-angle tail. Referee reproduced all three facts including the mismatch vs true VS.",
    "evaluation/audit/amplitudes/hv13_bootstrap.py; referee: " + REF,
    "ScatteringBridge's default string side, cheung_2025 ultrasoft check, three test files, docs - the headline 'AS vs string bootstrap' comparison compares AS against a rational function, not Virasoro-Shapiro.",
    "Defense (Virasoro's original 1969 positive-sum denominator form) FAILED: that form requires Gamma(+a) numerators and the constraint surface; the code pairs the opposite-sign numerator with that denominator, and Gamma(a_j+a_k)=Gamma(1+a_i) would require a_s=a_t=a_u. No convention or kinematic surface legitimizes it. BUG confirmed, High.")

add("HV-13d", "amplitudes", "INTERNAL-INCONSISTENCY", "Medium",
    "src/asymsafety/scattering/bootstrap.py:131",
    "StringAmplitude(kind='virasoro_shapiro') builds massless 2->2 kinematics (u=-s-t, m=0) and presents the result as the closed-string amplitude.",
    "Massless external states require alpha0=0 (sum a_i=0); the bosonic closed-tachyon convention requires s+t+u=-16/alpha' (sum a_i=-1). No convention yields sum=3 with massless kinematics.",
    "Defaults alpha0=1.0, alphap=0.25 with s+t+u=0 give sum a_i=3, on which the Gamma(a_i+a_j) form degenerates to the rational function of HV-13c.",
    "evaluation/audit/amplitudes/hv13_bootstrap.py",
    "Same as HV-13c: bridge default, cheung_2025 ultrasoft check, three test files.",
    "Defense (genuine closed-tachyon convention) FAILED: valid only on s+t+u=-16/alpha', unreachable since m is real and eval defaults u=-s-t; on the shipped surface the amplitude is a different function class (no Regge tower), not a small deformation. INTERNAL-INCONSISTENCY confirmed.")

add("HV-13e", "amplitudes", "MISLEADING-CLAIM", "High",
    "src/asymsafety/validation/cheung_2025.py:46",
    "validate_bootstrap 'validates the bootstrap reference amplitudes' against Strings-from-Almost-Nothing properties; all_passed=True.",
    "True VS falls super-polynomially at fixed angle (hard-scattering law); the implemented rational function falls as s^-9 exactly.",
    "All 5 checks pass for the wrong reasons: VS-crossing is 0.0 for ANY symmetric Gamma combination; higher_spin_cancellation is circular (blind to the HV-13b sign bug, returns exactly 0 even with the sign flipped); ultrasoft_falloff passes only via Gamma-pole spikes distorting the fit window - on a pole-free window the implemented 'VS' has constant slope -9.00 and ultrasoft=False.",
    "evaluation/audit/amplitudes/hv13_bootstrap.py",
    "tests/test_benchmarks_published.py asserts these passes; docs cite this module as published-comparison evidence; ScatteringBridge verdicts inherit 'string_ultrasoft'.",
    "Defense (certifies only properties of its reference formulas) FAILED on the decisive points: a correctly implemented VS passes ultrasoft_falloff on every window, the shipped one only on the pole-contaminated window; the 'ultrasoft' verdict is false of the code. MISLEADING-CLAIM confirmed, High.")

add("HV-14a", "amplitudes", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/scattering/amplitude.py:89",
    "GravitonMediatedAmplitude.eval(s,t,u) is fully S3-crossing symmetric for identical scalars, dressed and undressed.",
    "Identical-scalar graviton exchange M ~ su/t+tu/s+st/u is permutation symmetric; per-channel G(|x|) preserves it.",
    "Worst relative asymmetry over all 6 permutations at 12 random kinematics: 1.6e-16.",
    "evaluation/audit/amplitudes/hv14_amplitude.py",
    "consistency.crossing, bridge foundational checks.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-14b", "amplitudes", "DISCREPANT-VS-LITERATURE", "Medium",
    "src/asymsafety/scattering/amplitude.py:7",
    "Docstring: 'With the standard four-scalar graviton-exchange numerators (kappa^2/4 = 8piG): M_GR = -8piG_N[tu/s + su/t + st/u]' (overall minus presented as standard).",
    "M = +8piG su/t per channel (mostly-minus, S=1+iT, Born map giving attractive Newton limit); DKRS PRL 125,181301 (2007.04396) use A=+8piG tu/s with the same signature.",
    "Code M_GR = -1.0x the +8piG sum; forward amplitude negative, whereas the attraction chain requires positive. No numeric consequence inside the toolkit (all consumers use |M|), but a user extracting a potential with the stated conventions would get gravitational repulsion.",
    "evaluation/audit/amplitudes/hv14_amplitude.py; defense probe: evaluation/audit/defense/hv14b_sign_chain.py",
    "No internal numeric consequence; external interference/potential extraction affected.",
    "Defense (signature or S=1-iT convention) FAILED: explicit 4-vector computation gives +8piG su/t in BOTH signatures under S=1+iT; DKRS itself uses the same signature with opposite sign; Weinberg's S=1-iT would flip it but is nowhere declared and contradicts consistency.py's normalizations. A global sign of M remains convention-dependent, so DISCREPANT-VS-LITERATURE (not BUG) confirmed; fix is to declare or flip the convention.")

add("HV-14c", "amplitudes", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/scattering/amplitude.py:129",
    "IR limit: dressed amplitude reduces to classical GR, M_dressed/M_GR -> 1 as s -> 0.",
    "G(k)->G_N in the IR implies pointwise M_improved/M_GR -> 1.",
    "Ratio within 1e-2 of 1 over six decades s=1e-8..1e-2; residual ~7e-4 from the G_N reference point.",
    "evaluation/audit/amplitudes/hv14_amplitude.py",
    "draper_2020 ir_newtonian_limit validation - confirmed genuine.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-14d", "amplitudes", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/validation/knorr_2026.py:54",
    "UV softening: dressed fixed-angle |M| growth exponent ~0 vs GR ~1; FixedScale retains ~1 - the safe/unsafe dichotomy.",
    "G(k)=g*/k^2 with per-channel k^2=|x| makes each term s-independent at fixed angle; frozen k keeps GR growth (Knorr 2602.21285).",
    "Fits over s in [1e4,1e8]: dressed +0.0000, GR +1.0000, 'unsafe' +1.0000; UV plateau matches the analytic -8pi g* prediction to 7 digits.",
    "evaluation/audit/amplitudes/hv14_amplitude.py",
    "Headline AS claim of the scattering subsystem; epistemic caveat (RG-improvement, not first principles) properly stated.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-14e", "amplitudes", "CORRECT-WITH-DOCUMENTED-APPROXIMATION", "Low",
    "src/asymsafety/scattering/consistency.py:81",
    "unitarity check: AS partial waves bounded (~0 growth) vs GR (~1), with disclosed angular cutoff cos_max=0.99; literal |a_l|<=1 not claimed.",
    "GR cutoff partial wave diverges only logarithmically; DKRS a_0=Gs/12, a_2=-Gs/60; cutoff disclosed in module docstring.",
    "No conclusion flips across cos_max in {0.9,0.99,0.999}; projector normalization validated against DKRS. Two undisclosed nuances, neither flipping conclusions: dressed forward divergence is ~1/delta (stronger than GR's log); n_theta=400 under-resolves cos_max>=0.999.",
    "evaluation/audit/amplitudes/hv14_amplitude.py",
    "bridge.verify 'partial_waves_bounded' - robust to cutoff choice.",
    "Uncontested; no skeptic defense mounted. Referee accepts CORRECT-WITH-DOCUMENTED-APPROXIMATION.")

add("HV-14f", "amplitudes", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/scattering/consistency.py:236 (disclosure :27-28)",
    "consistency.crossing tests s<->t symmetry of M and is counted among bridge.verify's foundational checks.",
    "Crossing is exact by construction for the per-channel form-factor identification; the global k^2=s improvement would violate it (and does fail the check, rel. residual 1.0).",
    "eval(s,t,u) vs eval(t,s,u) are bit-identical for every toolkit amplitude (residual exactly 0.0); the check is non-vacuous only against external crossing-violating scale choices.",
    "evaluation/audit/amplitudes/hv14_amplitude.py; defense probe: evaluation/audit/defense/hv14f_defense.py; referee: " + REF,
    "bridge.verify foundational_checks.crossing - by-construction pass, disclosed.",
    "DEFENSE SUCCEEDED: under the bootstrap convention crossing is an axiom and manifest by-construction satisfaction counts as passing; the manifest symmetry encodes a real scheme choice (per-channel identification vs crossing-violating global k^2=s, which the check does fail); disclosure is at the definition site ('exact by construction' - referee confirmed in source); the string references are certified identically, so the standard is uniform. Referee adjudication: upgraded MISLEADING-CLAIM -> VERIFIED-CORRECT.")

# ---------------------------------------------------------------- observables
add("HV-15a", "observables", "VERIFIED-CORRECT", "Low",
    "src/asymsafety/cosmology/rg_improved_bh.py:98",
    "RGImprovedSchwarzschild: G(r)->G_N at large r; two/one/zero horizons across critical mass; via G(r)=g(k(r))/k(r)^2, k=xi/r.",
    "Bonanno-Reuter closed form (hep-th/0002196, gamma=0): r_pm=G0M[1+-sqrt(1-omega_bar/(G0M^2))], M_cr=1/g* here.",
    "G(100) rel err 3.5e-4; horizons at M=2 match exact to 1e-3; critical_mass()=1.42898 vs exact 1.428571; bracketing robust to M=(1+1e-6)M_cr.",
    "evaluation/audit/observables/hv15a_bh.py",
    "cosmology visualization, figures, notebook 04, tests; form_factor doc xref.",
    "Uncontested; no skeptic defense mounted. Referee accepts VERIFIED-CORRECT.")

add("HV-15a-b", "observables", "MISLEADING-CLAIM", "Medium",
    "src/asymsafety/cosmology/rg_improved_bh.py:17",
    "Module docstring: 'f(r) -> 1 as r -> 0 (de Sitter core) -> no curvature singularity'; notebook 04 repeats 'no curvature singularity'.",
    "BR's de Sitter core requires the softened distance d(r)~r^{3/2} (k~r^{-3/2}); plain k=xi/r gives f=1-(2M/omega_bar)r with R=12M/(omega_bar r) divergent - 'much milder', not removed.",
    "Default InverseDistanceScale: f'(0)=-2.80 != 0; numerical Ricci R(1e-3)=16800 = 12g*M/r diverging as 1/r. GeodesicDistanceScale gives classical-strength singularity. No implemented scale realizes BR's d(r).",
    "evaluation/audit/observables/hv15a_bh.py; defense probe: evaluation/audit/defense/hv15ab_defense.py",
    "Notebook 04 and docs/visualization-guide.md:371 propagate the false gloss; tests only assert f(0)->1.",
    "Defense (paraphrase of BR's headline; ProperDistanceScale flat core) FAILED: the r^{3/2} exponent is convention-free (curvature invariants are diffeo-invariant); implementing BR's actual d(r) - absent from the codebase - gives finite R(0), proving the missing ingredient; the ProperDistance flat core is a clamp artifact, not de Sitter, not default. MISLEADING-CLAIM confirmed.")

add("HV-15b", "observables", "INTERNAL-INCONSISTENCY", "Medium",
    "src/asymsafety/cosmology/rg_improved_flrw.py:143",
    "RGImprovedFLRW implements Bonanno-Reuter RG-improved cosmology (cites hep-th/0106133): Friedmann-I with running G,Lambda.",
    "hep-th/0106133: Bianchi identity forces rhodot+3H(rho+p) = -(Gdot rho + Lamdot/8pi)/G; BR impose standard conservation AND the constraint Lamdot+8 pi rho Gdot=0, which fixes xi.",
    "integrate() hardcodes standard conservation with arbitrary xi (default 1): Bianchi residual up to 0.11 (default HubbleScale) and 1.1e4 (k=1/t); BR constraint violated by factors 1.9-690; module contains no mention of Bianchi/conservation/constraint.",
    "evaluation/audit/observables/hv15b_flrw.py; defense probes: evaluation/audit/defense/hv15b_defense.py, hv15b_defense2.py",
    "cosmology visualization, figures, tests (smoke only). No published-comparison test consumes it.",
    "Defense (line 143 is BR's own postulate) PARTIALLY FACTUAL - with the BR-consistent xi the code reproduces BR's fixed-point cosmology exactly (residual 1.5e-6) - but FAILED overall: the constraint fixing xi is never imposed or documented, xi defaults to 1, and the default HubbleScale path is inconsistent for ANY xi (hardwired k=1/t lookup in H_est, Lambda/3 omitted). INTERNAL-INCONSISTENCY confirmed.")

add("HV-15c", "observables", "CORRECT-WITH-DOCUMENTED-APPROXIMATION", "Low",
    "src/asymsafety/beta/foliated.py:146",
    "beta_lambda_ADM factors as g*(lambda_ADM-1)*h(lambda), so lambda_ADM=1 is an exact fixed plane.",
    "h_factor = (Phi^1_1(-2l)-Phi^1_1(0))/(2pi) = lambda/(pi(1-2lambda)) for Litim - algebra confirmed by sympy.",
    "Factorization holds exactly, but by construction: foliated.py:141-146 inserts the (lambda_ADM-1) factor by hand ('use the known result'), not from a K^2-projection.",
    "evaluation/audit/observables/hv15c_foliated.py",
    "CLI 'foliated', GUI foliated portraits, tests pin the structure.",
    "Uncontested; no skeptic defense mounted. Referee accepts CORRECT-WITH-DOCUMENTED-APPROXIMATION (schematic nature disclosed inline; the factor test is vacuous as physics validation).")

add("HV-15c-b", "observables", "INTERNAL-INCONSISTENCY", "High",
    "src/asymsafety/beta/foliated.py:52",
    "foliated.py docstring: 'The NGFP exists in the foliated formulation'; benchmark coordinates g*~0.96, lambda*~0.20, lambda_ADM*=1.",
    "Published foliated NGFP (MRS PRL 106 251302 Eq.(10)): Euclidean g*=0.19, lambda*=0.31, theta=1.07+-3.31i.",
    "beta at the claimed benchmark = (3.258, 0.184, 0) - not remotely a root (referee reproduced). 81-seed multi-start finds only the Gaussian FP; beta_g=0 with g>0 requires lambda<-1/2; the sole non-Gaussian root is (16.278,-0.671,1.0) - nothing like any published foliated FP.",
    "evaluation/audit/observables/hv15c_foliated.py ; hv15c_supplement.py; referee: " + REF,
    "CLI-exposed truncation, GUI 3D portraits, notebooks; tests admit the Gaussian collapse while module/validation docstrings still advertise the NGFP.",
    "Defense (regulator/threshold scheme freedom) FAILED structurally: A_fol>0 and B_fol<0 for ANY admissible regulator at lambda>=0, so eta=-2 is unreachable and no physical NGFP can exist; benchmark coordinates are pinned to this builder by the docstring. INTERNAL-INCONSISTENCY confirmed, High.")

add("HV-15c-c", "observables", "MISLEADING-CLAIM", "High",
    "src/asymsafety/validation/manrique_2011.py:29",
    "FOLIATED_EH_FP g*=0.96, lambda*=0.20, lambda_ADM*=1 presented as Manrique-Rechenberger-Saueressig PRL 106 251302 results; lambda_ADM=1 'at the NGFP (full-Diff restoration)'.",
    "MRS Eq.(10): Euclidean g*=0.19, lambda*=0.31; Lorentzian g*=0.21, lambda*=0.30. MRS's truncation contains NO lambda_ADM coupling - lambda_ADM=1 is imposed by the diffeo-invariant ansatz, not derived.",
    "g* off by ~5x, lambda* by ~35%; g*lambda* product off 3.3x; the 20% fp_rtol cannot bridge it. LORENTZIAN_FP (1.57,0.12) also disagrees with its attributed source.",
    "evaluation/audit/observables/hv15c_foliated.py",
    "test_benchmarks_published.py::TestManriqueFoliated only checks the dict's internal values (self-certified); beta/foliated.py docstrings propagate it.",
    "Defense (garbled citation of Biemans-Platania-Saueressig 2017, whose (0.90,0.24) is close to the dict) FAILED: the file attributes the values to MRS PRL 106 251302, read verbatim as (0.19,0.31); MRS excludes any lambda_ADM coupling; and the benchmark is not even a root of the toolkit's own betas. MISLEADING-CLAIM confirmed, High.")

add("HV-15c-d", "observables", "INTERNAL-INCONSISTENCY", "Medium",
    "src/asymsafety/beta/foliated.py:143 (vs tests/test_foliated_beta.py:96-98)",
    "At the NGFP lambda_ADM -> 1, 'restoring full diffeomorphism invariance', implying the lambda_ADM direction is UV-attractive at lambda_ADM=1.",
    "ddelta/dt = M[2,2] delta with t=ln k; M[2,2]>0 means lambda_ADM runs AWAY from 1 toward the UV.",
    "d(beta_lambda_ADM)/d(lambda_ADM) = g*lambda/(pi(1-2lambda)) > 0 for 0<lambda<1/2; at the claimed benchmark M[2,2]=+0.10186 (referee reproduced): UV-REPULSIVE, the opposite of dynamical restoration. The repo's own test demands the attractive signs but its assertion (above*below<0) is sign-blind and masks the violation.",
    "evaluation/audit/observables/hv15c_supplement.py; referee: " + REF,
    "Same consumers as HV-15c; GUI restoration narrative; no test pins the eigenvalue sign.",
    "Defense (fixed-plane location claim + IR-attractive restoration mechanism, cf. Knorr PLB 2019) acquits the docstring in isolation but FAILED against the repo's own test, which demands UV-attraction while the code yields the opposite sign at the test's own point. INTERNAL-INCONSISTENCY confirmed.")

add("HV-15d", "observables", "MISLEADING-CLAIM", "Medium",
    "src/asymsafety/transforms/bridge/cross_analogue.py:97",
    "CrossAnalogueBridge.verify_commutativity() 'checks all paths give consistent critical exponents' across the RG/transfer-matrix/resolvent diagram (tol=0.1).",
    "These are exact linear-algebra identities (eig(expm(M dt)) etc.); any tolerance >>1e-10 on linear inputs reflects implementation slack, not physics.",
    "ResolventOperator.poles() returns stability.eigenvalues verbatim (zero computation, deviation exactly 0.0); transfer path re-diagonalizes expm of the SAME Jacobian (spectral-mapping identity, deviation 5.4e-13); tol=0.1 is >=11 orders looser than anything reachable; hydraulic best-effort, quantum excluded.",
    "evaluation/audit/observables/hv15d_bridge.py; defense probe: evaluation/audit/defense/hv15d_defense.py",
    "tests/test_bridge.py, gauge_higgs bridge, bridge_diagram figure; README/figure captions claim cross-domain verification.",
    "Defense (legitimate regression check, locally honest docstrings, stiff-truncation tolerance budget) FAILED: the resolvent path performs no computation; no supported system exercises the tolerance within 11 orders; README and bridge_diagram.py:269 claim 'all paths yield consistent theta_i' - cross-domain verification the code never performs. MISLEADING-CLAIM confirmed (the math actually performed is correct).")

add("HV-15d-b", "observables", "MISLEADING-CLAIM", "Low",
    "src/asymsafety/transforms/linear/koopman.py:221",
    "ClassicalKoopmanOperator.compare_with_stability 'compares leading Koopman eigenvalues with stability eigenvalues'.",
    "A genuine check compares log(eig(K_EDMD))/dt against eigenvalues of M (independent EDMD agrees to 5.5e-13; repo's own compute_edmd reproduces them to 1.3e-5).",
    "Lines 215+221 set both comparands to np.sort(stability.eigenvalues): max_deviation identically 0.0 (referee reproduced), agrees always True, no EDMD output consulted; returns agrees=True even for garbage stability objects.",
    "evaluation/audit/observables/hv15d_bridge.py; defense probe: evaluation/audit/defense/hv15db_defense.py; referee: " + REF,
    "Not used by verify_commutativity; near-dead validation path that can never fail.",
    "Defense (the theoretical identity makes the numbers analytically correct) FAILED: the method is presented as a validation but validates nothing; the sibling matrix_exponential.compare_with_flow does a genuine comparison, so this is not a house convention; the docstring half-disclosure does not cure the method name and 'koopman_edmd' labels. MISLEADING-CLAIM confirmed.")

add("HV-15d-c", "observables", "BUG", "High",
    "src/asymsafety/transforms/bridge/gauge_higgs.py:157",
    "charged_fp_guess returns the quartic root 'continuously connected to the decoupled Wilson-Fisher FP as alpha->0'; module/validation claim one-loop nu(Nf) consistency with Bonati's nu = 1 - 9.727/Nf.",
    "Quadratic-root limit is self-contained algebra: as alpha->0 the roots -> {0, eps/(N+4)}; the WF-connected, charged (IR-stable, one relevant direction) root is u_+ (HLM PRL 32,292); large-n 3D WF has nu->1, matching Bonati 2410.05823.",
    "Code returns u_-: toolkit nu(30,40,60)=(0.5655,0.5330,0.5174) DECREASING toward 1/2 (referee reproduced); its own validate_nu_vs_nf fails (rel err up to 0.36, monotonic=False) and the FP is tricritical (2 relevant directions). With u_+: nu=(0.666,0.754,0.832), within 1.5% of Bonati. One-character-class fix.",
    "evaluation/audit/observables/hv15d_bridge.py (section D); defense probes: evaluation/audit/defense/hv15dc_defense.py, hv15dc_pipeline.py; referee: " + REF,
    "GaugeHiggsAnalogue.nu, correlation_length_exponent; docs/cross-analogue-gauge-higgs.md documents the inverted trend as expected; tests were loosened to accept either monotone direction, pinning the wrong branch.",
    "Defense (deliberately qualitative one-loop proxy; scheme dependence) FAILED: the same scheme with the plus root agrees with Bonati MC to ~4%, so the discrepancy is root choice, not scheme; the code's own selection criterion algebraically picks u_+; the relevant-direction count (2 vs 1) is convention-free. BUG confirmed, High.")

with open("/root/cdev/exasymptoticsafety/evaluation/audit/findings.json", "w") as f:
    json.dump(F, f, indent=2, ensure_ascii=False)
print(f"wrote {len(F)} findings")

from collections import Counter
print(Counter(x["classification"] for x in F))
print(Counter(x["severity"] for x in F))
