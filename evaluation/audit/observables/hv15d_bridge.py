"""HV-15d: cross-analogue bridge audit.

(A) Run CrossAnalogueBridge.verify_commutativity() at the EH NGFP; record
    the ACTUAL per-path deviations vs the 0.1 tolerance.
(B) Independent plain-numpy verification of the bridge identities
    (spectral mapping for expm, resolvent poles, EDMD-linear) at machine
    precision, including the log-branch aliasing caveat |Im mu| dt < pi.
(C) Tautology analysis: what does each path actually recompute?
(D) Bonati/gauge-Higgs sanity: nu(Nf) trend of the implemented one-loop
    AHM system vs validation/bonati_2025.py monotonicity claim
    (nu = 1 - 9.727/Nf increasing), and the quartic root-branch choice.
"""
import sys
import numpy as np
from scipy.linalg import expm, svdvals

sys.path.insert(0, "/root/cdev/exasymptoticsafety")

from asymsafety.beta.einstein_hilbert import build_eh_beta_system  # noqa: E402
from asymsafety.analysis.fixed_points import FixedPointFinder  # noqa: E402
from asymsafety.analysis.stability import analyze_stability  # noqa: E402
from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge  # noqa: E402

system = build_eh_beta_system()
finder = FixedPointFinder(system)
fp = finder.find_fixed_point({"g": 0.7, "lambda": 0.19}, tol=1e-9)
stab = analyze_stability(system, fp)
print("EH NGFP:", {k: f"{v:.12f}" for k, v in fp.location.items()})
print("theta (direct eig of -M):", np.round(stab.critical_exponents, 10))

print("=" * 72)
print("(A) verify_commutativity(tol=0.1):")
bridge = CrossAnalogueBridge(system, fp, stab)
res = bridge.verify_commutativity(tol=0.1)
for m, vals in res["critical_exponents"].items():
    print(f"  {m:16s}: {vals}")
for m, a in res["agreements"].items():
    print(f"  {m:16s}: max_deviation = {a['max_deviation']:.3e} "
          f"(tol=0.1 -> {a['max_deviation']/0.1:.1e} of budget)")
print("  all_agree:", res["all_agree"])

print("=" * 72)
print("(B) independent numpy verification at machine precision")
M = system.jacobian_numerical(fp.location)
mu = np.linalg.eigvals(M)
theta_direct = np.sort_complex(-mu)
print("  mu(M) =", np.round(mu, 12))

for dt in [0.1, 0.5, 1.0, 2.0]:
    T = expm(M * dt)
    mu_T = np.log(np.linalg.eigvals(T)) / dt
    dev = np.max(np.abs(np.sort_complex(-mu_T) - theta_direct))
    alias = np.max(np.abs(mu.imag)) * dt
    print(f"  transfer dt={dt}: max|theta_T - theta| = {dev:.3e}  "
          f"(|Im mu|dt = {alias:.3f}, branch-safe iff < pi={np.pi:.3f})")

# resolvent: poles located independently by minimizing sigma_min(sI - M)
n = M.shape[0]
for m in mu:
    smin_at = svdvals(m * np.eye(n) - M)[-1]
    eps = 1e-3
    smin_off = svdvals((m + eps) * np.eye(n) - M)[-1]
    print(f"  resolvent: sigma_min(sI-M) at s=mu = {smin_at:.3e} "
          f"(off by 1e-3: {smin_off:.3e}) -> pole exactly at mu")

# EDMD with linear dictionary on the time-dt map of ydot = My
rng = np.random.default_rng(0)
dt = 0.1
T = expm(M * dt)
X = rng.normal(size=(n, 40))  # 40 random ICs, full row rank
Y = T @ X
K = Y @ np.linalg.pinv(X)
mu_K = np.log(np.linalg.eigvals(K)) / dt
dev = np.max(np.abs(np.sort_complex(-mu_K) - theta_direct))
print(f"  EDMD(linear dict, exact linear data): max dev = {dev:.3e}")

print("=" * 72)
print("(C) tautology analysis (code inspection, confirmed numerically):")
import asymsafety.transforms.linear.resolvent as rmod  # noqa: E402
poles = rmod.ResolventOperator(stab).poles()
print("  ResolventOperator.poles() is stab.eigenvalues:",
      poles is stab.eigenvalues)
print("  rg path = stab.critical_exponents = -stab.eigenvalues (analyze_stability:101)")
print("  => resolvent-path deviation is identically",
      float(np.max(np.abs(np.sort(-poles.real)[::-1]
                          - np.sort(stab.critical_exponents.real)[::-1]))))
from asymsafety.transforms.linear.koopman import ClassicalKoopmanOperator  # noqa: E402
ck = ClassicalKoopmanOperator(system, fp)
cmp_res = ck.compare_with_stability(stab)
print("  ClassicalKoopmanOperator.compare_with_stability max_dev =",
      cmp_res.max_deviation, "(compares stab.eigenvalues with itself; koopman.py:215/221)")

print("=" * 72)
print("(D) Bonati / gauge-Higgs nu(Nf)")
from asymsafety.transforms.bridge.gauge_higgs import (  # noqa: E402
    GaugeHiggsAnalogue, charged_fp_guess, correlation_length_exponent,
)
from asymsafety.validation.bonati_2025 import (  # noqa: E402
    BONATI_SU2_MC, large_nf_nu, validate_nu_vs_nf,
)

print(f"  {'Nf':>4} {'u-(code)':>10} {'u+(WF branch)':>13} "
      f"{'nu(code)':>9} {'nu(bridge)':>10} {'nu(u+)':>8} {'1-9.727/Nf':>10} {'MC':>6}")
computed = {}
for Nf in [30, 40, 60, 100, 200]:
    eps = 1.0
    a_st = eps / Nf
    A, B, C = Nf + 4, eps + 6 * a_st, 9 * a_st**2
    disc = B**2 - 4 * A * C
    um = (B - disc**0.5) / (2 * A)
    up = (B + disc**0.5) / (2 * A)
    nu_code = correlation_length_exponent(Nf)
    th_up = 2.0 - (Nf + 2) * up + 6 * a_st
    nu_up = 1.0 / th_up
    try:
        nu_bridge = GaugeHiggsAnalogue(N=Nf).nu
    except Exception as e:
        nu_bridge = float("nan")
    mc = BONATI_SU2_MC.get(Nf, {}).get("nu", float("nan"))
    lnf = large_nf_nu(Nf)
    computed[Nf] = nu_code
    print(f"  {Nf:>4} {um:>10.6f} {up:>13.6f} {nu_code:>9.4f} "
          f"{nu_bridge:>10.4f} {nu_up:>8.4f} {lnf:>10.4f} {mc:>6}")

print("\n  charged_fp_guess returns (B - sqrt(disc))/(2A) [gauge_higgs.py:157];")
print("  as alpha->0 roots -> {B/A (Wilson-Fisher), 0}: the docstring's")
print("  'branch continuously connected to WF' is the PLUS branch.")
print("  validate_nu_vs_nf on toolkit nu(Nf):",
      {k: (v if not isinstance(v, dict) else
           {kk: vv for kk, vv in v.items() if kk in ('passed', 'relative_error')})
       for k, v in validate_nu_vs_nf(
           {n: correlation_length_exponent(n) for n in (30, 40, 60)}).items()})
print("  validate_nu_vs_nf on u+ branch nu(Nf):",
      {k: (v if not isinstance(v, dict) else
           {kk: round(vv, 4) if isinstance(vv, float) else vv
            for kk, vv in v.items() if kk in ('passed', 'relative_error')})
       for k, v in validate_nu_vs_nf(
           {n: 1.0 / (2.0 - (n + 2) * ((1 + 6 / n + (1 + 12 / n - 108 / n**2
            - 24 / n)**0.5 * 0 + ((1 + 6.0 / n)**2 - 36.0 * (n + 4) / n**2)**0.5)
            / (2 * (n + 4))) + 6.0 / n) for n in (30, 40, 60)}).items()})

# stability of the code's FP: count relevant directions
print("\n  relevant directions at the code's FP (charged FP should have 1):")
for Nf in (30, 60):
    ga = GaugeHiggsAnalogue(N=Nf)
    th = ga.stability.critical_exponents
    print(f"   Nf={Nf}: theta = {np.round(th.real, 4)} -> "
          f"{int(np.sum(th.real > 0))} relevant")
