"""HV-15a: Bonanno-Reuter RG-improved Schwarzschild audit.

Synthetic AS trajectory g(t) = g* e^{2t}/(1+e^{2t}) (tests/conftest.py
make_as_trajectory) => closed forms with k = 1/r (InverseDistanceScale, xi=1):
    g(1/r) = g*/(1+r^2),  G(r) = g* r^2/(1+r^2),
    f(r)   = 1 - 2 g* M r/(1+r^2)
i.e. exactly BR gamma=0 with G0 = g*, omega-bar*G0 = 1:
    r_pm = g* M [1 +- sqrt(1 - 1/(g*^2 M^2))],  M_cr = 1/g*.
Small r: f = 1 - 2 g* M r + O(r^3)  -> f(0)=1 but f'(0) != 0,
Ricci R = 2(1-f)/r^2 - 4 f'/r - f'' = 12 g* M /(2r) *... -> 6c/r with c=2g*M.
(NOT a de Sitter core; R ~ 1/r divergence, milder than classical r^-3.)
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety")
from tests.conftest import make_as_trajectory  # noqa: E402

from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild  # noqa: E402
from asymsafety.cosmology.scale_identification import InverseDistanceScale  # noqa: E402

GSTAR = 0.7
traj = make_as_trajectory(g_star=GSTAR, lam_star=0.2)

print("=" * 70)
print("(i) IR limit: G(r) -> G_N = g* = 0.7 (trajectory units, k0=1)")
bh = RGImprovedSchwarzschild(trajectory=traj, M=2.0)
for r in [10.0, 100.0, 1000.0]:
    G = bh.G(r)
    exact = GSTAR * r**2 / (1 + r**2)
    print(f"  r={r:8.1f}  G(r)={G:.10f}  exact={exact:.10f} "
          f"rel_to_GN={abs(G - GSTAR)/GSTAR:.2e}")

print("=" * 70)
print("(ii) small-r behavior of f(r) (default InverseDistanceScale k=1/r)")
M = 2.0
bh = RGImprovedSchwarzschild(trajectory=traj, M=M)
rs = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
fs = np.array([bh.lapse(float(r)) for r in rs])
print("  r, f(r), (1-f)/r [expect -> 2 g* M = %.3f]" % (2 * GSTAR * M))
for r, f in zip(rs, fs):
    print(f"  {r:.1e}  {f:.10f}  {(1-f)/r:.6f}")

# Ricci scalar numerically: R = 2(1-f)/r^2 - 4 f'/r - f''
def ricci(r, h=None):
    h = h or r * 1e-4
    f0 = bh.lapse(r)
    fp_ = (bh.lapse(r + h) - bh.lapse(r - h)) / (2 * h)
    fpp = (bh.lapse(r + h) - 2 * f0 + bh.lapse(r - h)) / h**2
    return 2 * (1 - f0) / r**2 - 4 * fp_ / r - fpp

print("  Ricci R(r) [expect 6c/r with c=2g*M=%.2f -> %.2f/r, divergent]:"
      % (2 * GSTAR * M, 12 * GSTAR * M))
for r in [1e-1, 1e-2, 1e-3]:
    print(f"    r={r:.0e}  R={ricci(r):10.3f}   6c/r={12*GSTAR*M/r:10.3f}")
print("  => f(0)=1 but f'(0) = -2g*M != 0: linear (NOT de Sitter r^2) core;")
print("     R ~ 1/r diverges. Docstring line 17 claims 'de Sitter core ->")
print("     no curvature singularity' -- FALSE for default scale (BR gamma=0")
print("     case: singularity only 'much milder', cf. notes sec.1).")

print("=" * 70)
print("(iii) horizon structure; closed-form M_cr = 1/g* = %.6f" % (1 / GSTAR))
for M in [1.0, 1.2, 1.4, 1 / GSTAR, 1.5, 2.0, 5.0]:
    bh.M = M
    hs = bh.horizons()
    disc = (GSTAR * M) ** 2 - 1
    if disc > 0:
        exact = [GSTAR * M - np.sqrt(disc), GSTAR * M + np.sqrt(disc)]
    elif disc == 0:
        exact = [GSTAR * M]
    else:
        exact = []
    print(f"  M={M:8.5f}  n_found={len(hs)}  found={[f'{h:.5f}' for h in hs]} "
          f" exact={[f'{e:.5f}' for e in exact]}")

bh.M = 2.0
Mcr = bh.critical_mass()
print(f"  critical_mass() = {Mcr}   exact = {1/GSTAR:.6f}  "
      f"rel.err = {abs(Mcr - 1/GSTAR)*GSTAR:.2e}")

print("=" * 70)
print("(iv) bracketing soundness near extremality (default n_samples=10000)")
for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    M = (1 / GSTAR) * (1 + delta)
    bh.M = M
    hs = bh.horizons()
    sep = 2 * GSTAR * M * np.sqrt((GSTAR * M) ** 2 - 1) / 1.0
    print(f"  M=(1+{delta:.0e})M_cr: found {len(hs)} horizons "
          f"(expect 2; r+-r- = {sep:.4e})")
print("  large-M inner horizon vs default r_min=1e-3:")
for M in [100.0, 500.0, 1000.0]:
    bh.M = M
    hs = bh.horizons()
    r_minus = GSTAR * M - np.sqrt((GSTAR * M) ** 2 - 1)
    print(f"  M={M:7.1f}: found {len(hs)} (r_- exact={r_minus:.2e}, "
          f"r_min=1e-3 {'MISSES inner' if r_minus < 1e-3 else 'ok'})")
