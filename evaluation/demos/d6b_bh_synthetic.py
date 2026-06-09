"""D6b: RG-improved BH with the synthetic AS trajectory pattern (conftest-style)."""
import matplotlib
matplotlib.use("Agg")
import sys, numpy as np
sys.path.insert(0, "tests")
from conftest import make_as_trajectory
from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
from asymsafety.cosmology.visualization import plot_lapse_with_horizons

traj = make_as_trajectory()
probe = RGImprovedSchwarzschild(traj, M=1.0, k0=1.0)
for r in [1e2, 1e4, 1e6]:
    print(f"G(r={r:.0e}) = {float(probe.G(r)):.6g}")
G_N = float(probe.G(1e4)); M_pl = G_N ** -0.5
print(f"G_N={G_N:.4g}, M_pl={M_pl:.4g}")

def hc(M):
    bh = RGImprovedSchwarzschild(traj, M=M, k0=1.0)
    return len(bh.horizons(r_min=1e-8, r_max=1e6, n_samples=400000)), bh

for f in [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]:
    n, _ = hc(f * M_pl)
    print(f"M = {f:5.1f} M_pl: horizons = {n}")

lo, hi = 0.05 * M_pl, 20 * M_pl
n_lo, _ = hc(lo); n_hi, _ = hc(hi)
if n_lo == 0 and n_hi == 2:
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        n, _ = hc(mid)
        lo, hi = (mid, hi) if n == 0 else (lo, mid)
    print(f"M_cr = {0.5*(lo+hi):.6g} = {0.5*(lo+hi)/M_pl:.4f} M_pl")
n2, bh2 = hc(3 * M_pl)
rh = bh2.horizons(r_min=1e-8, r_max=1e6, n_samples=400000)
print("horizons at 3 M_pl:", [f"{r:.4g}" for r in rh])
fig = plot_lapse_with_horizons(bh2, r_range=(max(min(rh)/20, 1e-9) if rh else 1e-6, (max(rh)*4) if rh else 1.0))
fig.savefig("evaluation/demos/d6b_lapse_horizons.png", dpi=110)
print("figure saved")
