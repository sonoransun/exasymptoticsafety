"""D6: RG-improved Schwarzschild black hole (Bonanno-Reuter pattern)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
from asymsafety.cosmology.visualization import plot_lapse_with_horizons

system = build_eh_beta_system(d=4)
# trajectory from near the NGFP (UV) into the IR
traj = FlowIntegrator(system).integrate(
    {"g": 0.69, "lambda": 0.142}, t_span=(0.0, -12.0))

probe = RGImprovedSchwarzschild(traj, M=1.0, k0=1.0)
G_N = float(probe.G(1e6))
print(f"IR Newton constant from trajectory: G_N ~ {G_N:.4g} "
      f"(plateau check G(1e5)={float(probe.G(1e5)):.4g}, G(1e6)={G_N:.4g})")
M_pl = G_N ** -0.5
print(f"=> Planck mass scale M_pl ~ {M_pl:.4g}; scanning M around it")

def horizon_count(M):
    bh = RGImprovedSchwarzschild(traj, M=M, k0=1.0)
    return len(bh.horizons(r_min=1e-6, r_max=1e9, n_samples=200000)), bh

for M in [0.05 * M_pl, 0.2 * M_pl, 0.5 * M_pl, M_pl, 2 * M_pl, 10 * M_pl]:
    n, bh = horizon_count(M)
    print(f"M={M:.4g} ({M/M_pl:.2f} M_pl): horizons={n}")

# bisect for critical mass between a no-horizon and two-horizon mass
lo, hi = 0.01 * M_pl, 10 * M_pl
n_lo, _ = horizon_count(lo); n_hi, _ = horizon_count(hi)
print(f"horizon count at M={lo}: {n_lo}, at M={hi}: {n_hi}")
if n_lo == 0 and n_hi >= 1:
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n, _ = horizon_count(mid)
        if n == 0:
            lo = mid
        else:
            hi = mid
    print(f"critical mass M_cr ~ {0.5*(lo+hi):.6g} (code units)")

n2, bh2 = horizon_count(2 * M_pl)
r_h = bh2.horizons(r_min=1e-6, r_max=1e9, n_samples=200000)
r_hi = (max(r_h) * 4) if r_h else 1.0
fig = plot_lapse_with_horizons(bh2, r_range=(max(min(r_h) / 20, 1e-9) if r_h else 1e-6, r_hi))
fig.savefig("evaluation/demos/d6_lapse_horizons.png", dpi=110)
print("figure saved to evaluation/demos/d6_lapse_horizons.png")
