"""Defense probe HV-15a-b: is there ANY implemented ScaleIdentification /
configuration under which 'de Sitter core -> no curvature singularity' holds?

de Sitter core test: f(r) = 1 - c r^2 + o(r^2)  <=>  (1-f)/r^2 -> const, f'(0)=0.
Curvature regularity test: R = 2(1-f)/r^2 - 4f'/r - f'' bounded as r->0.
"""
import sys
import numpy as np
sys.path.insert(0, "/root/cdev/exasymptoticsafety")
from tests.conftest import make_as_trajectory
from asymsafety.cosmology.rg_improved_bh import RGImprovedSchwarzschild
from asymsafety.cosmology.scale_identification import (
    InverseDistanceScale, GeodesicDistanceScale, ProperDistanceScale)

np.seterr(all="ignore")
traj = make_as_trajectory(g_star=0.7, lam_star=0.2)
M = 2.0

def ricci(bh, r, h=None):
    h = h or r * 1e-4
    f0 = bh.lapse(r)
    fp = (bh.lapse(r + h) - bh.lapse(r - h)) / (2 * h)
    fpp = (bh.lapse(r + h) - 2 * f0 + bh.lapse(r - h)) / h**2
    return 2 * (1 - f0) / r**2 - 4 * fp / r - fpp

def probe(name, scale, rs=(1e-1, 1e-2, 1e-3, 1e-4)):
    bh = RGImprovedSchwarzschild(trajectory=traj, scale=scale, M=M)
    print(f"--- {name}")
    for r in rs:
        f = bh.lapse(r)
        print(f"    r={r:.0e}  f={f:+.6e}  (1-f)/r={ (1-f)/r:+.4e}  "
              f"(1-f)/r^2={(1-f)/r**2:+.4e}  R={ricci(bh, r):+.4e}")

probe("InverseDistanceScale (DEFAULT, k=xi/r)", InverseDistanceScale(xi=1.0))
probe("GeodesicDistanceScale delta=0.1 ('BR softening', k bounded)",
      GeodesicDistanceScale(xi=1.0, delta=0.1))
probe("GeodesicDistanceScale delta=1.0", GeodesicDistanceScale(xi=1.0, delta=1.0))
probe("ProperDistanceScale r_h=0 delta=0 (== inverse)", ProperDistanceScale())
probe("ProperDistanceScale r_h=0.5 delta=0 (clamped core, k=inf inside)",
      ProperDistanceScale(xi=1.0, r_h=0.5), rs=(0.6, 0.4, 1e-2, 1e-4))
probe("ProperDistanceScale r_h=0.5 delta=0.05",
      ProperDistanceScale(xi=1.0, r_h=0.5, delta=0.05), rs=(0.6, 0.4, 1e-2, 1e-4))

# include_lambda with default scale: Lambda(r) r^2/3 = lam* xi^2/3 = const offset
bh = RGImprovedSchwarzschild(trajectory=traj, M=M, include_lambda=True)
print("--- InverseDistance + include_lambda=True")
for r in (1e-1, 1e-2, 1e-3):
    f = bh.lapse(r)
    print(f"    r={r:.0e}  f={f:+.6e}  R={ricci(bh, r):+.4e}")

# What WOULD work: BR proper distance d(r)=[r^3/(r+gamma G0 M)]^{1/2} (NOT implemented)
class BRProperDistance(InverseDistanceScale):
    gamma_G0M = 4.5 * 0.7 * M
    def k_of_r(self, r):
        r = np.asarray(r, dtype=float)
        return self.xi / np.sqrt(r**3 / (r + self.gamma_G0M))

probe("BR d(r)=[r^3/(r+gamma G0 M)]^{1/2}  (NOT in the codebase)",
      BRProperDistance(xi=1.0))
