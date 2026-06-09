"""HV-15b defense probe 2: k = xi/t path with BR's constraint-chosen xi.

BR hep-th/0106133 Sec. III: fixed-point regime, k = xi/t, standard
conservation, constraint Lamdot + 8 pi rho Gdot = 0 fixes xi.
Analytically (radiation, flat): xi^2 = 3/(2 lam*), attractor
u = rho t^4 -> lam* xi^4/(8 pi g*), a ~ t, H = 1/t.
The code's try-branch uses scale.k_of_t(t) directly, so a subclass
returning xi/t realizes BR's identification EXACTLY (no H_est hack).
Test: does the Bianchi residual -> 0 along the code's solution?
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety")
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")

from asymsafety.analysis.flow import RGTrajectory  # noqa: E402
from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW  # noqa: E402
from asymsafety.cosmology.scale_identification import (  # noqa: E402
    ScaleIdentification,
)

G_STAR, LAM_STAR = 0.7, 0.2
XI_BR = np.sqrt(3.0 / (2.0 * LAM_STAR))

t_rg = np.linspace(-20.0, 20.0, 801)
fp_traj = RGTrajectory(
    t_values=t_rg,
    coupling_values={"g": np.full_like(t_rg, G_STAR),
                     "lambda": np.full_like(t_rg, LAM_STAR)},
    coupling_names=["g", "lambda"],
)


class TimeScale(ScaleIdentification):
    def __init__(self, xi):
        self.xi = xi

    def k_of_r(self, r):
        raise NotImplementedError

    def k_of_t(self, t, hubble=None):
        return self.xi / np.asarray(t)


w = 1.0 / 3.0
# BR attractor initial data: rho0 = lam xi^4 / (8 pi g) / t0^4, a ~ t
t0, t1 = 0.1, 50.0
rho_br = LAM_STAR * XI_BR**4 / (8 * np.pi * G_STAR) / t0**4

for label, xi, rho0 in [
    ("xi=xi_BR, BR-attractor rho0 (exact BR FP solution)", XI_BR, rho_br),
    ("xi=xi_BR, generic rho0=1 (attractor test)", XI_BR, 1.0),
    ("xi=1 (auditor's case), generic rho0=1", 1.0, 1.0),
]:
    cosmo = RGImprovedFLRW(trajectory=fp_traj, scale=TimeScale(xi),
                           equation_of_state=w)
    out = cosmo.integrate(a0=t0, t_span=(t0, t1), rho0=rho0, n_steps=6000)
    t, a, rho, H, G, Lam = (out[k] for k in
                            ("t", "a", "rho", "H", "G", "Lambda"))
    rhodot = np.gradient(rho, t)
    Gdot = np.gradient(G, t)
    Lamdot = np.gradient(Lam, t)
    cons = rhodot + 3 * H * (1 + w) * rho
    bianchi = cons + (Gdot * rho + Lamdot / (8 * np.pi)) / G
    constraint = Lamdot + 8 * np.pi * rho * Gdot
    ref = 3 * H * (1 + w) * rho
    sl = slice(20, -20)
    relb = np.abs(bianchi[sl]) / np.maximum(np.abs(ref[sl]), 1e-300)
    relc = np.abs(constraint[sl]) / np.maximum(np.abs(ref[sl]), 1e-300)
    p = np.polyfit(np.log(t[sl]), np.log(a[sl]), 1)
    print(f"{label}")
    print(f"  Bianchi residual: max={relb.max():.3e}  "
          f"final-decade max={relb[-600:].max():.3e}")
    print(f"  BR constraint:    max={relc.max():.3e}  "
          f"final-decade max={relc[-600:].max():.3e}")
    print(f"  Ht at end = {(H[-30]*t[-30]):.6f} (BR: 1);  "
          f"a~t^alpha, alpha={p[0]:.4f} (BR: 1)")
    u = rho * t**4
    print(f"  u=rho t^4: start {u[20]:.4g} -> end {u[-20]:.4g}; "
          f"BR value {LAM_STAR*XI_BR**4/(8*np.pi*G_STAR):.4g}")
    print()
