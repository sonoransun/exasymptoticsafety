"""HV-15b defense probe.

Claim to test: line 143's drho/dt = -3H(1+w)rho is exactly BR's own
postulate (hep-th/0106133 imposes STANDARD conservation, plus the
integrability constraint Lamdot + 8 pi rho Gdot = 0 which fixes xi).
In the fixed-point regime with k = xi*H, G = g*/k^2, Lam = lam*k^2,
Friedmann-I gives 8 pi g* rho/(xi^2 H^4) = 3 - lam* xi^2, so the
constraint bracket is (2 lam* xi^2 - 3): it vanishes iff
xi = sqrt(3/(2 lam*)).  If the code with that xi gives a ~0 Bianchi
residual, the equations as coded ARE the BR system for that parameter
choice; the failure is only that xi is free/defaulted to 1 and the
constraint is undocumented.
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety")
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")

from asymsafety.analysis.flow import RGTrajectory  # noqa: E402
from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW  # noqa: E402
from asymsafety.cosmology.scale_identification import HubbleScale  # noqa: E402
from tests.conftest import make_as_trajectory  # noqa: E402

G_STAR, LAM_STAR = 0.7, 0.2
XI_BR = np.sqrt(3.0 / (2.0 * LAM_STAR))  # constraint-determined xi

# Pure fixed-point trajectory: g, lambda constant for all RG times
# (this is the regime BR actually solve in hep-th/0106133 Sec. III).
t_rg = np.linspace(-15.0, 15.0, 601)
fp_traj = RGTrajectory(
    t_values=t_rg,
    coupling_values={"g": np.full_like(t_rg, G_STAR),
                     "lambda": np.full_like(t_rg, LAM_STAR)},
    coupling_names=["g", "lambda"],
)
crossover_traj = make_as_trajectory(g_star=G_STAR, lam_star=LAM_STAR)

w = 1.0 / 3.0


def residuals(cosmo, label):
    out = cosmo.integrate(a0=1e-4, t_span=(1e-2, 10.0), rho0=1.0,
                          n_steps=4000)
    t, rho, H, G, Lam = (out[k] for k in ("t", "rho", "H", "G", "Lambda"))
    rhodot = np.gradient(rho, t)
    Gdot = np.gradient(G, t)
    Lamdot = np.gradient(Lam, t)
    cons = rhodot + 3 * H * (1 + w) * rho
    bianchi = cons + (Gdot * rho + Lamdot / (8 * np.pi)) / G
    constraint = Lamdot + 8 * np.pi * rho * Gdot
    ref = 3 * H * (1 + w) * rho
    sl = slice(20, -20)

    def rel(x):
        return np.max(np.abs(x[sl]) / np.maximum(np.abs(ref[sl]), 1e-300))

    print(f"{label}")
    print(f"  Bianchi residual / 3H(rho+p):   max = {rel(bianchi):.3e}")
    print(f"  BR constraint    / 3H(rho+p):   max = {rel(constraint):.3e}")
    # check a ~ t (BR fixed-point solution) by fitting log a vs log t
    p = np.polyfit(np.log(t[sl]), np.log(out["a"][sl]), 1)
    print(f"  power-law a ~ t^alpha fit: alpha = {p[0]:.4f} "
          f"(BR fixed-point radiation prediction: 1)")


print(f"xi_BR = sqrt(3/(2*lam)) = {XI_BR:.6f}\n")

residuals(RGImprovedFLRW(trajectory=fp_traj, scale=HubbleScale(xi=XI_BR),
                         equation_of_state=w),
          "[A] pure FP trajectory, xi = xi_BR (constraint-chosen)")
print()
residuals(RGImprovedFLRW(trajectory=fp_traj, scale=HubbleScale(xi=1.0),
                         equation_of_state=w),
          "[B] pure FP trajectory, xi = 1 (code default)")
print()
residuals(RGImprovedFLRW(trajectory=crossover_traj,
                         scale=HubbleScale(xi=XI_BR),
                         equation_of_state=w),
          "[C] crossover fixture trajectory, xi = xi_BR")
print()
residuals(RGImprovedFLRW(trajectory=crossover_traj,
                         scale=HubbleScale(xi=1.0),
                         equation_of_state=w),
          "[D] crossover fixture trajectory, xi = 1 (auditor's default case)")
