"""HV-15b: RG-improved FLRW Bianchi-consistency audit.

rg_improved_flrw.integrate() solves (line 143):
    drho/dt = -3 H (1+w) rho        <- STANDARD conservation
together with Friedmann-I with running G(k(t)), Lambda(k(t)).

Bianchi identity of the improved Einstein eqs (hep-th/0106133) requires
    rhodot + 3H(rho+p) = -(Gdot rho + Lamdot/(8pi))/G,
and BR's choice (standard conservation imposed separately) requires the
constraint  Lamdot + 8 pi rho Gdot = 0  (fixes xi). Neither the modified
continuity equation nor the constraint appears anywhere in the module.
This script integrates a radiation universe and monitors the residual.
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety")
from tests.conftest import make_as_trajectory  # noqa: E402

from asymsafety.cosmology.rg_improved_flrw import RGImprovedFLRW  # noqa: E402
from asymsafety.cosmology.scale_identification import (  # noqa: E402
    HubbleScale, ScaleIdentification,
)

traj = make_as_trajectory(g_star=0.7, lam_star=0.2)

for scale_name, scale in [
    ("HubbleScale(xi=1) [default]", None),
    ("k = 1/t (base-class k_of_t)", type(
        "InvTime", (ScaleIdentification,),
        {"k_of_r": lambda self, r: 1.0 / np.asarray(r)})()),
]:
    cosmo = RGImprovedFLRW(trajectory=traj, scale=scale,
                           equation_of_state=1.0 / 3.0)
    out = cosmo.integrate(a0=1e-4, t_span=(1e-2, 10.0), rho0=1.0,
                          n_steps=2000)
    t, a, rho, H, G, Lam = (out[k] for k in
                            ("t", "a", "rho", "H", "G", "Lambda"))
    w = 1.0 / 3.0
    rhodot = np.gradient(rho, t)
    Gdot = np.gradient(G, t)
    Lamdot = np.gradient(Lam, t)

    cons = rhodot + 3 * H * (1 + w) * rho            # what the code solves
    bianchi = cons + (Gdot * rho + Lamdot / (8 * np.pi)) / G
    constraint = Lamdot + 8 * np.pi * rho * Gdot      # BR's condition
    scale_term = 3 * H * (1 + w) * rho                # magnitude reference

    sl = slice(10, -10)  # avoid gradient end effects
    def rel(x):
        return np.max(np.abs(x[sl]) / np.maximum(np.abs(scale_term[sl]), 1e-300))

    print("=" * 72)
    print(f"scale = {scale_name}")
    print(f"  G range: [{G.min():.4g}, {G.max():.4g}]  "
          f"Lambda range: [{Lam.min():.4g}, {Lam.max():.4g}]")
    print(f"  |rhodot + 3H(rho+p)| / |3H(rho+p)|        max = {rel(cons):.3e}"
          "   (standard conservation: what integrate() enforces)")
    print(f"  |Bianchi residual|   / |3H(rho+p)|        max = {rel(bianchi):.3e}"
          "   (should be ~0 for a consistent improvement)")
    print(f"  |Lamdot + 8 pi rho Gdot| / |3H(rho+p)|    max = {rel(constraint):.3e}"
          "   (BR integrability constraint)")
    i = len(t) // 2
    print(f"  sample @t={t[i]:.2f}: rhodot+3H(rho+p)={cons[i]:.3e}, "
          f"(Gdot rho + Lamdot/8pi)/G={((Gdot*rho+Lamdot/(8*np.pi))/G)[i]:.3e}")

print("=" * 72)
print("Documentation grep (module source):")
import inspect  # noqa: E402
import asymsafety.cosmology.rg_improved_flrw as mod  # noqa: E402
src = inspect.getsource(mod)
for kw in ("Bianchi", "conserv", "consistency", "constraint", "energy"):
    print(f"  '{kw}' occurs: {kw.lower() in src.lower()}")
