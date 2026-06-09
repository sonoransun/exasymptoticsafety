"""HV-12 follow-up: does build_eh_matter_beta_system (zero matter) have ANY
non-Gaussian fixed point? Brute-force grid + fsolve."""
import numpy as np
from scipy.optimize import fsolve

from asymsafety.actions.matter import MatterContent
from asymsafety.beta.matter import build_eh_matter_beta_system, scan_matter_content

sys_m = build_eh_matter_beta_system(MatterContent(), d=4)
f = sys_m.rhs_vector()

def F(u):
    return np.asarray(f(0.0, u), dtype=float)

found = {}
for g0 in np.linspace(0.05, 20, 41):
    for l0 in np.linspace(-0.45, 0.45, 19):
        try:
            sol, info, ier, _ = fsolve(F, [g0, l0], full_output=True,
                                       xtol=1e-12)
        except Exception:
            continue
        if ier != 1:
            continue
        gs, ls = sol
        if gs > 1e-6 and abs(ls) < 0.499 and np.linalg.norm(F(sol)) < 1e-9:
            key = (round(gs, 6), round(ls, 6))
            found[key] = found.get(key, 0) + 1

print("non-Gaussian roots found (g>0, |lam|<0.5):", sorted(found))
if not found:
    # widen lambda beyond pole region and allow g<0 to characterize
    for g0 in np.linspace(-5, 40, 46):
        for l0 in np.linspace(-3, 3, 25):
            try:
                sol, info, ier, _ = fsolve(F, [g0, l0], full_output=True,
                                           xtol=1e-12)
            except Exception:
                continue
            if ier != 1:
                continue
            gs, ls = sol
            if abs(gs) > 1e-6 and np.linalg.norm(F(sol)) < 1e-9:
                key = (round(gs, 4), round(ls, 4))
                found[key] = found.get(key, 0) + 1
    print("any nontrivial roots anywhere:", sorted(found)[:20])

print("\nmodule's own scan_matter_content(range(0,4)):")
for row in scan_matter_content(range(0, 4)):
    print(" ", row)
