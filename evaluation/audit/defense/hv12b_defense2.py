import numpy as np, itertools, sympy
from scipy.optimize import fsolve
from sympy import Rational, pi, Symbol
from asymsafety.actions.matter import MatterContent, ScalarFieldAction, matter_eta_N_correction
from asymsafety.beta.matter import build_eh_matter_beta_system, build_gravity_matter_fp_system

# (1) grid-hunt NGFP of build_eh_matter_beta_system
print("=== (1) build_eh_matter_beta_system: grid hunt for any NGFP ===")
for ns in [0, 4]:
    sys_ = build_eh_matter_beta_system(MatterContent(n_scalars=ns))
    f = sys_.rhs_vector()
    fps = set()
    for g0 in np.linspace(0.05, 3.0, 12):
        for l0 in np.linspace(-0.45, 0.45, 12):
            sol, info, ier, msg = fsolve(lambda y: f(0.0, y), [g0, l0], full_output=True)
            if ier == 1 and sol[0] > 1e-3 and abs(sol[1]) < 0.5 and max(abs(np.array(f(0.0, sol)))) < 1e-9:
                fps.add((round(sol[0], 6), round(sol[1], 6)))
    print(f"  N_s={ns}: NGFPs found: {sorted(fps) if fps else 'NONE in g in (0,3], |lambda|<0.5'}")

# (2) does FP ever get destroyed with scalars in build_gravity_matter_fp_system?
print("\n=== (2) build_gravity_matter_fp_system: scalar scan to N_s=60 ===")
x0 = np.array([0.65, 0.14]); alive = []
for n in range(0, 61):
    f = build_gravity_matter_fp_system(MatterContent(n_scalars=n)).rhs_vector()
    sol, info, ier, msg = fsolve(lambda y: f(0.0, y), x0, full_output=True)
    if ier == 1 and sol[0] > 1e-4:
        alive.append((n, sol[0])); x0 = sol
    else:
        print(f"  FP lost at N_s={n}"); break
else:
    print(f"  FP survives to N_s=60: g*({alive[0][0]})={alive[0][1]:.4f} ... g*({alive[-1][0]})={alive[-1][1]:.4f}")
    print("  (toolkit's own validation/korver_2024: NGFP must CEASE TO EXIST beyond N_s~12 foliated / ~22 covariant)")

# (3) internal consistency: ScalarFieldAction.gravity_beta_contribution vs matter_eta_N_correction
print("\n=== (3) ScalarFieldAction.gravity_beta_contribution (minimal, xi=0) ===")
lam = Symbol("lambda")
contrib = ScalarFieldAction().gravity_beta_contribution(lam)
Rterm = sympy.simplify(contrib["R"])
print(f"  R-term contribution = {Rterm} = {float(Rterm):+.6f} per k^2  (POSITIVE)")
print(f"  -> implied eta_N contribution = 16*pi*Rterm*g = {sympy.nsimplify(sympy.simplify(16*pi*Rterm))} * g = {float(16*np.pi*float(Rterm)):+.6f} g")
A_m, _ = matter_eta_N_correction(MatterContent(n_scalars=1))
print(f"  matter_eta_N_correction A_scalar      = {sympy.nsimplify(A_m)} = {float(A_m):+.6f}  (NEGATIVE)")
print(f"  DEP per-scalar                        = 1/(6*pi) = {1/(6*np.pi):+.6f}")
