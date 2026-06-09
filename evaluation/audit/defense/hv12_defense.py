import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings("ignore")

from asymsafety.actions.matter import MatterContent
from asymsafety.beta.matter import build_eh_matter_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

sys_m = build_eh_matter_beta_system(MatterContent(), d=4)
f = sys_m.rhs_vector()
F = lambda u: np.asarray(f(0.0, u), dtype=float)

# 1. Exhaustive fine search in the physically published region
found = {}
for g0 in np.linspace(0.01, 50, 120):
    for l0 in np.linspace(-0.49, 0.49, 50):
        try:
            sol, info, ier, _ = fsolve(F, [g0, l0], full_output=True, xtol=1e-13)
        except Exception:
            continue
        if ier != 1: continue
        gs, ls = sol
        if gs > 1e-6 and -0.5 < ls < 0.5 and np.linalg.norm(F(sol)) < 1e-10:
            found[(round(gs,5), round(ls,5))] = True
print("roots with g>0, |lam|<0.5 (fine 120x50 grid):", sorted(found))

# 2. Analyze the exotic root at (16.28, -0.671)
sol, info, ier, msg = fsolve(F, [16.28, -0.671], full_output=True, xtol=1e-13)
print("\nexotic root converged:", ier==1, sol, "residual:", np.linalg.norm(F(sol)))
fp = FixedPointFinder(sys_m).find_fixed_point({"g": float(sol[0]), "lambda": float(sol[1])})
if fp is not None:
    sa = analyze_stability(sys_m, fp)
    print("location:", fp.location)
    print("critical exponents (theta):", fp.critical_exponents)
    print("g*lambda* =", fp.location["g"]*fp.location["lambda"])

# 3. eta_N sign structure in physical region: is eta>0 for all g>0, -0.5<lam<0.5?
import sympy
from sympy import Symbol, lambdify
g, lam = Symbol("g", positive=True), Symbol("lambda", real=True)
beta_g = sys_m.beta("g").expression
eta = sympy.simplify(beta_g/g - 2)
eta_f = lambdify((g, lam), eta, "numpy")
gv = np.linspace(0.001, 50, 300); lv = np.linspace(-0.499, 0.499, 300)
G, L = np.meshgrid(gv, lv)
E = eta_f(G, L)
print("\nmin eta_N over physical region:", np.nanmin(E))
print("eta reaches -2 anywhere in region?", np.nanmin(E) <= -2)
