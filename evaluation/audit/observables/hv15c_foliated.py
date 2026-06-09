"""HV-15c: foliated (ADM) beta system audit.

1. factor(beta_lambda_ADM) must contain (lambda_ADM - 1) exactly.
2. Locate fixed points (multi-start fsolve on (g, lambda) with
   lambda_ADM = 1, plus the full 3D system).
3. Compare against validation/manrique_2011.py claims (g*=0.96, l*=0.20)
   AND the published MRS PRL 106, 251302 Eq.(10): Euclidean g*=0.19,
   lambda*=0.31, theta=1.07+-3.31i; Lorentzian g*=0.21, lambda*=0.30.
4. Sign of the lambda_ADM-direction eigenvalue at the FP.
"""
import sys
import itertools
import numpy as np
import sympy
from scipy.optimize import fsolve

sys.path.insert(0, "/root/cdev/exasymptoticsafety")

from asymsafety.beta.foliated import (  # noqa: E402
    build_foliated_eh_beta_system, foliated_eh_benchmark,
)
from asymsafety.validation.manrique_2011 import (  # noqa: E402
    FOLIATED_EH_FP, LORENTZIAN_FP,
)

system = build_foliated_eh_beta_system(d=4)
lam_adm = sympy.Symbol("lambda_ADM", real=True)
b_adm = system.beta("lambda_ADM").expression

print("=" * 72)
print("(1) factor structure of beta_lambda_ADM:")
fact = sympy.factor(b_adm)
print("  factored:", fact)
quotient = sympy.simplify(b_adm / (lam_adm - 1))
print("  beta/(lambda_ADM-1) free of lambda_ADM:",
      lam_adm not in quotient.free_symbols)
print("  quotient h(g,lambda)*g =", sympy.simplify(quotient))

print("=" * 72)
print("(2) fixed-point hunt")
names = system.coupling_names
rhs = system.rhs_vector()

def F(y):
    return rhs(0.0, y)

roots = []
gg = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.96, 1.5, 2.5]
ll = [-1.5, -1.0, -0.7, -0.3, 0.0, 0.1, 0.2, 0.3, 0.45]
for g0, l0 in itertools.product(gg, ll):
    y0 = np.array([g0, l0, 1.0])
    try:
        sol, info, ier, msg = fsolve(F, y0, full_output=True)
    except Exception:
        continue
    if ier != 1 or np.max(np.abs(info["fvec"])) > 1e-10:
        continue
    if not np.all(np.isfinite(sol)):
        continue
    if any(np.allclose(sol, r, atol=1e-6) for r in roots):
        continue
    roots.append(sol)

print(f"  distinct roots found ({len(roots)}):")
J_func = system.jacobian_numerical
for r in sorted(roots, key=lambda x: x[0]):
    loc = dict(zip(names, [float(v) for v in r]))
    M = J_func(loc)
    mu = np.linalg.eigvals(M)
    theta = -mu
    print(f"  g*={r[0]: .6f} lambda*={r[1]: .6f} lambda_ADM*={r[2]: .6f}")
    print(f"    theta_i = {np.round(theta, 4)}")
    # lambda_ADM-direction eigenvalue: dbeta_adm/dlambda_ADM at FP
    d_adm = M[2, 2]
    print(f"    dbeta_ADM/dlambda_ADM = {d_adm:+.6f} -> theta_ADM = "
          f"{-d_adm:+.6f} ({'UV-attractive (relevant)' if -d_adm > 0 else 'UV-REPULSIVE (irrelevant)'})")

print("=" * 72)
print("(3) comparisons")
print("  toolkit benchmark (foliated_eh_benchmark / manrique_2011.py):",
      foliated_eh_benchmark(), FOLIATED_EH_FP)
print("  PUBLISHED MRS PRL 106 251302 Eq.(10):")
print("    Euclidean : g*=0.19, lambda*=0.31, theta=1.07+-3.31i")
print("    Lorentzian: g*=0.21, lambda*=0.30, theta=0.94+-3.10i")
print("  toolkit LORENTZIAN_FP claim (attributed Biemans 2017):", LORENTZIAN_FP)
print("  NOTE: MRS truncation has NO lambda_ADM coupling; lambda_ADM=1 is")
print("        imposed by their ansatz (verbatim: 'no running couplings that")
print("        could encode a breaking of Diff'), not derived at the FP.")

print("=" * 72)
print("(4) NGFP existence in the (g,lambda) sector with lambda_ADM=1:")
print("    beta_g = (2+eta)g = 0 requires eta = -2; eta = gA/(1-gB).")
g, lam = sympy.Symbol("g", positive=True), sympy.Symbol("lambda", real=True)
bg = system.beta("g").expression
eta = sympy.simplify(bg / g - 2)
# g*(lambda) solving eta = -2:
gsol = sympy.solve(sympy.Eq(eta, -2), g)
print("    g*(lambda) from eta=-2:", gsol)
if gsol:
    gfun = sympy.lambdify(lam, gsol[0], "numpy")
    for lv in [-2.0, -1.0, -0.75, -0.6, -0.51, 0.0, 0.1, 0.2, 0.3]:
        try:
            print(f"    lambda={lv:+.2f}: g*={float(gfun(lv)):+.5f}")
        except Exception as e:
            print(f"    lambda={lv:+.2f}: {e}")
