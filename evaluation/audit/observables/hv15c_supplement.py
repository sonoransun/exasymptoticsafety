"""HV-15c supplement: exhaustive NGFP search for the foliated system.

On the eta=-2 curve g*(lambda) (required by beta_g=0 with g!=0),
scan beta_lambda for roots; also evaluate the lambda_ADM-direction
eigenvalue at the *claimed* benchmark (0.96, 0.20, 1.0).
"""
import sys
import numpy as np
import sympy

sys.path.insert(0, "/root/cdev/exasymptoticsafety")
from asymsafety.beta.foliated import build_foliated_eh_beta_system  # noqa: E402

system = build_foliated_eh_beta_system(d=4)
g, lam = sympy.Symbol("g", positive=True), sympy.Symbol("lambda", real=True)
lam_adm = sympy.Symbol("lambda_ADM", real=True)

bg = system.beta("g").expression
bl = system.beta("lambda").expression
eta = sympy.simplify(bg / g - 2)
gstar = sympy.solve(sympy.Eq(eta, -2), g)[0]
bl_on_curve = sympy.simplify(bl.subs(g, gstar))
f = sympy.lambdify(lam, bl_on_curve, "numpy")
gf = sympy.lambdify(lam, gstar, "numpy")

print("beta_lambda restricted to the eta=-2 curve (g = g*(lambda) > 0 needs lambda < -1/2):")
lams = np.linspace(-6.0, -0.502, 4000)
vals = np.array([f(x) for x in lams])
sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
print("  sign changes on lambda in (-6, -0.502):", len(sign_changes))
for idx in sign_changes:
    from scipy.optimize import brentq
    r = brentq(f, lams[idx], lams[idx + 1])
    print(f"  root: lambda*={r:.6f}, g*={gf(r):.6f}")
print("  sample values:", [(round(x, 2), round(float(f(x)), 4))
                           for x in (-5.0, -3.0, -2.0, -1.0, -0.6, -0.51)])

print()
print("lambda_ADM eigenvalue at the CLAIMED benchmark (0.96, 0.20, 1.0):")
M = system.jacobian_numerical({"g": 0.96, "lambda": 0.20, "lambda_ADM": 1.0})
print("  dbeta_ADM/dlambda_ADM =", M[2, 2], "-> theta_ADM =", -M[2, 2])
print("  (positive M[2,2] => lambda_ADM=1 UV-REPULSIVE; perturbations grow toward UV)")
print("  residuals at claimed benchmark: beta =",
      system.evaluate({"g": 0.96, "lambda": 0.20, "lambda_ADM": 1.0}))
