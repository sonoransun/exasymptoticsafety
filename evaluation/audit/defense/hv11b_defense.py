"""HV-11b defense: exhaust legitimate schemes for d(beta_lambda)/dg at the GFP.

Tests:
 1. Toolkit slope (exact, symbolic) and whether eta_N terms could rescue it.
 2. Mode-count table across every consistent scheme variant:
    alpha=1 (all 10 graviton modes at w=-2lam),
    alpha->0 Landau (6 modes at w=-2lam, 4 gauge modes at w=0),
    exponential parametrization / physical gauge variants,
    Type I vs Type II cutoff (w-shifts vanish at lam=0 on flat projection).
 3. Toolkit's own NGFP + critical exponents vs its claimed benchmark
    (Reuter g*=0.707, lam*=0.193, theta=1.47+-3.04i).
"""
import numpy as np, sympy
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from scipy.optimize import fsolve

sys = build_eh_beta_system()
g, lam = sympy.Symbol("g", positive=True), sympy.Symbol("lambda", real=True)
bl = sys.beta("lambda").expression
bg = sys.beta("g").expression

# 1. exact GFP slope incl. all eta_N feedback
slope = sympy.limit(sympy.diff(bl, g).subs(lam, 0), g, 0, '+')
print("toolkit d(beta_lam)/dg|GFP exact =", sympy.simplify(slope), "=", float(slope))

# any other term at O(g) at lam=0?
series = sympy.series(bl.subs(lam, 0), g, 0, 2).removeO()
print("beta_lam(lam=0) O(g) series:", sympy.simplify(series))

# 2. mode-count at lam=0 (volume term ~ sum_i n_i * Phi^1_2(w_i), Phi^1_2 Litim = 1/(2(1+w)))
# any consistent scheme: graviton-sector components - 2*d ghost components
d = 4
print("\nflat-space net mode counts at lam=0 (sign of GFP slope):")
print("  alpha=1 single-metric:  10 - 8 =", d*(d+1)//2 - 2*d)
print("  alpha->0 Landau (TT5+conf1 at -2lam, 4 gauge at 0): 6 + 4 - 8 =", 6+4-8)
print("  York+Jacobians absorbed: 5+3+1+1 - 8 =", 5+3+1+1-8)
print("  toolkit: 6 - 8 =", 6-8, "  <-- only obtainable by DROPPING the 4 gauge modes")

# Landau-gauge bracket would be 3/(1-2lam) - 2 (not -4): slope sign still +
print("  Landau bracket at lam=0: 3-2 =", 3-2, "(positive)")
print("  alpha=1 bracket at lam=0: 5-4 =", 5-4, "(positive)")
print("  toolkit bracket at lam=0: 3-4 =", 3-4, "(negative)")

# 3. toolkit NGFP and exponents
f = sympy.lambdify((g, lam), [bg, bl], "numpy")
def F(u): return np.array(f(*u), dtype=float)
u0 = fsolve(F, [0.69, 0.14], xtol=1e-13)
J = np.zeros((2,2)); h=1e-7
for j in range(2):
    up, um = u0.copy(), u0.copy(); up[j]+=h; um[j]-=h
    J[:, j] = (F(up)-F(um))/(2*h)
th = -np.linalg.eigvals(J)
eta_at_fp = float(sympy.lambdify((g, lam), sympy.simplify((bg/g - 2)))(*u0))
print(f"\ntoolkit NGFP: g*={u0[0]:.6f} lam*={u0[1]:.6f} g*lam*={u0[0]*u0[1]:.4f}")
print("toolkit theta:", th)
print("eta_N at toolkit FP:", eta_at_fp, "(consistent scheme requires -2)")
print("claimed benchmark (validation/reuter_1998): g*=0.707 lam*=0.193 theta=1.47+-3.04i, g*lam*=0.136")
