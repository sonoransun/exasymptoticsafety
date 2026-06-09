"""HV-11: Gaussian FP of the EH system + d!=4 hardcoding hazard."""
import sympy
from sympy import Matrix, Symbol, simplify, symbols

from asymsafety.beta.einstein_hilbert import build_eh_beta_system

g = Symbol("g", positive=True)
lam = Symbol("lambda", real=True)

sys4 = build_eh_beta_system(d=4)
bg, bl = sys4.beta("g").expression, sys4.beta("lambda").expression

J = Matrix([[sympy.diff(b, v) for v in (g, lam)] for b in (bg, bl)])
J0 = simplify(J.subs({g: 0, lam: 0}))
print("Jacobian at (0,0):")
sympy.pprint(J0)
print("eigenvalues:", J0.eigenvals())
print("critical exponents theta = -eigs:", {-k: v for k, v in J0.eigenvals().items()})

# beta_lambda(0, lam) == -2 lam identically?
print("\nbeta_lambda(g=0) simplified:", simplify(bl.subs(g, 0)))
print("identically -2*lambda?:", simplify(bl.subs(g, 0) + 2 * lam) == 0)

# eta_N(0, lam) == 0?
eta = simplify(bg / g - 2)
print("eta_N(g=0, lam) =", simplify(eta.subs(g, 0)))

# off-diagonal GFP slope sign (structural, rescaling-invariant in sign)
print("\nd beta_lam/d g at GFP:", J0[1, 0], "=", float(J0[1, 0]),
      " (literature +1/(2*pi) ~ +0.159; toolkit sign is NEGATIVE)")

# ---- d != 4 hazard ----
sys3 = build_eh_beta_system(d=3)
bg3 = sys3.beta("g").expression
bl3 = sys3.beta("lambda").expression
print("\nd=3 system identical to d=4?:",
      simplify(bg3 - bg) == 0 and simplify(bl3 - bl) == 0)
print("d=3 beta_g canonical term: coefficient of g at eta=0 ->",
      sympy.limit(bg3 / g, g, 0), " (should be d-2 = 1 in d=3)")

# contrast: beta/matter.py uses (d - 2 + eta_N)
from asymsafety.actions.matter import MatterContent
from asymsafety.beta.matter import build_eh_matter_beta_system
m3 = build_eh_matter_beta_system(MatterContent(), d=3)
bg_m3 = m3.beta("g").expression
print("matter builder d=3 canonical coefficient:",
      sympy.limit(bg_m3 / g, g, 0), " (correctly d-2 = 1)")
