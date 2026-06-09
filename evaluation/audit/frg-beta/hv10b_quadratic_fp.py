"""HV-10 follow-up: does the quadratic (g,lam,alpha,beta) system admit any
fixed point? beta_alpha/beta_beta depend on lambda only, so all four betas
vanish only if beta_alpha(lam*)=beta_beta(lam*)=0 at the EH lam*."""
import sympy
from sympy import Rational, Symbol, nsolve, solve

from asymsafety.beta.quadratic import build_quadratic_beta_system

lam = Symbol("lambda", real=True)
s = build_quadratic_beta_system(4)
ba, bb = s.beta("alpha").expression, s.beta("beta").expression

la = solve(sympy.Eq(ba, 0), lam)
lb = solve(sympy.Eq(bb, 0), lam)
print("beta_alpha = 0 at lambda =", [sympy.nsimplify(x) for x in la],
      "=", [float(x) for x in la])
print("beta_beta  = 0 at lambda =", [sympy.nsimplify(x) for x in lb],
      "=", [float(x) for x in lb])
print("simultaneous zero possible?:", set(la) & set(lb))

lam_star = 0.14228896515894982  # EH-sector NGFP lambda
print("beta_alpha(lam*) =", float(ba.subs(lam, lam_star)))
print("beta_beta (lam*) =", float(bb.subs(lam, lam_star)))
print("=> the 4-coupling quadratic truncation has NO complete fixed point;"
      " codello_2009.QUADRATIC_FP (alpha*=0.006, beta*=0.002, n_relevant=4)"
      " cannot be a fixed point of this system.")
