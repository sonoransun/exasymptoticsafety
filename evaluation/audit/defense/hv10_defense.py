import sympy as sp
from sympy import Rational, pi, Symbol, nsimplify, solve

lam = Symbol("lambda", real=True)
Phi_w = 1/(1 - 2*lam)   # Litim Phi^1_0(-2lam), matches frg/threshold.py
Phi_0 = sp.Integer(1)

ba = Rational(1,16)/pi**2 * (Rational(133,10)*Phi_w + Rational(5,36) - Rational(196,15)*Phi_0)
bb = Rational(1,16)/pi**2 * (Rational(7,10)*Phi_w - Rational(196,45)*Phi_0)

ba0 = sp.simplify(16*pi**2*ba.subs(lam, 0))
bb0 = sp.simplify(16*pi**2*bb.subs(lam, 0))
print("16pi^2 beta_alpha(0) =", ba0, "=", float(ba0), " claimed 53/45 =", float(Rational(53,45)))
print("16pi^2 beta_beta(0)  =", bb0, "=", float(bb0), " claimed -196/45 =", float(Rational(-196,45)))

# Is there ANY lambda where both match their claimed universals?
sol_a = solve(sp.Eq(16*pi**2*ba, Rational(53,45)), lam)
sol_b = solve(sp.Eq(16*pi**2*bb, Rational(-196,45)), lam)
print("lambda solving beta_alpha = 53/45/(16pi^2):", sol_a)
print("lambda solving beta_beta  = -196/45/(16pi^2):", sol_b)

# Overall threshold-fn rescaling Phi -> c*Phi: lambda=0 limit becomes
# c*(133/10 - 196/15) + 5/36 for alpha, c*(7/10 - 196/45) for beta.
c = Symbol("c", positive=True)
ca = solve(sp.Eq(c*(Rational(133,10)-Rational(196,15)) + Rational(5,36), Rational(53,45)), c)
cb = solve(sp.Eq(c*(Rational(7,10)) - c*Rational(196,45), Rational(-196,45)), c)
print("rescale c reconciling alpha:", ca, " reconciling beta:", cb, "-> must be equal & =1 to be a normalization; they are not")

# Cross-check via the actual toolkit
from asymsafety.beta.quadratic import build_quadratic_beta_system
sys4 = build_quadratic_beta_system()
ea = sys4.beta("alpha").expression
eb = sys4.beta("beta").expression
print("toolkit 16pi^2 beta_alpha(lam=0) =", sp.nsimplify(sp.simplify(16*pi**2*ea.subs(Symbol('lambda',real=True),0))))
print("toolkit 16pi^2 beta_beta(lam=0)  =", sp.nsimplify(sp.simplify(16*pi**2*eb.subs(Symbol('lambda',real=True),0))))
