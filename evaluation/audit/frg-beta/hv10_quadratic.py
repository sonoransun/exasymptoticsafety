"""HV-10: quadratic-gravity one-loop universals, exact rational comparison.

Literature basis (Codello-Percacci hep-th/0607128; Fradkin-Tseytlin;
Avramidi-Barvinsky), action S = int [ f_R2 R^2 + f_C2 C^2 + f_E E ]:
  d_t f_C2 = +(133/20)/(16 pi^2)        (C^2; AF means f_C2 -> +inf)
  d_t f_R2 = +(5/36)/(16 pi^2) at f_R2=0 (R^2 sector, omega=0)
  d_t f_E  = +(196/45)/(16 pi^2)        (Gauss-Bonnet, exact universal)
Toolkit action (actions/quadratic.py line 101): alpha*R^2 + beta*C^2,
so alpha ~ f_R2, beta ~ f_C2.
"""
from fractions import Fraction

import sympy
from sympy import Rational, Symbol, nsimplify, pi, simplify

from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.validation.codello_2009 import ONE_LOOP_UNIVERSAL

sys4 = build_quadratic_beta_system(d=4)
g = Symbol("g", positive=True)
lam = Symbol("lambda", real=True)
alpha = Symbol("alpha", real=True)
beta_ = Symbol("beta", real=True)

ba = sys4.beta("alpha").expression
bb = sys4.beta("beta").expression
print("beta_alpha expr:", ba)
print("beta_beta  expr:", bb)
print("depends on alpha/beta/g?:",
      ba.has(alpha) or ba.has(beta_) or ba.has(g),
      bb.has(alpha) or bb.has(beta_) or bb.has(g))

# evaluate at lambda = 0 (one-loop universal limit), extract exact rational
ba0 = simplify(ba.subs(lam, 0) * 16 * pi**2)
bb0 = simplify(bb.subs(lam, 0) * 16 * pi**2)
print("\n16*pi^2 * beta_alpha(lam=0) =", ba0, "=", Rational(ba0))
print("16*pi^2 * beta_beta (lam=0) =", bb0, "=", Rational(bb0))

print("\ncode internal sum check: 133/10 + 5/36 - 196/15 =",
      Fraction(133, 10) + Fraction(5, 36) - Fraction(196, 15))
print("code internal sum check: 7/10 - 196/45 =",
      Fraction(7, 10) - Fraction(196, 45))

print("\nvalidation/codello_2009 ONE_LOOP_UNIVERSAL claims:")
print("  beta_alpha_1loop =", ONE_LOOP_UNIVERSAL["beta_alpha_1loop"],
      " (53/45 =", Fraction(53, 45), "=", 53 / 45, ")")
print("  beta_beta_1loop  =", ONE_LOOP_UNIVERSAL["beta_beta_1loop"],
      " (-196/45 =", -196 / 45, ")")

print("\n--- exact comparisons ---")
cands = {
    "code beta_alpha(0)": Rational(ba0),
    "code beta_beta(0)": Rational(bb0),
}
lits = {
    "lit f_R2 universal (+5/36)": Rational(5, 36),
    "lit f_C2 universal (+133/20)": Rational(133, 20),
    "lit f_E  universal (+196/45)": Rational(196, 45),
    "validation claim alpha (53/45)": Rational(53, 45),
    "validation claim beta (-196/45)": Rational(-196, 45),
    "CP beta_lambda coeff (-133/10)": Rational(-133, 10),
}
for cn, cv in cands.items():
    print(f"{cn} = {cv} = {float(cv):+.6f}")
    for ln, lv in lits.items():
        if cv == lv:
            print(f"   MATCHES {ln}")
    if not any(cv == lv for lv in lits.values()):
        print("   matches NO literature universal and NOT the validation dict")

# Misassignment detectors from the derivation notes:
print("\n133/10 is attached to beta_ALPHA (R^2) in the code -> per notes,"
      " 133/10 belongs ONLY to the C^2 normalization. MISASSIGNED.")
print("196/45 (the exact Gauss-Bonnet universal) appears in beta_beta"
      " with sign -, and 196/15 = 3*(196/45)... check:",
      Fraction(196, 15) == 3 * Fraction(196, 45))

# AF sign check for C^2 in the toolkit's own convention (beta = coefficient
# of C^2 in the action): AF (lambda_CP -> 0+) requires d_t beta > 0.
print("\nAF sign: toolkit beta_beta(0) =", float(Rational(bb0)) / (16 * 3.14159**2),
      "< 0, i.e. the C^2 COEFFICIENT decreases ->"
      " lambda_CP = 1/(2*beta) grows: NOT asymptotic freedom in the"
      " coefficient convention. Literature: d_t f_C2 = +133/20/(16pi^2) > 0.")
print("validation dict asserts beta_beta_1loop < 0 == AF; that is only true"
      " if beta were the INVERSE coupling (lambda_CP), contradicting the"
      " action definition alpha*R^2 + beta*C^2.")
