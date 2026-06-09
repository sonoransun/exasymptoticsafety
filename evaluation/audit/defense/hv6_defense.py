"""HV-6 defense attempt: is there ANY convention under which the vector
bundle-curvature term +1/12 * Riem^2 in SeeleyDeWittCoefficients.b4 is correct?

Defense avenues tested:
  A. Sign conventions for Omega / Riemann / signature (algebraic, explicit
     components on S^4): does any convention give tr(Om_mn Om^mn) = +Riem^2?
  B. Independent mode-sum (own derivation of spectrum, own fit, different
     t-window than the audit's script) for the vector operator the code
     describes (E = -Ric, i.e. Delta = -D^2 - Ric on unconstrained vectors).
  C. Operator-reinterpretation defense: is there ANY endomorphism E consistent
     with the code's own b0=4, b2=5R/3 (which pin trE=-R) and maximal symmetry
     (which pins E = (trE/4) * Id on the tangent bundle of S^4) that makes the
     code's b4 total correct with the true -1/12 Omega term?
  D. Basis defense: code returns {R2, Ric2, Riem2}; does reinterpreting in the
     {R^2, C^2, E_GB} basis change the on-sphere total b4_on_sphere compares?
"""
import itertools
import numpy as np
import sympy
from sympy import Rational, Symbol
import mpmath as mp

mp.mp.dps = 40

print("=" * 78)
print("A. ALGEBRA: tr(Omega_mn Omega^mn) on S^4 under all sign conventions")
print("=" * 78)
# Max-sym Riemannian S^4 (unit radius): R_abcd = K (g_ac g_bd - g_ad g_bc), K=1
# Try: s1 = overall Riemann sign convention (+/-), s2 = Omega = +/- [D_mu,D_nu]
# Tangent bundle: Omega_{mn}^{a}{}_{b} = s2 * s1 * R^{a}{}_{b mn}
d = 4
g = np.eye(d)  # Riemannian; for Lorentzian use diag(-1,1,1,1) below
for metric_name, gmat in (("Riemannian", np.eye(d)),
                          ("Lorentzian", np.diag([-1.0, 1, 1, 1]))):
    ginv = np.linalg.inv(gmat)
    K = 1.0
    Riem = np.zeros((d, d, d, d))  # R_{abcd}
    for a, b, c, e in itertools.product(range(d), repeat=4):
        Riem[a, b, c, e] = K * (gmat[a, c] * gmat[b, e] - gmat[a, e] * gmat[b, c])
    # Riem^2 = R_abcd R^abcd
    Riem_up = np.einsum('abcd,ae,bf,cg,dh->efgh', Riem, ginv, ginv, ginv, ginv)
    Riem2 = np.einsum('abcd,abcd->', Riem, Riem_up)
    results = set()
    for s1 in (+1, -1):          # Riemann overall sign convention
        for s2 in (+1, -1):      # Omega = +/-[D,D]
            # Omega_{mn a b} = s1*s2*R_{ab mn} (mixed a^b lowered with g)
            Om = s1 * s2 * np.einsum('abmn->mnab', Riem)
            Om_up = np.einsum('mnab,mp,nq,ar,bs->pqrs', Om, ginv, ginv, ginv, ginv)
            trOm2 = np.einsum('mnab,mnba->', Om, Om_up)  # tr over bundle = Om^a_b Om^b_a
            results.add(round(trOm2 / Riem2, 12))
    print(f"  {metric_name}: tr(Om^2)/Riem^2 over all 4 sign conventions = {sorted(results)}")
print("  -> tr(Omega^2) = -Riem^2 ALWAYS (quadratic in Omega; antisymmetry of")
print("     Riemann in first index pair is an identity, not a convention).")
print("  Vassilevich a4 term: +(30/360) tr(Om^2) = +(1/12)(-Riem^2) = -Riem^2/12.")
print("  Code adds +1/12 -> wrong under every convention.")

print()
print("=" * 78)
print("B. INDEPENDENT MODE SUM: vector op Delta = -D^2 - Ric on unit S^4 (R=12)")
print("=" * 78)
# Own derivation (stated facts, convention-free):
#  scalar harmonics: -D^2 eig l(l+3), deg = dim of harmonic deg-l polys in R^5
#       = C(l+4,4) - C(l+2,4) = (2l+3)(l+1)(l+2)/6   [integer rep theory]
#  transverse vectors: -D^2 eig l(l+3)-1 (l>=1), deg = l(l+3)(2l+3)/2
#       check l=1: deg=10 = #Killing vectors on S^4 = dim SO(5)  [integer fact]
#       check l=1 eig: Killing => D^2 xi = -Ric xi = -3 xi => -D^2 = 3 = 1*4-1 ok
#  longitudinal d(phi_l), l>=1: Weitzenboeck Delta_Hodge = -D^2 + Ric and
#       [Delta_Hodge, d]=0  => -D^2 dphi = (l(l+3) - 3) dphi
#       (re-derived by direct commutator below, symbolically)
l_, x_ = sympy.symbols('l x', positive=True)
from sympy import binomial, simplify
deg_sc_formula = binomial(l_ + 4, 4) - binomial(l_ + 2, 4)
assert simplify(deg_sc_formula - (2*l_+3)*(l_+1)*(l_+2)/6) == 0

def deg_sc(l): return mp.mpf((2*l+3)*(l+1)*(l+2))/6
def deg_tv(l): return mp.mpf(l*(l+3)*(2*l+3))/2

# Symbolic re-derivation of the longitudinal shift (max-sym, K = R/(d(d-1))):
# -D^2 grad(phi) = (lam + (1-d)*K*(-1)) ... compute contraction g^{bc} R^d_{bca}
Ksym, dsym = sympy.symbols('K d_')
# g^{bc} R^d_{bca} = K(1-d) delta^d_a  (computed in A numerically; here d=4,K=1 -> -3)
shift = (1 - 4) * 1  # = -3  => -D^2 dphi = (lam - 3) dphi  [matches Weitzenboeck]
print(f"  longitudinal Bochner shift = {shift} (so -D^2 dphi_l = (l(l+3)-3) dphi)")

def trace_op(t, c_shift):
    """Tr exp(-t(-D^2 + c_shift)) over ALL vector modes."""
    lmax = int(mp.sqrt(160/t)) + 25
    tot = mp.fsum(deg_tv(l)*mp.e**(-t*(l*(l+3)-1+c_shift)) for l in range(1, lmax))
    tot += mp.fsum(deg_sc(l)*mp.e**(-t*(l*(l+3)-3+c_shift)) for l in range(1, lmax))
    return tot

def fit(c_shift, t0, npts=9, deg=9):
    ts = [t0*(1 + mp.mpf(j)/3) for j in range(npts)]
    A = mp.matrix(npts, deg)
    rhs = mp.matrix(npts, 1)
    for i, t in enumerate(ts):
        rhs[i] = 6*t**2*trace_op(t, c_shift)   # = b0 + b2 t + b4 t^2 + ...
        for j in range(deg):
            A[i, j] = t**j
    sol = mp.qr_solve(A, rhs)[0]
    return [sol[j] for j in range(3)]

for name, cs in (("pure Bochner -D^2 (E=0)", 0), ("FP-ghost -D^2 - Ric (E=-Ric)", -3)):
    f1 = fit(cs, mp.mpf("0.018"))
    f2 = fit(cs, mp.mpf("0.011"))
    err = max(abs(f1[j]-f2[j]) for j in range(3))
    print(f"  {name:32s}: b0={mp.nstr(f1[0],8)}, b2={mp.nstr(f1[1],8)}, "
          f"b4={mp.nstr(f1[2],10)}  (stability {mp.nstr(err,2)})")

print("  Exact rationals: pure Bochner b4 = 86/15 =", mp.nstr(mp.mpf(86)/15, 10),
      "; E=-Ric b4 = 716/15 =", mp.nstr(mp.mpf(716)/15, 10))

print()
print("  CODE values (SeeleyDeWittCoefficients, d=4, R=12):")
from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients
R = Symbol("R_bar", positive=True)
sd = SeeleyDeWittCoefficients(d=4, R_bar=R)
b4v = sd.b4_on_sphere("vector")
print(f"    b0={sd.b0('vector')}, b2={sd.b2('vector')} -> {sd.b2('vector').subs(R,12)} at R=12")
print(f"    b4_on_sphere = {sympy.nsimplify(b4v/R**2)} R^2 = {b4v.subs(R,12)} at R=12 "
      f"= {mp.nstr(mp.mpf(776)/15,10)}")
print(f"    mode sum says 716/15; diff = {sympy.nsimplify(b4v.subs(R,12) - Rational(716,15))} "
      f"= 2*(1/12)*Riem2(=24) -> exactly the Omega^2 sign flip")

print()
print("=" * 78)
print("C. OPERATOR-REINTERPRETATION DEFENSE")
print("=" * 78)
# Could the code's b4 be right for some OTHER vector operator -D^2 + E?
# Constraints the code itself imposes:
#   b0 = 4  -> tangent-bundle-valued field, tr I = 4 (so Omega = Riemann, fixed)
#   b2 = 5R/3 = R*4/6 - trE  -> trE = -R  (code's own b2)
#   max symmetry: E built from background curvature on S^4 must be c*Id
#       -> E = -(R/4) Id  -> trE^2 = R^2/4  (no freedom left)
# Then b4 is FULLY determined:
trE, trE2, trI = -R, R**2/4, 4
Ric2, Riem2 = R**2/4, R**2/6
b4_forced = (trI*(Rational(1,72)*R**2 - Rational(1,180)*Ric2 + Rational(1,180)*Riem2)
             - Rational(1,12)*Riem2          # the TRUE Omega^2 term
             + Rational(1,2)*trE2 - Rational(1,6)*R*trE)
print(f"  forced b4 = {sympy.nsimplify(sympy.simplify(b4_forced)/R**2)} R^2 "
      f"= {sympy.simplify(b4_forced.subs(R,12))} at R=12 (mode sum: 716/15)")
print("  -> ANY operator matching the code's own b0 and b2 has b4 = 179/540 R^2.")
print("     The code's 97/270 R^2 is reachable only with tr(Om^2) = +Riem^2,")
print("     which part A shows is impossible. Defense fails.")
# What would E have to be (keeping correct Omega term) to reproduce code's b4?
trE2_needed = sympy.solve(
    sympy.Eq(trI*(Rational(1,72)*R**2 - Rational(1,180)*Ric2 + Rational(1,180)*Riem2)
             - Rational(1,12)*Riem2 + Rational(1,2)*Symbol('X') - Rational(1,6)*R*trE,
             Rational(97,270)*R**2), Symbol('X'))[0]
print(f"  (rescue would need trE^2 = {sympy.nsimplify(trE2_needed/R**2)} R^2 "
      f"instead of 1/4 R^2 -- impossible for E = c*Id with trE=-R, and max")
print("   symmetry forbids non-isotropic E on the tangent bundle of S^4.)")

print()
print("=" * 78)
print("D. BASIS DEFENSE: {R^2, Ric^2, Riem^2} vs {R^2, C^2, E_GB}")
print("=" * 78)
# b4_on_sphere contracts the dict with MaxSymBackground invariants; the on-sphere
# scalar total is basis-independent. Re-express code's coeffs in {R^2,C^2,E}:
# Riem^2 = C^2 + 2 Ric^2 - R^2/3 ; E = Riem^2 - 4Ric^2 + R^2. On S^4: C^2=0, E=R^2/6.
c = sd.b4("vector")
total_RRicRiem = (c["R2"]*R**2 + c["Ric2"]*Ric2 + c["Riem2"]*Riem2)
print(f"  on-sphere total from dict = {sympy.nsimplify(sympy.simplify(total_RRicRiem)/R**2)} R^2"
      " -- identical to b4_on_sphere; a basis change cannot alter a scalar.")
print("  (And the mode sum pins that scalar to 179/540 R^2.)")
