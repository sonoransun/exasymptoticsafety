"""HV-5/HV-6: mode-sum heat traces on unit S^4 (R_bar=12) vs SeeleyDeWittCoefficients.

Normalization: Tr e^{-t Delta} = (4 pi t)^{-2} V [b0 + b2 t + b4 t^2 + ...], V = 8 pi^2/3
=> F(t) := 6 t^2 Tr e^{-t Delta} = b0 + b2 t + b4 t^2 + ...
Fit F(t) at small t with a degree-7 polynomial (mpmath, dps=50).
"""
import mpmath as mp
import sympy
from sympy import Symbol, Rational, nsimplify

mp.mp.dps = 50

# ground-truth degeneracies (unit S^4)
sc = lambda l: mp.mpf((2*l+3)*(l+1)*(l+2))/6
tv = lambda l: mp.mpf(l*(l+3)*(2*l+3))/2
tt = lambda l: mp.mpf(5*(l-1)*(l+4)*(2*l+3))/6
# code degeneracies
sc_code = lambda l: mp.mpf((l+1)**2*(l+2)**2)/4
tv_code = lambda l: mp.mpf((2*l+3)*(l+2)*(l+1))/2

def trace(terms, t):
    """terms: list of (degfun, eigfun, lmin)."""
    total = mp.mpf(0)
    lmax = int(mp.sqrt(140/t)) + 20
    for deg, eig, lmin in terms:
        total += mp.fsum(deg(l)*mp.e**(-t*eig(l)) for l in range(lmin, lmax))
    return total

def fit_b(terms, t0=mp.mpf("0.02")):
    ts = [t0*(1 + mp.mpf(j)/4) for j in range(8)]
    A = mp.matrix(8, 8)
    rhs = mp.matrix(8, 1)
    for i, t in enumerate(ts):
        F = 6*t**2*trace(terms, t)
        rhs[i] = F
        for j in range(8):
            A[i, j] = t**j
    sol = mp.lu_solve(A, rhs)
    return [sol[j] for j in range(3)]  # b0, b2, b4

def report(name, terms, expect, t0s=("0.02", "0.014")):
    fits = [fit_b(terms, mp.mpf(s)) for s in t0s]
    b = fits[0]
    fiterr = max(abs(fits[0][j]-fits[1][j]) for j in range(3))
    ok = all(abs(b[j]-expect[j]) <= max(1e-3*abs(expect[j]), 1e-3) for j in range(3))
    print(f"{name:42s} b0={mp.nstr(b[0],10):>12s} b2={mp.nstr(b[1],10):>12s} "
          f"b4={mp.nstr(b[2],10):>12s} | expect {expect} | fit-stability {mp.nstr(fiterr,2)} | rtol1e-3 pass={ok}")
    return b

print("=== Mode-sum fits, GROUND-TRUTH degeneracies (unit S^4, R=12) ===")
report("scalar -D^2", [(sc, lambda l: l*(l+3), 0)], (1, 2, mp.mpf(29)/15))
report("transverse vector -D^2 (l>=1)", [(tv, lambda l: l*(l+3)-1, 1)], (3, 3, mp.mpf("-0.7")))
report("vector bundle Hodge -D^2+Ric", [(tv, lambda l: l*(l+3)+2, 1), (sc, lambda l: l*(l+3), 1)],
       (4, -4, mp.mpf(-4)/15))
report("vector bundle FP-ghost -D^2-Ric", [(tv, lambda l: l*(l+3)-4, 1), (sc, lambda l: l*(l+3)-6, 1)],
       (4, 20, mp.mpf(716)/15))
report("TT -D^2 (l>=2)", [(tt, lambda l: l*(l+3)-2, 2)], (5, -10, mp.mpf(-1)/3))
report("TT Lichnerowicz -D^2+2R/3", [(tt, lambda l: l*(l+3)+6, 2)], (5, -50, mp.mpf(719)/3))

print("\n=== Mode-sum with CODE degeneracies (demonstrate discrepancy) ===")
print("scalar, code deg (l+1)^2(l+2)^2/4: F(t)=6t^2 Tr should -> b0=1; instead diverges ~ t^(-1/2):")
for s in ("0.08", "0.04", "0.02", "0.01"):
    t = mp.mpf(s)
    F = 6*t**2*trace([(sc_code, lambda l: l*(l+3), 0)], t)
    print(f"  t={s}: F(t) = {mp.nstr(F,8)}   F*sqrt(t) = {mp.nstr(F*mp.sqrt(t),8)}")
report("vector, code deg = 3 x scalar deg", [(tv_code, lambda l: l*(l+3)-1, 1)],
       (3, 9, mp.mpf(199)/15))  # analytic: 3 e^t (Tr_s -1) -> b0=3,b2=9,b4=3(29/15+2+1/2)=199/15? see note
print("   (exact transverse-vector values are b0=3, b2=3, b4=-0.7 -> code deg gives b2 3x too large)")

print("\n=== CODE SeeleyDeWittCoefficients (d=4, R_bar=12) ===")
from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients
R = Symbol("R_bar", positive=True)
sd = SeeleyDeWittCoefficients(d=4, R_bar=R)
for ft in ("scalar", "vector", "TT_tensor", "ghost_vector"):
    b0 = sd.b0(ft)
    b2 = sd.b2(ft)
    b4s = sd.b4_on_sphere(ft)
    b4d = sd.b4(ft)
    print(f"{ft:>13}: b0={b0}, b2={b2} (={b2.subs(R,12)} at R=12), "
          f"b4_sphere={sympy.nsimplify(b4s/R**2)}*R^2 (={b4s.subs(R,12)} at R=12)")
    print(f"               b4 basis coeffs: R2={b4d['R2']}, Ric2={b4d['Ric2']}, Riem2={b4d['Riem2']}")

print("\n=== HV-6 exact-rational diffs ===")
print("vector Riem2 coeff: code = 1/45+1/12 = 19/180 ; ground truth (Christensen-Duff) = -11/180")
print("  -> bundle term should be (1/12)tr(Omega^2) = -(1/12)Riem2 [tr(Om^2) = -Riem2], code adds +1/12")
print("vector b4 on sphere: code", sympy.nsimplify(sd.b4_on_sphere('vector')/R**2),
      "R^2 vs exact (E=-Ric Vassilevich) 179/540 R^2 ; diff =",
      sympy.nsimplify(sd.b4_on_sphere('vector')/R**2 - Rational(179, 540)), "(= +1/36 = the Omega^2 sign flip x R^2/6... 2*(1/12)*(R^2/6))")
print("TT b2: code", sd.b2('TT_tensor'), "vs exact -D^2: -(5/6)R ; vs exact Lichnerowicz: -(25/6)R")
print("TT b4 on sphere: code", sympy.nsimplify(sd.b4_on_sphere('TT_tensor')/R**2),
      "R^2 vs exact -D^2: -R^2/432 ; vs exact Lichnerowicz: 719 R^2/432")
print("TT bundle term: code Riem2 += (1/12)*2*(d-1) = +1/2 ; unconstrained Sym^2 truth: (1/12)*(-(d+2)Riem2) = -1/2")

print("\n=== Vassilevich master check for vector E=-Ric (self-contained) ===")
# 360 b4 = 60 R trE_V + 180 trE_V^2 + 30 trOm^2 + trI (5R^2 - 2Ric^2 + 2Riem^2); E_V=+Ric
# = 60R^2 + 180 Ric^2 - 30 Riem^2 + 4(5R^2-2Ric^2+2Riem^2) = 80R^2 + 172Ric^2 - 22Riem^2
Ric2, Riem2 = R**2/4, R**2/6  # on S^4
b4_vass = (80*R**2 + 172*Ric2 - 22*Riem2)/360
print("Vassilevich vector b4 on sphere:", sympy.nsimplify(sympy.simplify(b4_vass)/R**2), "R^2 =",
      sympy.simplify(b4_vass.subs(R, 12)), "at R=12  (mode sum above: 716/15 = 47.733...)")
