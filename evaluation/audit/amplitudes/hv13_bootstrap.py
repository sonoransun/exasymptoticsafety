"""HV-13: audit of scattering/bootstrap.py (Veneziano / Virasoro-Shapiro).

Ground truth via mpmath (dps=40). Run with the repo venv python.
"""
import numpy as np
import mpmath as mp
mp.mp.dps = 40

from asymsafety.scattering import bootstrap as B
from asymsafety.validation.cheung_2025 import validate_bootstrap

rng = np.random.default_rng(20260609)
print("=" * 72)

# ---------------------------------------------------------------- 1. Veneziano value
print("\n[1] Veneziano vs mpmath ground truth")


def ven_true(s, t, a0=1.0, ap=1.0):
    a_s, a_t = a0 + ap * mp.mpc(s), a0 + ap * mp.mpc(t)
    return mp.gamma(-a_s) * mp.gamma(-a_t) / mp.gamma(-a_s - a_t)


# 1a: complex reference point from the derive notes
s_ref, t_ref = 1.3 + 0.7j, -2.2 + 0.4j
ref = ven_true(s_ref, t_ref)
print("   mpmath A(sref,tref) =", mp.nstr(ref, 16),
      " (notes: -0.7158388052 - 0.5047036815i)")
try:
    out = B.veneziano(s_ref, t_ref)
    print("   code at complex point:", out,
          " rel err:", abs(complex(out) - complex(ref)) / abs(complex(ref)))
except Exception as e:
    print("   code at complex point RAISES:", type(e).__name__, "--", e)

# 1b: real points, rtol vs ground truth
worst = 0.0
for _ in range(40):
    s = float(rng.uniform(-3, 6)) + 0.37  # avoid integers (poles at s=-1,0,1,..)
    t = float(rng.uniform(-6, 0)) + 0.13
    if abs((1 + s) - round(1 + s)) < 0.05 or abs((1 + t) - round(1 + t)) < 0.05:
        continue
    a = complex(B.veneziano(s, t))
    e = complex(ven_true(s, t))
    worst = max(worst, abs(a - e) / abs(e))
print(f"   real points: worst rtol vs mpmath = {worst:.3e}  (target 1e-10)")

# ---------------------------------------------------------------- 2. crossing
print("\n[2] Veneziano crossing A(s,t)=A(t,s), 20 random real points")
worst = 0.0
for _ in range(20):
    s = float(rng.uniform(-4, 6)) + 0.21
    t = float(rng.uniform(-6, 2)) + 0.09
    a, b = complex(B.veneziano(s, t)), complex(B.veneziano(t, s))
    worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
print(f"   worst relative asymmetry = {worst:.3e}")

# ---------------------------------------------------------------- 3. pole locations
print("\n[3] mass_spectrum vs actual poles of code amplitude")
spec = B.mass_spectrum(5)
print("   mass_spectrum(5) =", spec)
for sn in spec:
    eps = 1e-7
    big = abs(complex(B.veneziano(sn + eps, -2.3)))
    away = abs(complex(B.veneziano(sn + 0.5, -2.3)))
    print(f"   s_n={sn:+.1f}: |A(s_n+1e-7)|={big:.3e}  |A(s_n+0.5)|={away:.3e}  "
          f"pole={'YES' if big > 1e4 * max(away, 1e-30) else 'NO'}")

# ---------------------------------------------------------------- 4. residues
print("\n[4] residues: numerical lim (a_s-n) A vs code veneziano_residue")
print("   ground truth (w.r.t. a_s): R_n(a_t) = -(1/n!) prod_{k=1..n}(a_t+k)")
for n in range(0, 4):
    sn = (n - 1.0) / 1.0  # a0=1, ap=1
    for t in (-2.7, -3.9, -0.6):
        a_t = 1.0 + t
        # numerical residue via symmetric limit, Richardson in eps
        vals = []
        for eps in (1e-5, 1e-6):
            vals.append(eps * complex(B.veneziano(sn + eps, t)))
        num = (1e1 * vals[1] - vals[0]) / (1e1 - 1.0)  # crude extrapolation
        truth = -1.0
        for kk in range(1, n + 1):
            truth *= (a_t + kk)
        truth /= mp.factorial(n)
        truth = float(truth)
        code = B.veneziano_residue(n, t)
        print(f"   n={n} t={t:+.1f}: numeric={num.real:+.8f}  truth={truth:+.8f}"
              f"  code={code:+.8f}  code_matches_numeric="
              f"{abs(code - num.real) < 1e-4 * max(1, abs(num))}")

# n=2 polynomial coefficient extraction (degree check) from numeric residues
print("\n   n=2 residue polynomial fit in t (numeric, from amplitude limit):")
ts = np.array([-4.3, -2.9, -1.7, -0.8, 0.6])
res = []
for t in ts:
    eps = 1e-6
    r1 = eps * complex(B.veneziano(1.0 + eps, t))
    r2 = 1e-7 * complex(B.veneziano(1.0 + 1e-7, t))
    res.append(((10 * r2 - r1) / 9).real)
coef = np.polyfit(ts, res, 3)
print("   cubic fit coeffs (t^3,t^2,t,1):", np.round(coef, 9),
      "  expected (0, -0.5, -2.5, -3)")

# ---------------------------------------------------------------- 5. Regge
print("\n[5] Regge: slope of log|A| vs log s at a_s=N+1/2 (between poles)")
t0 = -2.2  # a_t = -1.2 -> expect slope ~ -1.2 along half-integer ladder
Ns = np.arange(5, 60, 3)
svals = (Ns + 0.5 - 1.0)  # a0=1, ap=1: s = a_s - 1
amps = np.array([abs(complex(B.veneziano(s, t0))) for s in svals])
good = np.isfinite(amps) & (amps > 0)
slope = np.polyfit(np.log(svals[good]), np.log(amps[good]), 1)[0]
print(f"   fitted slope = {slope:.4f} over {good.sum()} pts, "
      f"alpha(t) = {1 + t0:.4f}")
# mpmath check of the Regge limit off the real axis (notes: ratio -> 1)
sbig = mp.mpc(0, 1e4)
ratio = ven_true(sbig, t0) / (mp.gamma(-(1 + t0)) * (-(sbig + 0)) ** (1 + t0))
print("   mpmath |s|=1e4i: A / [Gamma(-a_t)(-a's)^a_t] =", mp.nstr(ratio, 8))

# ---------------------------------------------------------------- 6. Virasoro-Shapiro
print("\n[6] Virasoro-Shapiro implementation")


def vs_true(s, t, u, a0, ap):
    a_s, a_t, a_u = (a0 + ap * mp.mpc(x) for x in (s, t, u))
    return (mp.gamma(-a_s) * mp.gamma(-a_t) * mp.gamma(-a_u)
            / (mp.gamma(1 + a_s) * mp.gamma(1 + a_t) * mp.gamma(1 + a_u)))


# 6a: massless convention (a0=0, ap=0.25), notes reference value
s, t = 1.3 + 0.7j, -2.2 + 0.4j
u = -s - t
print("   true VS (Gamma(1+a) denom), massless a0=0 ap=0.25, ref point:",
      mp.nstr(vs_true(s, t, u, 0.0, 0.25), 16),
      "(notes: 8.874936257+7.385956960i)")

# code with massless convention at REAL massless points: is it identically 1?
print("   code VS with alpha0=0 (massless constraint sum a_i = 0):")
for (ss, tt) in [(2.3, -1.1), (7.7, -3.3), (0.9, -0.2)]:
    uu = -ss - tt
    val = B.virasoro_shapiro(ss, tt, uu, alpha0=0.0, alphap=0.25)
    true = complex(vs_true(ss, tt, uu, 0.0, 0.25))
    print(f"     s={ss} t={tt}: code={val}   true_VS={true:.6f}")

# 6b: code defaults (alpha0=1, ap=0.25) with massless kinematics u=-s-t
# => sum a_i = 3; denominator Gamma(a_i+a_j) = Gamma(3-a_k) -> rational fn
print("   code VS defaults (alpha0=1) on u=-s-t: compare to rational prediction")
for (ss, tt) in [(5.0, -2.0), (11.0, -4.4), (30.1, -10.0)]:
    uu = -ss - tt
    a = [1 + 0.25 * x for x in (ss, tt, uu)]
    pred = 1.0
    for ak in a:
        pred *= (-ak) * (1 - ak) * (2 - ak)
    pred = 1.0 / pred
    val = B.virasoro_shapiro(ss, tt, uu)
    true = complex(vs_true(ss, tt, uu, 1.0, 0.25))
    print(f"     s={ss}: code={complex(val):.6e}  rational-pred={pred:.6e}  "
          f"true-VS={true:.6e}")

# 6c: pole count of the code VS along s (alpha0=1, ap=0.25, massless kinematics)
print("   poles of code VS in s-channel (should be infinite tower for true VS):")
print("     code denominator vanishing <=> a_k in {0,1,2}: only 3 levels/channel")
for n in range(0, 8):
    sn = 4.0 * (n - 1.0)  # a_s = n
    if sn == 0:
        sn = 1e-9
    cc = 0.3
    tt, uu = -(sn / 2) * (1 - cc), -(sn / 2) * (1 + cc)
    v_near = B.virasoro_shapiro(sn + 1e-6, tt, uu)
    v_far = B.virasoro_shapiro(sn + 0.7, tt - 0.35 * 0.7, uu - 0.35 * 0.7)
    pole = abs(complex(v_near)) > 1e3 * max(abs(complex(v_far)), 1e-30)
    print(f"     level n={n} (s={sn:+.1f}): pole_in_code={pole}")

# 6d: full s,t,u symmetry of the code function (any Gamma-symmetric fn passes)
worst = 0.0
import itertools
for _ in range(10):
    ss = float(rng.uniform(0.5, 8)) + 0.123
    tt = float(rng.uniform(-8, -0.5))
    uu = -ss - tt
    vals = [complex(B.virasoro_shapiro(*p)) for p in itertools.permutations((ss, tt, uu))]
    m = max(abs(v - vals[0]) for v in vals)
    worst = max(worst, m / max(abs(vals[0]), 1e-300))
print(f"   full S3 symmetry of code VS: worst rel diff = {worst:.3e} (trivially true)")

# ---------------------------------------------------------------- 7. cheung_2025
print("\n[7] validation/cheung_2025.validate_bootstrap()")
res = validate_bootstrap()
for k, v in res.items():
    if isinstance(v, dict):
        print(f"   {k}: passed={v['passed']} computed={v['computed']}")
print("   all_passed =", res["all_passed"])

# 7b: what does ultrasoft actually measure for the (broken) VS adapter?
sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
print("   StringAmplitude defaults: alpha0 =", sa.alpha0, " (massless kinematics u=-s-t)")
soft = B.ultrasoft_falloff(sa, cos_theta=0.3, s_lo=5.0, s_hi=50.0)
print("   ultrasoft dict:", {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in soft.items()})
# asymptotic slope of the rational function (pole-free regime):
s_hi = np.geomspace(2e2, 2e4, 40)
A_hi = np.abs(np.asarray(sa.amplitude_vs_s(s_hi, 0.3)))
slope_tail = np.polyfit(np.log(s_hi), np.log(A_hi), 1)[0]
print(f"   asymptotic tail slope of code 'VS' = {slope_tail:.3f} "
      "(a pure power law ~ s^-9, NOT super-polynomial/exponential)")
soft2 = B.ultrasoft_falloff(sa, cos_theta=0.3, s_lo=2e2, s_hi=2e4)
print("   ultrasoft re-run on pole-free window [2e2,2e4]:",
      {k: round(v, 4) if isinstance(v, float) else v for k, v in soft2.items()})

# 7c: higher-spin residue check circularity: verify the AMPLITUDE residues
# actually vanish at residue_zeros (independent numeric check)
print("   independent check of residue zeros against the amplitude itself:")
for n in (1, 2, 3):
    for tz in B.residue_zeros(n):
        r = 1e-6 * complex(B.veneziano((n - 1.0) + 1e-6, tz))
        print(f"     n={n} t={tz:+.1f}: numeric residue = {abs(r):.2e}")

# 7d: odd-n sign: does the cheung check catch it? (it evaluates only at zeros)
print("   n=1 sign: code residue(1, t=-3) =", B.veneziano_residue(1, -3.0),
      " truth =", -(1 + (1 + (-3.0))))
print("=" * 72)
