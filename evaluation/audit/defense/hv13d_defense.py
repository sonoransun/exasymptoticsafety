"""HV-13d defense attempt: hunt for ANY convention legitimizing
StringAmplitude(kind='virasoro_shapiro', alpha0=1, alphap=0.25, m=0)."""
import numpy as np
import mpmath as mp
mp.mp.dps = 30
from asymsafety.scattering import bootstrap as B

print("== Defense candidate 1: documented 'effectively massless' approximation ==")
print("If massless kinematics were a small O(m^2/s) deformation of the tachyonic")
print("constraint surface, code(s) ~ trueVS(s) at large s. Compare:")
def vs_gsw(s, t, u):  # GSW closed-bosonic-tachyon form, Gamma(1+a) denominators
    a = [mp.mpf(1) + mp.mpf(x)/4 for x in (s, t, u)]
    return (mp.gamma(-a[0])*mp.gamma(-a[1])*mp.gamma(-a[2]) /
            (mp.gamma(1+a[0])*mp.gamma(1+a[1])*mp.gamma(1+a[2])))
for s in (10.3, 30.1, 60.7):
    t = -0.35*s; u_ms = -s-t; u_os = -s-t-16.0   # massless vs on-shell tachyon
    code = complex(B.virasoro_shapiro(s, t, u_ms))
    true_ms = complex(vs_gsw(s, t, u_ms))   # true form, massless kinematics
    true_os = complex(vs_gsw(s, t, u_os))   # true form, on-shell surface
    print(f"  s={s:6.1f}: code={code:.4e}  trueForm(massless)={true_ms:.4e}  "
          f"trueForm(on-shell)={true_os:.4e}")

print("\n== Defense candidate 2: any constraint surface sum(a_i)=sigma that makes")
print("the LITERAL code formula prod G(-a_i)/prod G(a_i+a_j) a dual amplitude ==")
print("Test: residue at the a_s=n pole must be polynomial in t (no t-poles).")
for sigma in (-1.0, 0.0, 1.0, 2.0, 3.0, 0.5):
    # parametrize surface: a_s=n+eps, a_t free, a_u = sigma - a_s - a_t
    n = 4; eps = 1e-7
    a_s = n + eps
    out = []
    for a_t in (-2.3, -2.3+1e-6):   # probe t-analyticity of the residue
        a_u = sigma - a_s - a_t
        with np.errstate(all='ignore'):
            num = float(mp.re(mp.gamma(-a_s)*mp.gamma(-a_t)*mp.gamma(-a_u) /
                  (mp.gamma(a_s+a_t)*mp.gamma(a_t+a_u)*mp.gamma(a_u+a_s))))
        out.append(eps*num)
    print(f"  sigma={sigma:+.1f}: residue(level n=4) = {out[0]:.6e} "
          f"({'POLE SURVIVES' if abs(out[0])>1e-12 else 'pole cancelled -> no tower'})")

print("\n== Defense candidate 3: code formula == true VS under intercept redef? ==")
print("Code on sum=3 equals rational 1/prod[(-a)(1-a)(2-a)]: check & count poles")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(200):
    s = float(rng.uniform(0.3, 40)); t = float(rng.uniform(-40, -0.3)); u = -s-t
    a = [1+0.25*x for x in (s,t,u)]
    if min(abs(ai-round(ai)) for ai in a) < 0.02: continue
    pred = 1.0
    for ak in a: pred *= (-ak)*(1-ak)*(2-ak)
    pred = 1.0/pred
    val = complex(B.virasoro_shapiro(s,t,u))
    worst = max(worst, abs(val-pred)/max(abs(pred),1e-300))
print(f"  max rel diff code-vs-rational over 200 random massless pts: {worst:.2e}")
print("  -> exactly rational; a rational function is NEVER a VS amplitude")
print("     (VS has an infinite pole tower and exponential fixed-angle falloff).")

print("\n== Defense candidate 4: does the shipped diagnostic conclusion survive? ==")
sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
for lo, hi in [(5,50),(2e2,2e4)]:
    d = B.ultrasoft_falloff(sa, cos_theta=0.3, s_lo=lo, s_hi=hi)
    print(f"  window [{lo:g},{hi:g}]: slope_lo={d['slope_lo']:.3f} "
          f"slope_hi={d['slope_hi']:.3f} ultrasoft={d['ultrasoft']}")
print("  -> 'ultrasoft=True' on [5,50] is a pole-straddling fit artifact;")
print("     the function is a pure s^-9 power law (not ultrasoft).")
