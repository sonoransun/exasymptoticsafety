"""HV-13e defense probes.

D1: Is there a kinematic convention (tachyon constraint sum a_i = -1)
    where code's Gamma(a_s+a_t) denominator equals the standard VS?
D2: Would the TRUE VS pass ultrasoft_falloff on the same window [5,50]
    (i.e., is the check methodology sound if the amplitude were correct)?
D3: Local log-slope of the code 'VS' rational function in the upper half
    window: is slope_hi=-9.66 genuine local behavior or spike-driven?
D4: Does higher_spin check pass even with the residue sign bug *and*
    would it pass for ANY polynomial sharing the same roots?
"""
import numpy as np
from scipy.special import gamma
from asymsafety.scattering import bootstrap as B

print("=" * 70)
print("[D1] tachyonic-kinematics convention rescue attempt (sum a_i = -1)")
# closed bosonic tachyon: alpha0=1, alphap=0.25 -> m^2=-4, s+t+u=4m^2=-16
# then code denom Gamma(a_s+a_t)=Gamma(-1-a_u): compare code vs standard VS
def vs_std(s, t, u, a0, ap):  # standard: Gamma(-as-at) denom
    a = [a0 + ap*x for x in (s, t, u)]
    num = gamma(-a[0])*gamma(-a[1])*gamma(-a[2])
    den = gamma(-a[0]-a[1])*gamma(-a[1]-a[2])*gamma(-a[2]-a[0])
    return num/den
for (s, t) in [(5.3, -7.7), (13.1, -20.0), (33.3, -40.0)]:
    u = -16.0 - s - t
    code = complex(B.virasoro_shapiro(s, t, u))
    std = complex(vs_std(s, t, u, 1.0, 0.25))
    # code on the constraint reduces to product of (-1-a_k): cubic poly
    a = [1 + 0.25*x for x in (s, t, u)]
    cubic = np.prod([-1.0 - ak for ak in a])
    print(f"  s={s} t={t}: code={code:.4e} cubic-pred={cubic:.4e} std-VS={std:.4e}")

print()
print("[D2] ultrasoft_falloff applied to the TRUE VS, same windows")
class TrueVS:
    """Correct massless closed-string VS: G(-a_s)G(-a_t)G(-a_u)/[G(1+a_s)G(1+a_t)G(1+a_u)], a=ap*x."""
    def __init__(self, ap=0.25): self.ap = ap
    def amplitude_vs_s(self, s_values, cos_theta, dressed=True):
        s = np.asarray(s_values, float)
        t = -(s/2)*(1-cos_theta); u = -(s/2)*(1+cos_theta)
        a_s, a_t, a_u = self.ap*s, self.ap*t, self.ap*u
        with np.errstate(all="ignore"):
            return (gamma(-a_s)*gamma(-a_t)*gamma(-a_u)
                    / (gamma(1+a_s)*gamma(1+a_t)*gamma(1+a_u)))
tv = TrueVS(0.25)
for (lo, hi) in [(5.0, 50.0), (2e2, 2e4), (50.0, 500.0)]:
    r = B.ultrasoft_falloff(tv, cos_theta=0.3, s_lo=lo, s_hi=hi)
    print(f"  true VS window [{lo:g},{hi:g}]: slope_lo={r['slope_lo']:.3f} "
          f"slope_hi={r['slope_hi']:.3f} super_poly={r['super_polynomial']} "
          f"ultrasoft={r['ultrasoft']}")
# smooth (between-pole midpoint) sampling of true VS to show the genuine trend
mids = np.array([4*(n+0.5) for n in range(1, 40)])  # between s-channel poles a_s=n
A = np.abs(tv.amplitude_vs_s(mids, 0.3))
g = np.isfinite(A) & (A > 0)
ls, lA = np.log(mids[g]), np.log(A[g])
h = len(ls)//2
print(f"  true VS midpoint ladder: slope_lo={np.polyfit(ls[:h],lA[:h],1)[0]:.2f} "
      f"slope_hi={np.polyfit(ls[h:],lA[h:],1)[0]:.2f}  (should keep steepening)")

print()
print("[D3] local slope of code 'VS' in upper half-window [15.8, 50] (pole-free?)")
sa = B.StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
# poles of code rational fn at fixed cos=0.3: a_k in {0,1,2}
# s-channel: s=-4,0,4 ; t: s=0,-11.43,+11.43? a_t=1-0.0875 s in {0,1,2} -> s=11.43,0,-11.43
# u: a_u=1-0.1625 s in {0,1,2} -> s=6.154,0,-6.154
print("  rational-fn poles in s>0: s = 4.0, 6.154, 11.43  (all below 15.8)")
ss = np.geomspace(15.8, 50, 30)
A = np.abs(np.asarray(sa.amplitude_vs_s(ss, 0.3)))
sl = np.polyfit(np.log(ss), np.log(A), 1)[0]
print(f"  genuine local slope on [15.8,50] = {sl:.3f}  (asymptote -9; "
      "steeper than -9 here, relaxing UP to -9 => NOT steepening)")
ss2 = np.geomspace(50, 500, 30)
A2 = np.abs(np.asarray(sa.amplitude_vs_s(ss2, 0.3)))
print(f"  local slope on [50,500] = {np.polyfit(np.log(ss2), np.log(A2),1)[0]:.3f}")

print()
print("[D4] higher_spin check power: evaluate WRONG-sign and even WRONG-degree")
for n in (1, 2, 3):
    z = B.residue_zeros(n)
    r_code = B.veneziano_residue(n, z)
    r_flip = -np.atleast_1d(r_code)          # opposite sign bug
    print(f"  n={n}: |code(zeros)|max={np.max(np.abs(np.atleast_1d(r_code))):.1e} "
          f"|flipped|max={np.max(np.abs(r_flip)):.1e}  -> both pass")
print("=" * 70)
