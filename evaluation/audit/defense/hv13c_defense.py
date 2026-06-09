"""HV-13c DEFENSE: is there ANY convention under which
   A = G(-a_s)G(-a_t)G(-a_u) / [G(a_s+a_t)G(a_t+a_u)G(a_u+a_s)]
is the Virasoro-Shapiro amplitude?

Candidate conventions tested:
  (A) Virasoro 1969 original symmetric form (the ONE literature form with
      positive-sum denominators): A ~ G(a)G(b)G(c)/[G(a+b)G(b+c)G(c+a)]
      with a = -alpha(s)/2 etc. and constraint a+b+c = 1
      (closed-tachyon trajectory alpha(x) = 2 + alpha' x/2, s+t+u = -16/a').
      -> requires numerator G(+a), code has G(-a). Check both sign
         identifications numerically.
  (B) Closed-tachyon kinematics matching the code defaults
      (alpha0=1, alphap=1/4 => tachyon at s=-4, i.e. m^2=-4):
      u = 4 m^2 - s - t = -16 - s - t, so sum a_i = -1.
      On THIS surface G(-a_j-a_k) = G(1+a_i) is the legit symmetric form.
      Code has G(+a_j+a_k) = G(-1-a_i). Compare code vs true VS vs the
      polynomial prediction -(1+a_s)(1+a_t)(1+a_u).
  (C) Algebraic impossibility: on any constraint surface sum a_i = C,
      code denominator G(a_j+a_k) = G(C-a_i); the ratio
      G(-a_i)/G(C-a_i) is rational for integer C -> finitely many poles,
      never an infinite Regge tower. Scan C in {-1, 0, 1, 2, 3} and count
      poles up to level 12.
"""
import numpy as np
import mpmath as mp
import itertools

mp.mp.dps = 40
from asymsafety.scattering import bootstrap as B


def vs_true(a_s, a_t, a_u):
    """Reference VS: prod G(-a_i) / prod G(1+a_i)."""
    return (mp.gamma(-a_s) * mp.gamma(-a_t) * mp.gamma(-a_u)
            / (mp.gamma(1 + a_s) * mp.gamma(1 + a_t) * mp.gamma(1 + a_u)))


def code_form(a_s, a_t, a_u):
    """The code's formula evaluated in mpmath (no trajectory wrapper)."""
    return (mp.gamma(-a_s) * mp.gamma(-a_t) * mp.gamma(-a_u)
            / (mp.gamma(a_s + a_t) * mp.gamma(a_t + a_u) * mp.gamma(a_u + a_s)))


def virasoro_1969(a, b, c):
    """Virasoro's ORIGINAL form: G(a)G(b)G(c)/[G(a+b)G(b+c)G(c+a)],
    valid VS only with a = -alpha_s/2 etc. and a+b+c = 1."""
    return (mp.gamma(a) * mp.gamma(b) * mp.gamma(c)
            / (mp.gamma(a + b) * mp.gamma(b + c) * mp.gamma(c + a)))


print("=" * 76)
print("(A) Virasoro 1969 form, constraint a+b+c=1 (a=-alpha_s/2, sum alpha=-2)")
print("    First: confirm Virasoro-1969 == standard VS on its surface")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(6):
    # pick alpha_s, alpha_t random, alpha_u = -2 - alpha_s - alpha_t
    al_s = float(rng.uniform(-3, 5)) + 0.217
    al_t = float(rng.uniform(-6, 1)) + 0.133
    al_u = -2.0 - al_s - al_t
    a, b, c = -al_s / 2, -al_t / 2, -al_u / 2          # a+b+c = 1
    v69 = virasoro_1969(a, b, c)
    # standard form with half-trajectory variables h_i = alpha_i/2,
    # sum h_i = -1 (closed-tachyon constraint)
    std = vs_true(al_s / 2, al_t / 2, al_u / 2)
    worst = max(worst, abs(complex(v69 - std)) / abs(complex(std)))
print(f"    Virasoro1969 vs standard VS: worst rtol = {worst:.2e}  "
      "(positive-sum denominators ARE a real convention)")

print("\n    Now: can the CODE formula match it under either sign identification?")
print("    code a_i := -alpha_i/2  (numerator would match Virasoro: G(-a_i)=G(+a))")
for _ in range(3):
    al_s = float(rng.uniform(-3, 5)) + 0.31
    al_t = float(rng.uniform(-6, 1)) + 0.11
    al_u = -2.0 - al_s - al_t
    A = [-al_s / 2, -al_t / 2, -al_u / 2]   # sum A_i = +1
    cf = code_form(*A)
    v69 = virasoro_1969(*A)
    # prediction: with sum A = 1, G(A_j+A_k)=G(1-A_i) and
    # G(-A_i)/G(1-A_i) = 1/(-A_i)  => code = -1/(A_s A_t A_u)
    pred = -1.0 / (A[0] * A[1] * A[2])
    print(f"      alpha=({al_s:+.3f},{al_t:+.3f},{al_u:+.3f}): "
          f"code={complex(cf):+.6e}  trueVS={complex(v69):+.6e}  "
          f"rational -1/(A_s A_t A_u)={pred:+.6e}")

print("\n    code a_i := +alpha_i/2  (denominator G(a_j+a_k)=G(-1-a_i): wrong)")
for _ in range(3):
    al_s = float(rng.uniform(-3, 5)) + 0.41
    al_t = float(rng.uniform(-6, 1)) + 0.23
    al_u = -2.0 - al_s - al_t
    A = [al_s / 2, al_t / 2, al_u / 2]      # sum A_i = -1
    cf = code_form(*A)
    tv = vs_true(*A)
    pred = -(1 + A[0]) * (1 + A[1]) * (1 + A[2])   # polynomial prediction
    print(f"      alpha=({al_s:+.3f},{al_t:+.3f},{al_u:+.3f}): "
          f"code={complex(cf):+.6e}  trueVS={complex(tv):+.6e}  "
          f"poly -(1+a)(1+b)(1+c)={pred:+.6e}")

print("\n" + "=" * 76)
print("(B) Most charitable kinematics for the code DEFAULTS alpha0=1, ap=1/4:")
print("    closed-tachyon m^2=-4, u = -16 - s - t  => sum a_i = -1")
for (ss, tt) in [(5.3, -2.2), (13.7, -4.4), (2.1, -0.7)]:
    uu = -16.0 - ss - tt
    a = [1 + 0.25 * x for x in (ss, tt, uu)]
    cf = complex(B.virasoro_shapiro(ss, tt, uu))      # actual code call
    tv = complex(vs_true(*a))
    pred = -(1 + a[0]) * (1 + a[1]) * (1 + a[2])
    print(f"      s={ss:5.1f} t={tt:5.1f}: code={cf:+.6e}  "
          f"trueVS={tv:+.6e}  poly={pred:+.6e}  "
          f"code==poly: {abs(cf - pred) < 1e-9 * max(1, abs(pred))}")

print("\n" + "=" * 76)
print("(C) Pole census on every integer constraint surface sum a_i = C")
print("    (true VS has an infinite tower on ALL of them)")
for C in (-1, 0, 1, 2, 3):
    npoles_code, npoles_true = 0, 0
    for n in range(0, 13):
        # a_s = n + eps, split remainder C - a_s between a_t, a_u, off-integer
        a_s = n + 1e-8
        a_t = (C - a_s) * 0.5 - 0.37
        a_u = C - a_s - a_t
        c_near = abs(complex(code_form(a_s, a_t, a_u)))
        t_near = abs(complex(vs_true(a_s, a_t, a_u)))
        a_s2 = n + 0.5
        a_t2 = (C - a_s2) * 0.5 - 0.37
        a_u2 = C - a_s2 - a_t2
        c_far = abs(complex(code_form(a_s2, a_t2, a_u2)))
        t_far = abs(complex(vs_true(a_s2, a_t2, a_u2)))
        if c_near > 1e4 * max(c_far, 1e-300):
            npoles_code += 1
        if t_near > 1e4 * max(t_far, 1e-300):
            npoles_true += 1
    print(f"      sum a_i = {C:+d}: code poles (n<=12) = {npoles_code:2d}   "
          f"true VS poles = {npoles_true:2d}")

print("\n" + "=" * 76)
print("(D) Exhaustive: G(a_j+a_k) = G(1+a_i) would need sum a = 1 + 2 a_i")
print("    for every channel simultaneously -> a_s=a_t=a_u, measure-zero.")
print("    No linear kinematic constraint can equate the code form with VS.")
