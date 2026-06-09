"""HV-14: audit of scattering/amplitude.py + consistency.py (graviton exchange).

Run with the repo venv python.
"""
import itertools
import math

import numpy as np
import mpmath as mp
mp.mp.dps = 30

import sys
sys.path.insert(0, "/root/cdev/exasymptoticsafety/tests")
from conftest import make_as_trajectory

from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.scattering.scale import FixedScale
from asymsafety.scattering import consistency as C
from asymsafety.validation.knorr_2026 import (
    make_unsafe_amplitude, validate_safe_vs_unsafe)

rng = np.random.default_rng(20260609)
traj = make_as_trajectory()          # g*=0.7, crossover g(t)=g* e^2t/(1+e^2t)
ff = GravitonFormFactor(traj)
amp = GravitonMediatedAmplitude(ff)
print("=" * 72)
print("G_N =", ff.newton_constant())

# ---------------------------------------------------------------- (a) crossing perms
print("\n[a] permutation symmetry of eval() (dressed), 12 random points")
worst = 0.0
for _ in range(12):
    s = float(rng.uniform(0.5, 50.0))
    c = float(rng.uniform(-0.95, 0.95))
    t = -(s / 2) * (1 - c)
    u = -s - t
    vals = [complex(amp.eval(*p, dressed=True))
            for p in itertools.permutations((s, t, u))]
    m = max(abs(v - vals[0]) for v in vals) / max(abs(vals[0]), 1e-300)
    worst = max(worst, m)
print(f"   worst relative asymmetry over all 6 perms = {worst:.3e} (tol 1e-8)")

# bitwise tautology demonstration for the s<->t swap used by consistency.crossing
s0, t0 = 5.0, -2.0
u0 = -s0 - t0
d1 = complex(amp.eval(s0, t0, u0, dressed=True))
d2 = complex(amp.eval(t0, s0, u0, dressed=True))
print(f"   eval(s,t,u) = {d1}, eval(t,s,u) = {d2}, bitwise equal: {d1 == d2}")

# ---------------------------------------------------------------- (b) overall sign
print("\n[b] sign adjudication")
GN = ff.newton_constant()
# code GR amplitude vs the +8piG[su/t+tu/s+st/u] reference at one point
ref = 8.0 * math.pi * GN * (s0 * u0 / t0 + t0 * u0 / s0 + s0 * t0 / u0)
code = complex(amp.gr_amplitude(s0, t0, u0)).real
print(f"   code M_GR(5,-2,-3) = {code:+.6f}; notes(+8piG sum) = {ref:+.6f};"
      f" ratio = {code / ref:+.3f}")
# forward limit: notes say M ~ -8 pi G s^2/t > 0 for attraction (mostly-minus,
# S=1+iT, Born Vtilde = -M/(4 m1 m2))
s_f, t_f = 10.0, -1e-6
u_f = -s_f - t_f
fwd = complex(amp.gr_amplitude(s_f, t_f, u_f)).real
print(f"   code forward M_GR(s=10, t=-1e-6) = {fwd:+.4e}  "
      f"(notes' attractive convention requires > 0)")
# DKRS single-channel comparison: code's s-channel term is -8piG tu/s
m_s_code = complex(amp._channel(np.asarray(s0), t0 * u0, False))
print(f"   code s-channel term = {m_s_code.real:+.6f};"
      f"  DKRS 2007.04396 A = +8 pi G tu/s = {8 * math.pi * GN * t0 * u0 / s0:+.6f}")

# ---------------------------------------------------------------- (c) IR limit
print("\n[c] IR limit: M_dressed/M_GR -> 1 over 6 decades (cos=0.3)")
for s in np.geomspace(1e-8, 1e-2, 7):
    d = complex(amp.amplitude_vs_s(np.array([s]), 0.3, dressed=True)[0])
    g = complex(amp.amplitude_vs_s(np.array([s]), 0.3, dressed=False)[0])
    print(f"   s={s:.1e}: ratio = {abs(d) / abs(g):.6f}")

# ---------------------------------------------------------------- (d) UV exponents
print("\n[d] UV growth exponents (fit of log|M| vs log s, s in [1e4,1e8], cos=0.3)")
s_uv = np.geomspace(1e4, 1e8, 60)


def slope(a, dressed):
    M = np.abs(a.amplitude_vs_s(s_uv, 0.3, dressed=dressed))
    return float(np.polyfit(np.log(s_uv), np.log(M), 1)[0])


print(f"   dressed (EnergyScale):  {slope(amp, True):+.4f}   (expect ~0)")
print(f"   undressed GR:           {slope(amp, False):+.4f}   (expect ~1)")
unsafe = make_unsafe_amplitude(traj, k_fixed=1.0)
print(f"   FixedScale 'unsafe':    {slope(unsafe, True):+.4f}   (expect ~1)")
print("   knorr validate_safe_vs_unsafe:",
      {k: (v if not isinstance(v, dict) else
           {"computed": round(v["computed"], 4), "passed": v["passed"]})
       for k, v in validate_safe_vs_unsafe(amp, unsafe).items()})

# UV constant value check: M -> -8 pi g* [u/t*sgn + ...]? each term
# -8pi g(x)/|x| * num/x; analytic prediction at cos=0.3 for k^2=|x| per channel:
c = 0.3
t_frac, u_frac = -(1 - c) / 2, -(1 + c) / 2  # t/s, u/s
g_star = 0.7
pred = -8 * math.pi * g_star * (
    (t_frac * u_frac) / 1.0          # s-channel: G(s)=g*/s, num tu/s -> tu/s^2
    + (1 * u_frac) / (t_frac * abs(t_frac))   # t-channel: g*/|t| * su/t /s^2
    + (1 * t_frac) / (u_frac * abs(u_frac)))
M_uv = complex(amp.amplitude_vs_s(np.array([1e8]), c, dressed=True)[0]).real
print(f"   UV plateau: code M(1e8) = {M_uv:+.5f}, analytic = {pred:+.5f}")

# ---------------------------------------------------------------- (e) partial waves
print("\n[e0] partial_wave normalization vs ground truth (mock DKRS amplitude)")


class MockDKRS:
    """M = 8 pi G tu/s = 2 pi G s (1-c^2) with G=1 (annihilation channel)."""

    def eval(self, s, t, u, *, dressed=True):
        return 8.0 * math.pi * np.asarray(t) * np.asarray(u) / np.asarray(s)


for ell, truth in ((0, 1.0 / 12), (1, 0.0), (2, -1.0 / 60), (3, 0.0), (4, 0.0)):
    a = C.partial_wave(MockDKRS(), ell, 1.0, cos_max=1.0, n_theta=20001)
    print(f"   l={ell}: code a_l/s = {a.real:+.9f}   truth (Gs={1}) = {truth:+.9f}")

print("\n[e] unitarity() conclusions vs cos_max")
rows = {}
for cm in (0.9, 0.99, 0.999):
    r = C.unitarity(amp, cos_max=cm)
    rows[cm] = r
    print(f"   cos_max={cm}: as_max={r['as_max']:.4g} gr_max={r['gr_max']:.4g} "
          f"as_growth={r['as_growth_exponent']:+.3f} "
          f"gr_growth={r['gr_growth_exponent']:+.3f} "
          f"elastic_bound={r['as_satisfies_elastic_bound']} "
          f"as_bounded={r['as_bounded']} gr_unbounded={r['gr_unbounded']} "
          f"passed={r['passed']}")
flips = [k for k in ("as_satisfies_elastic_bound", "as_bounded",
                     "gr_unbounded", "passed")
         if len({rows[cm][k] for cm in rows}) > 1]
print("   flags that flip with cutoff:", flips if flips else "none")
# log-divergence scaling check of the AS plateau: a0 ~ ln(2/delta)?
import numpy as _np
plat = {cm: rows[cm]["a_dressed"][-1] for cm in rows}
print("   a_0 plateau at s=1e6 vs cutoff:",
      {cm: f"{v:.4g}" for cm, v in plat.items()},
      "ratios:", [round(plat[0.99] / plat[0.9], 3),
                  round(plat[0.999] / plat[0.99], 3)],
      "ln ratios:", [round(math.log(2 / 0.01) / math.log(2 / 0.1), 3),
                     round(math.log(2 / 0.001) / math.log(2 / 0.01), 3)])

# ---------------------------------------------------------------- (f) crossing check
print("\n[f] consistency.crossing(): what it compares")
r = C.crossing(amp)
print("   crossing dict:", r)
print("   residual is exactly 0.0:", r["residual"] == 0.0)


# Can it EVER fail? feed an intentionally crossing-violating amplitude
class Broken:
    """k^2 = s global scale identification (crossing-breaking per notes)."""

    def eval(self, s, t, u, *, dressed=True):
        s = np.asarray(s, dtype=float)
        G = 1.0 / (1.0 + np.abs(s))          # G depends on s ONLY
        return -8 * math.pi * G * (np.asarray(t) * np.asarray(u) / s
                                   + s * np.asarray(u) / np.asarray(t)
                                   + s * np.asarray(t) / np.asarray(u))


rb = C.crossing(Broken())
print("   crossing() on a deliberately crossing-VIOLATING amplitude:", rb)
print("=" * 72)
