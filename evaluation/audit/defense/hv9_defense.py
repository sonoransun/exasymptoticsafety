"""HV-9 defense probes.

1. Rescaling test: replace the volume prefactor 8/pi by c*8/pi AND rescale
   N,D in eta uniformly -> does (lam*, theta) move? (If not, 'threshold
   normalization' cannot explain the lam*/theta gap.)
2. Gauge-knob test: is there any single overall-weight freedom in the
   toolkit's own structure that brings theta to a complex pair ~1.47+-3.04i
   or lam* to 0.193?
3. Count relevant directions of toolkit FP (vs REUTER_FP n_relevant=2).
"""
import numpy as np
import sympy
from sympy import Rational, Symbol, nsolve, im, re

g = Symbol("g", positive=True)
lam = Symbol("lambda", real=True)

def make_system(c_vol=1.0, c_eta=1.0):
    one_m_2l = 1 - 2*lam
    N_eta = c_eta*(Rational(4,3)/one_m_2l + 8 - 8/one_m_2l**2)
    D_eta = c_eta*(Rational(1,3)/one_m_2l + 2 - 2/one_m_2l**2)
    eta = g*N_eta/(1 - g*D_eta)
    bg = (2 + eta)*g
    vol = (3/one_m_2l - 4 - eta*(1/one_m_2l - Rational(4,3)))
    bl = -(2 - eta)*lam + c_vol*8*g/sympy.pi*vol
    return bg, bl

def fp_and_theta(c_vol, c_eta, guess=(0.69, 0.142)):
    bg, bl = make_system(c_vol, c_eta)
    try:
        sol = sympy.nsolve((bg, bl), (g, lam), guess, prec=20)
    except Exception as e:
        return None
    gs, ls = float(sol[0]), float(sol[1])
    J = sympy.Matrix([[sympy.diff(b, v) for v in (g, lam)] for b in (bg, bl)])
    Jn = np.array(J.subs({g: gs, lam: ls}).evalf(), dtype=float)
    th = -np.linalg.eigvals(Jn)
    return gs, ls, th

print("baseline:", fp_and_theta(1.0, 1.0))
for c in (0.0625, 0.25, 0.5, 2.0, 4.0, 16.0):
    r = fp_and_theta(c, c)  # uniform 'threshold normalization' rescale
    print(f"c_vol=c_eta={c:8.4f} ->", None if r is None else
          f"g*={r[0]:.4f} lam*={r[1]:.6f} theta={r[2]}")

# independent knobs (a crude 'gauge weight' scan): can ANY (c_vol,c_eta) pair
# reach lam*=0.193 AND complex theta 1.47+-3.04i with the toolkit's structure?
print("\nindependent knob scan:")
for cv in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
    for ce in (0.25, 0.5, 1.0, 2.0, 4.0):
        r = fp_and_theta(cv, ce)
        if r is None: continue
        gs, ls, th = r
        cplx = abs(th[0].imag) > 1e-6
        print(f"  cv={cv:7.4f} ce={ce:5.2f}: lam*={ls:+.4f} "
              f"theta_re={th[0].real:+8.3f} theta_im={abs(th[0].imag):7.3f} "
              f"n_relevant={(th.real>0).sum()}")
