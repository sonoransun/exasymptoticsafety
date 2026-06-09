"""HV-5 defense check: independent verification of TT heat-kernel b2 on S^4.

Method differs from audit script: Richardson extrapolation of
b2_est(t) = (6 t^2 Tr(t) - b0)/t  instead of polynomial least-squares fit.
Degeneracies/eigenvalues are taken from the CODE'S OWN geometry.decomposition
(ModeSpectrum), not retyped, so any 'rigged ground truth' in the audit script
is ruled out.

Also tests every candidate operator -D^2 + c*R on the true TT bundle to see
which c (if any) reproduces the code's b2 = +5R/3, and verifies that the
code's value is exactly the '5 shifted scalar towers' model.
"""
import mpmath as mp
import sympy
from asymsafety.geometry.decomposition import ModeSpectrum
from asymsafety.frg.heat_kernel import SeeleyDeWittCoefficients

mp.mp.dps = 40
ms = ModeSpectrum(d=4)
one = sympy.Integer(1)

def trace_TT(t, cR=0.0):
    """Tr e^{-t(-D^2 + cR)} on TT, unit S^4 (R=12), using code's ModeSpectrum."""
    total = mp.mpf(0)
    lmax = int(mp.sqrt(120 / t)) + 30
    for l in range(2, lmax):
        deg = ms.multiplicity("TT", l)
        lam = float(ms.eigenvalue("TT", l, one)) + cR * 12.0
        total += deg * mp.e**(-t * lam)
    return total

def trace_5scalar_shifted(t):
    """The code's implicit model: 5 scalar towers with eigenvalue shift -2."""
    total = mp.mpf(0)
    lmax = int(mp.sqrt(120 / t)) + 30
    for l in range(0, lmax):
        deg = 5 * ms.multiplicity("scalar", l)
        lam = float(ms.eigenvalue("scalar", l, one)) - 2.0
        total += deg * mp.e**(-t * lam)
    return total

def richardson_b2(tracefun, b0=5.0):
    """b2 = lim_{t->0} (6 t^2 Tr - b0)/t via Richardson on geometric ladder."""
    ts = [mp.mpf("0.04") / 2**j for j in range(6)]
    est = [(6 * t**2 * tracefun(t) - b0) / t for t in ts]
    # Richardson: error ~ a t + b t^2 ... successive halving elimination
    for order in range(1, 4):
        est = [(2**order * est[i + 1] - est[i]) / (2**order - 1)
               for i in range(len(est) - 1)]
    return est[-1]

print("Independent (Richardson) b2, unit S^4 (R=12), code's own ModeSpectrum:")
b2_bochner = richardson_b2(lambda t: trace_TT(t, 0.0))
b2_lich = richardson_b2(lambda t: trace_TT(t, 2.0 / 3.0))
b2_codeE = richardson_b2(lambda t: trace_TT(t, -1.0 / 6.0))  # code's claimed E=-R/6
b2_model = richardson_b2(trace_5scalar_shifted)
print(f"  TT  -D^2            : b2 = {mp.nstr(b2_bochner, 8)}   (audit: -10 = -(5/6)R)")
print(f"  TT  Lichnerowicz    : b2 = {mp.nstr(b2_lich, 8)}   (audit: -50 = -(25/6)R)")
print(f"  TT  -D^2 - R/6 (code's documented E): b2 = {mp.nstr(b2_codeE, 8)}")
print(f"  5 shifted scalar towers (code model): b2 = {mp.nstr(b2_model, 8)}")

R = sympy.Symbol("R_bar", positive=True)
sd = SeeleyDeWittCoefficients(d=4, R_bar=R)
print(f"\nCode b2(TT_tensor) = {sd.b2('TT_tensor')}  -> {sd.b2('TT_tensor').subs(R, 12)} at R=12")

# Which c in -D^2 + cR would make the TRUE TT b2 equal the code's +5R/3?
# true b2(c) = -(5/6)R - 5cR  ==> -(5/6) - 5c = 5/3  ==> c = -1/2
print("\nOperator scan: c such that true TT b2(-D^2+cR) = +5R/3:")
c = sympy.Rational(-5, 6) - sympy.Rational(5, 3)
print(f"  c = {c / 5}  (i.e. operator -D^2 - R/2; not Bochner c=0, not "
      f"Lichnerowicz c=2/3, not code's documented E c=-1/6)")
b2_chalf = richardson_b2(lambda t: trace_TT(t, -0.5))
print(f"  numeric check -D^2 - R/2: b2 = {mp.nstr(b2_chalf, 8)} (should be +20)")

# Degeneracy comparison at low l: rep theory, no convention freedom
print("\nDegeneracies, TT vs 5x scalar(shifted-label candidates):")
for l in range(2, 7):
    print(f"  l={l}: TT={ms.multiplicity('TT', l):5d}   "
          f"5*scalar(l)={5 * ms.multiplicity('scalar', l):5d}   "
          f"5*scalar(l-2)={5 * ms.multiplicity('scalar', l - 2):5d}")
