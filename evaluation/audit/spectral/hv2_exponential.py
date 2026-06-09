"""HV-2: exponential-regulator numerical threshold functions vs ground truth."""
import mpmath as mp
import numpy as np
import sympy
from sympy import Symbol, lambdify

from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.frg.regulator import ExponentialRegulator

mp.mp.dps = 40

R0 = lambda y: y / mp.expm1(y)                      # shape R0(y) = y/(e^y-1)
NUM = lambda y: y**2 * mp.exp(y) / mp.expm1(y)**2   # R0 - y R0' (Reuter numerator)
P = lambda y: y + R0(y)

def phi(p, n, w):
    f = lambda y: y**(n - 1) * NUM(y) / (P(y) + w)**p
    return mp.quad(f, [mp.mpf("1e-12"), 1, 10, 60])

def phit(p, n, w):
    f = lambda y: y**(n - 1) * R0(y) / (P(y) + w)**p
    return mp.quad(f, [mp.mpf("1e-12"), 1, 10, 60])

def Phi(p, n, w): return phi(p, n, w) / mp.gamma(n)
def Phit(p, n, w): return phit(p, n, w) / mp.gamma(n)

print("=== Ground-truth anchors (exponential regulator, Reuter convention) ===")
print(f"Phi^1_1(0)   = {mp.nstr(Phi(1,1,0), 13)}   (exact pi^2/6 = {mp.nstr(mp.pi**2/6,13)})")
print(f"Phi^2_1(0)   = {mp.nstr(Phi(2,1,0), 13)}   (exact 1)")
print(f"Phit^1_1(0)  = {mp.nstr(Phit(1,1,0),13)}   (exact 1)")
print(f"Phi^1_2(0)   = {mp.nstr(Phi(1,2,0), 13)}   (exact 2*zeta(3) = {mp.nstr(2*mp.zeta(3),13)})")
print(f"Phi^1_1(1/2) = {mp.nstr(Phi(1,1,mp.mpf('0.5')),13)}   (notes 1.2713210370208803)")
print(f"Phi^2_1(1/2) = {mp.nstr(Phi(2,1,mp.mpf('0.5')),13)}   (notes 0.56195881707831066)")
print(f"Phit^1_1(1/2)= {mp.nstr(Phit(1,1,mp.mpf('0.5')),13)}   (notes 0.74722605965469218)")

tf = ThresholdFunctions(ExponentialRegulator())

print("\n=== Code evaluate_numerical('Phi') vs ground truth ===")
print("hypothesis: code numerator 2z(r - y r') = 2R_k + dt(R_k) => code = 2*Phi + 2*Phit")
cases = [(1, 1, 0.0), (1, 1, 0.5), (1, 1, -0.3), (2, 1, 0.0), (2, 1, 0.5),
         (1, 2, 0.0), (2, 2, 0.5), (1, 3, 0.0)]
for (p, n, w) in cases:
    code = tf.evaluate_numerical("Phi", p, n, w)
    truth = float(Phi(p, n, mp.mpf(w)))
    pred = float(2 * Phi(p, n, mp.mpf(w)) + 2 * Phit(p, n, mp.mpf(w)))
    print(f"Phi p={p} n={n} w={w:+.1f}: code={code:.10f} truth={truth:.10f} "
          f"ratio={code/truth:.4f}  2Phi+2Phit={pred:.10f} (code/pred={code/pred:.6f})")

print("\n=== Code evaluate_numerical('Phi_tilde') vs ground truth ===")
for (p, n, w) in cases:
    code = tf.evaluate_numerical("Phi_tilde", p, n, w)
    truth = float(Phit(p, n, mp.mpf(w)))
    print(f"Phit p={p} n={n} w={w:+.1f}: code={code:.10f} truth={truth:.10f} "
          f"rel.err={abs(code-truth)/abs(truth):.2e}")

print("\n=== Symbolic path (_phi_symbolic uses regulator.dR_k_dt = full dt R_k) ===")
# regulator.dR_k_dt symbolic check: should equal 2 z y e^y/(e^y-1)^2 = 2*NUM*k^2... verify
reg = ExponentialRegulator()
z, k = Symbol("z", positive=True), Symbol("k", positive=True)
dRdt = reg.dR_k_dt(z, k)
y = Symbol("y", positive=True)
expected = 2 * k**2 * y**2 * sympy.exp(y) / (sympy.exp(y) - 1)**2  # 2k^2 [R0 - yR0']
print("dR_k_dt - 2k^2(R0-yR0') =",
      sympy.simplify(dRdt.subs(z, y * k**2) - expected))
# integrate the symbolic integrand numerically at k=1: expect 2*Phi (factor 2 vs Litim branch)
for (p, n, w) in [(1, 1, 0.0), (2, 1, 0.5)]:
    integrand = tf._phi_symbolic(p, n, sympy.Float(w))
    zz = [s for s in integrand.free_symbols if s.name == "z_int"][0]
    kk = [s for s in integrand.free_symbols if s.name == "k_int"][0]
    f = lambdify(zz, integrand.subs(kk, 1), "mpmath")
    val = mp.quad(f, [mp.mpf("1e-12"), 1, 10, 60])
    print(f"symbolic-integrand integral p={p} n={n} w={w}: {mp.nstr(val,10)} "
          f"= {mp.nstr(val/Phi(p,n,mp.mpf(w)),6)} x Phi (Litim branch returns 1 x Phi)")
