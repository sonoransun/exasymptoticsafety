"""HV-1: Litim threshold closed forms vs definitional quadrature + identities + QFunctional."""
import mpmath as mp
import sympy
from sympy import Symbol, gamma, Rational, simplify, diff

from asymsafety.frg.threshold import ThresholdFunctions, QFunctional
from asymsafety.frg.regulator import LitimRegulator

mp.mp.dps = 40

# Definitional quadrature (Reuter convention, dimensionless y = z/k^2):
#   Phi^p_n(w)      = (1/G(n)) int_0^inf y^{n-1} [R0 - y R0'] / (y + R0 + w)^p
#   Phi_tilde^p_n(w)= (1/G(n)) int_0^inf y^{n-1} R0 / (y + R0 + w)^p
# Litim: R0(y) = (1-y) theta(1-y); on (0,1): R0 - yR0' = 1, y + R0 = 1; zero beyond.
def phi_quad(p, n, w):
    f = lambda y: y**(n - 1) * mp.mpf(1) / (1 + w)**p
    return mp.quad(f, [0, 1]) / mp.gamma(n)

def phit_quad(p, n, w):
    f = lambda y: y**(n - 1) * (1 - y) / (1 + w)**p
    return mp.quad(f, [0, 1]) / mp.gamma(n)

tf = ThresholdFunctions(LitimRegulator())

print("=== Litim Phi/Phi_tilde: quadrature vs evaluate_numerical vs symbolic ===")
worst = 0.0
for (p, n) in [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]:
    for w in [0.0, -0.3, 0.5]:
        q = float(phi_quad(p, n, mp.mpf(w)))
        num = tf.evaluate_numerical("Phi", p, n, w)
        symv = float(tf.Phi(p, n, Rational(w).limit_denominator(100)))
        qt = float(phit_quad(p, n, mp.mpf(w)))
        numt = tf.evaluate_numerical("Phi_tilde", p, n, w)
        symt = float(tf.Phi_tilde(p, n, Rational(w).limit_denominator(100)))
        e1 = max(abs(num - q), abs(symv - q)) / abs(q)
        e2 = max(abs(numt - qt), abs(symt - qt)) / abs(qt)
        worst = max(worst, e1, e2)
        print(f"p={p} n={n} w={w:+.1f}: Phi quad={q:.12f} code_num={num:.12f} "
              f"code_sym={symv:.12f} | Phit quad={qt:.12f} code={numt:.12f},{symt:.12f} "
              f"rel.err {e1:.1e}/{e2:.1e}")
print(f"worst relative error vs quadrature: {worst:.2e}  (PASS rtol 1e-10: {worst < 1e-10})")

print("\n=== Identities (symbolic, Litim) ===")
ws = Symbol("w", positive=True)
for (p, n) in [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]:
    d_id = simplify(diff(tf.Phi(p, n, ws), ws) + p * tf.Phi(p + 1, n, ws))
    r_id = simplify(tf.Phi_tilde(p, n, ws) * (n + 1) - tf.Phi(p, n, ws))
    print(f"p={p} n={n}: d/dw Phi + p Phi^(p+1) = {d_id} ; (n+1)Phit - Phi = {r_id}")

print("\n=== n=0 convention: Phi^p_0(w) = (1+w)^-p ===")
for p in (1, 2):
    print(f"p={p}: code Phi(p,0,w) = {simplify(tf.Phi(p, 0, ws))}  expected (1+w)^-{p}")

print("\n=== QFunctional normalization (factor 2) ===")
q = QFunctional(LitimRegulator())
k = Symbol("k", positive=True)
msq = Symbol("m_sq", positive=True)
for (n, p) in [(2, 1), (1, 1), (2, 2), (1, 2)]:
    code = q.evaluate(n, p, msq, k)
    # ground truth: Q_n[dt R_k/(P_k+m^2)^p] = 2 k^{2(n-p+1)} Phi^p_n(m^2/k^2)  (Litim)
    expected = 2 * k**(2 * (n - p + 1)) / (gamma(n + 1) * (1 + msq / k**2)**p)
    print(f"n={n} p={p}: code - 2k^(2(n-p+1))Phi = {simplify(code - expected)}")

# brute-force mpmath check of Q at k=1.3, m^2=0.4 (Litim: dtR = 2k^2 on z<k^2, P_k = k^2)
kv, m2 = mp.mpf("1.3"), mp.mpf("0.4")
for (n, p) in [(2, 1), (2, 2)]:
    brute = mp.quad(lambda z: z**(n - 1) * 2 * kv**2 / (kv**2 + m2)**p, [0, kv**2]) / mp.gamma(n)
    code = float(q.evaluate(n, p, float(m2), Symbol("k")).subs(Symbol("k"), Rational(13, 10)))
    print(f"n={n} p={p}: brute Q={float(brute):.12f} code Q={code:.12f} "
          f"rel {abs(code-float(brute))/float(brute):.1e}")

print("\n=== QFunctional/2*Phi consumption in traces.py ===")
import inspect
import asymsafety.frg.traces as tr
src = inspect.getsource(tr)
print("uses 'q_func.evaluate':", "q_func.evaluate" in src)
print("uses explicit '2 * phi':", src.count("* 2 * phi"), "occurrences (factor 2 = Q_n/Phi ratio)")
