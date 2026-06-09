"""HV-2b defense check: can any convention reconcile the two branches of Phi()?"""
import sympy
from sympy import Symbol, simplify, integrate, oo, gamma, Heaviside, nsimplify

from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.frg.regulator import LitimRegulator, ExponentialRegulator, TypeIIRegulator

z = Symbol("z", positive=True)
k = Symbol("k", positive=True)

# 1) Identity check: dtR_k == 2*(R_k - z dR_k/dz) for both regulators (scaling form).
for reg in (LitimRegulator(), ExponentialRegulator()):
    R = reg.R_k(z, k)
    dt = reg.dR_k_dt(z, k)
    reuter_num = R - z * sympy.diff(R, z)            # module-docstring numerator
    diff = simplify(dt - 2 * reuter_num)
    # Heaviside derivative terms: (k^2-z)*DiracDelta(k^2-z) == 0
    diff = diff.replace(sympy.DiracDelta, lambda a: 0) if diff.has(sympy.DiracDelta) else diff
    print(f"{reg.name:12s}: dtR_k - 2*[R_k - z R_k'] = {simplify(diff)}")

# 2) Same regulator FUNCTION, two branches: Litim direct vs Litim-wrapped-in-TypeII.
tf_litim = ThresholdFunctions(LitimRegulator())
tf_t2 = ThresholdFunctions(TypeIIRegulator(LitimRegulator()))   # not isinstance LitimRegulator
p, n = 1, 2
w = sympy.Symbol("w", positive=True)
direct = tf_litim.Phi(p, n, w)
sym = tf_t2.Phi(p, n, w)                                        # bare integrand
zi = [s for s in sym.free_symbols if s.name == "z_int"][0]
ki = [s for s in sym.free_symbols if s.name == "k_int"][0]
# integrate the symbolic-branch integrand exactly (Litim => piecewise, denominator k^2(1+w) on z<k^2)
sym_k1 = sym.subs(ki, 1)
val = integrate(sym_k1, (zi, 0, oo))
print(f"\nLitim branch Phi^{p}_{n}(w)            = {direct}")
print(f"Symbolic branch (same Litim R_k) int = {simplify(val)}")
print(f"ratio symbolic/direct                = {simplify(val/direct)}")

# 3) QFunctional internal cross-check: the codebase's own dtR_k-numerator object carries the 2.
from asymsafety.frg.threshold import QFunctional
q = QFunctional(LitimRegulator())
m2 = sympy.Symbol("m2", positive=True)
qe = q.evaluate(n, p, m2, k)
print(f"\nQFunctional Q_{n}[dtR/(P+m^2)^{p}]      = {qe}")
print(f" => Q/(k^(2(n-p+1)) Phi(w=m2/k^2))   = "
      f"{simplify(qe / (k**(2*(n-p+1)) * direct.subs(w, m2/k**2)))}")
