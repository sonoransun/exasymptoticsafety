import sympy
from sympy import Symbol, lambdify
from scipy.integrate import quad
from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.frg.regulator import LitimRegulator, TypeIIRegulator

tf_litim = ThresholdFunctions(LitimRegulator())
tf_t2 = ThresholdFunctions(TypeIIRegulator(LitimRegulator()))  # same R_k, symbolic branch

for (p, n, w) in [(1, 1, 0.0), (1, 2, 0.5), (2, 1, 0.5), (1, 3, 0.0)]:
    direct = float(tf_litim.Phi(p, n, sympy.Float(w)))
    sym = tf_t2.Phi(p, n, sympy.Float(w))
    zi = [s for s in sym.free_symbols if s.name == "z_int"][0]
    ki = [s for s in sym.free_symbols if s.name == "k_int"][0]
    f = lambdify(zi, sym.subs(ki, 1), "numpy")
    val, _ = quad(f, 0, 1)      # Heaviside support is z<k^2=1
    print(f"p={p} n={n} w={w}: Litim branch={direct:.10f}  symbolic branch (same R_k)={val:.10f}  ratio={val/direct:.6f}")
