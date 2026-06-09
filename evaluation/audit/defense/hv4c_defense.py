import sys
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
from asymsafety.frg.spectral import SpectralSumEvaluator
from asymsafety.geometry.decomposition import ModeSpectrum

ev = SpectralSumEvaluator(d=4, l_max=10)

# 1. Does any call to trace_on_S1xS3 crash?
for ft in ("scalar", "vector", "TT"):
    try:
        val = ev.trace_on_S1xS3(lambda z, w: 1.0/(1.0+z+w), ft, R3_val=6.0, beta_period=1.0, n_matsubara=1)
        print(f"{ft}: OK value={val}")
    except Exception as e:
        print(f"{ft}: RAISES {type(e).__name__}: {e}")

# 2. hasattr guard always True?
spec3 = ModeSpectrum(d=3)
print("hasattr(d=3 instance, '_scalar_mult_S4') =", hasattr(spec3, '_scalar_mult_S4'))

# 3. Fallback TT formula vs exact 2(l-1)(l+3)
print("l : code_int  code_float  exact")
for l in range(2, 7):
    code_i = (l - 1) * (l + 3) * (2 * l + 1) // 3
    code_f = (l - 1) * (l + 3) * (2 * l + 1) / 3
    exact = 2 * (l - 1) * (l + 3)
    print(f"{l} : {code_i}  {code_f:.4f}  {exact}")

# 4. Cross-check exact TT degeneracy via Weyl law: sum of degeneracies up to L
#    on a 3-manifold should grow ~ L^3 per polarization (2 polarizations) -> total ~ 2 * Vol-factor
import numpy as np
L = 2000
ls = np.arange(2, L)
exact_total = np.sum(2.0*(ls-1)*(ls+3))
scalar_total = np.sum((np.arange(0, L)+1.0)**2)
print("ratio TT/scalar mode counts (should -> 2, # polarizations):", exact_total/scalar_total)
code_total = np.sum((ls-1)*(ls+3)*(2*ls+1)/3.0)
print("ratio code-TT/scalar (degree-3 growth, diverges ~ L):", code_total/scalar_total)
