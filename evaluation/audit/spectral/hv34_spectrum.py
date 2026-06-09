"""HV-3/HV-4: S^4 degeneracies, eigenvalues, York exclusions, S1xS3 path."""
import sympy
from sympy import Symbol, Rational, Poly

from asymsafety.geometry.decomposition import ModeSpectrum, YorkDecomposition
from asymsafety.frg.spectral import SpectralSumEvaluator

# Ground truth (Rubin-Ordonez d=4, unit S^4):
t_scalar = lambda l: (2*l+3)*(l+1)*(l+2)//6
t_vector = lambda l: l*(l+3)*(2*l+3)//2 if l >= 1 else 0
t_TT     = lambda l: 5*(l-1)*(l+4)*(2*l+3)//6 if l >= 2 else 0

ms = ModeSpectrum(d=4)
print("=== HV-3/4: S^4 multiplicities, code vs ground truth (l=0..6) ===")
print(f"{'l':>2} | {'sc code':>8} {'sc true':>8} | {'vec code':>8} {'vec true':>8} | {'TT code':>8} {'TT true':>8}")
for l in range(7):
    print(f"{l:>2} | {ms.multiplicity('scalar', l):>8} {t_scalar(l):>8} | "
          f"{ms.multiplicity('vector', l):>8} {t_vector(l):>8} | "
          f"{ms.multiplicity('TT', l):>8} {t_TT(l):>8}")

print("\nKilling-vector anchor: dim SO(5) = 10; code vector mult at l=1 =",
      ms.multiplicity('vector', 1))
print("Scalar l=1 anchor: 5 Cartesian coords of S^4 in R^5; code =", ms.multiplicity('scalar', 1))

print("\n=== Weyl-law degree argument (degeneracy on a 4-manifold must grow as l^{d-1}=l^3) ===")
lsym = Symbol("l")
code_scalar = (lsym+1)**2 * (lsym+2)**2 / 4
true_scalar = (2*lsym+3)*(lsym+1)*(lsym+2)/6
print("degree of code scalar formula (l+1)^2(l+2)^2/4 :", Poly(code_scalar, lsym).degree(), " (l^4 -> Weyl-law violation)")
print("degree of true scalar formula                  :", Poly(true_scalar, lsym).degree())
code_vec = (2*lsym+3)*(lsym+2)*(lsym+1)/2
true_vec = lsym*(lsym+3)*(2*lsym+3)/2
print("code vector - true vector =", sympy.expand(code_vec - true_vec), " (= 2l+3 per level; code = 3 x scalar deg)")
print("code vector / true scalar deg =", sympy.simplify(code_vec / true_scalar))

print("\n=== Eigenvalues and a^2 handling ===")
a_sq = Symbol("a2", positive=True)
R = Symbol("R_bar", positive=True)
for ft, shift in [("scalar", 0), ("vector", 1), ("TT", 2)]:
    ev = ms.eigenvalue(ft, 3, a_sq)
    expected = (3*(3+3) - shift)/a_sq
    print(f"{ft}: eigenvalue(l=3) = {ev}  expected (l(l+3)-{shift})/a^2 -> diff = {sympy.simplify(ev-expected)}")
# R_bar = 12/a^2 in d=4: SpectralSumEvaluator uses a_sq = d(d-1)/R_bar
ss = SpectralSumEvaluator(d=4, l_max=8)
print("evaluator a_sq for R_bar=12:", 4*3/12.0, "(unit sphere) -> scalar l=2 eigenvalue:",
      float(ms.eigenvalue('scalar', 2, 4*3/12.0)), "expected 10")

print("\n=== min_l and York excluded modes ===")
print("min_l:", {ft: ms.min_l(ft) for ft in ("scalar", "vector", "TT")})
yd = YorkDecomposition(d=4)
print("dof: TT", yd.dof_TT, "vector", yd.dof_vector, "scalar", yd.dof_scalar, "total", yd.total_dof)
for ft in ("TT", "vector", "scalar_sigma", "scalar_h"):
    print(f"excluded_modes({ft!r}) = {yd.excluded_modes(ft)}")
print("GROUND TRUTH: xi l=1 (10 Killing vectors) must be excluded; code dict 'vector': [0] omits l=1")
print("              (its own docstring says 'l = 1 are Killing vectors (excluded)') -> contradiction")
print("sigma exclusions [0,1] match ground truth (l=0 const, l=1 conformal KVs)")

print("\n=== trace_on_sphere: numerical impact, W(z)=1/(1+z), R_bar=12 ===")
import math
def truth_trace(W, ft, lmax):
    deg = {"scalar": t_scalar, "vector": t_vector, "TT": t_TT}[ft]
    sh = {"scalar": 0, "vector": 1, "TT": 2}[ft]
    lmin = {"scalar": 0, "vector": 1, "TT": 2}[ft]
    return sum(deg(l)*W(l*(l+3)-sh) for l in range(lmin, lmax+1))
W = lambda z: 1.0/(1.0+z)
for lmax in (128, 256, 512):
    ss = SpectralSumEvaluator(d=4, l_max=lmax)
    for ft in ("scalar", "vector", "TT"):
        c = ss.trace_on_sphere(W, ft, 12.0)
        t = truth_trace(W, ft, lmax)
        print(f"l_max={lmax:>4} {ft:>7}: code={c:>14.3f} truth={t:>14.3f} ratio={c/t:.4f}")
print("NOTE: scalar code sum grows ~ l_max^2 (sum l^4/l^2) -> divergent, truth converges ~ log/linear")

print("\n=== trace_on_S1xS3 (foliated path) ===")
try:
    val = ss.trace_on_S1xS3(lambda z, om: 1.0/(1.0+z+om), "scalar", 6.0, 2*math.pi, n_matsubara=3)
    print("returned:", val)
except Exception as e:
    print(f"RAISES {type(e).__name__}: {e}")
print("cause: line 103 guard hasattr(spec_S3,'_scalar_mult_S4') =",
      hasattr(ModeSpectrum(d=3), "_scalar_mult_S4"),
      "-> calls ModeSpectrum(d=3).multiplicity which raises for d!=4")

print("\n=== S^3 multiplicities (fallback + vectorized) vs ground truth ===")
# truth: scalar (l+1)^2 ; transverse vector 2l(l+2) ; TT 2(l-1)(l+3)  [Rubin-Ordonez d=3]
from asymsafety.compute.batch.spectral import _S3_multiplicity_vectorized
import numpy as np
ls = np.arange(0, 7)
for ft, truth in [("scalar", lambda l: (l+1)**2),
                  ("vector", lambda l: 2*l*(l+2) if l >= 1 else 0),
                  ("TT", lambda l: 2*(l-1)*(l+3) if l >= 2 else 0)]:
    code_fb = [SpectralSumEvaluator._S3_multiplicity(ft, int(l)) for l in ls]
    code_vec = _S3_multiplicity_vectorized(ft, ls.astype(float))
    print(f"{ft:>7}: fallback={code_fb} vectorized={list(code_vec)} truth={[truth(int(l)) for l in ls]}")
