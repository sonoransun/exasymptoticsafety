import sys
import numpy as np
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge
from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.validation.codello_2009 import QUADRATIC_FP

system = build_quadratic_beta_system(d=4)
guess = {"g": QUADRATIC_FP["g_star"], "lambda": QUADRATIC_FP["lambda_star"],
         "alpha": QUADRATIC_FP["alpha_star"], "beta": QUADRATIC_FP["beta_star"]}
print("guess:", guess)
finder = FixedPointFinder(system)
fp = finder.find_fixed_point(guess)
if fp is None:
    print("no FP"); sys.exit()
print("FP:", {k: round(v, 6) for k, v in fp.location.items()})
stab = analyze_stability(system, fp)
mu = stab.eigenvalues
spread = np.max(mu.real) - np.min(mu.real)
br = CrossAnalogueBridge(system, fp, stab)
res = br.verify_commutativity(tol=0.1)
print("theta =", np.round(stab.critical_exponents, 4))
print(f"mu spread = {spread:.2f}, spread*dt = {spread*0.1:.2f}, "
      f"predicted expm round-trip err ~ {1e-16*np.exp(spread*0.1)/0.1:.2e}")
for m, a in res["agreements"].items():
    print(f"  {m}: dev = {a['max_deviation']:.3e}")
print("all_agree:", res["all_agree"])
