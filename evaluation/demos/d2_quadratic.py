"""D2: Quadratic gravity (R^2 + C^2) NGFP + one-loop structure."""
from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

system = build_quadratic_beta_system(d=4)
print("couplings:", system.coupling_names)
fp = FixedPointFinder(system).find_fixed_point(
    {"g": 0.97, "lambda": 0.14, "alpha": 0.006, "beta": 0.002})
print("location:", {k: f"{v:.10g}" for k, v in fp.location.items()})
stab = analyze_stability(system, fp)
print("critical exponents:", [f"{t:.6g}" for t in fp.critical_exponents])
print("relevant directions:", fp.relevant_directions)

# one-loop universal limits at origin
origin = system.evaluate({"g": 0.0, "lambda": 0.0, "alpha": 0.0, "beta": 0.0})
import math
print(f"beta_alpha(0) = {origin['alpha']:.10g}  ({origin['alpha']*16*math.pi**2:.6f} /16pi^2)")
print(f"beta_beta(0)  = {origin['beta']:.10g}  ({origin['beta']*16*math.pi**2:.6f} /16pi^2)")
print(f"validation/codello_2009 claims: 53/45={53/45:.6f}, -196/45={-196/45:.6f}")
print(f"C^2 asymptotic freedom (beta_beta<0 at origin): {'PASS' if origin['beta'] < 0 else 'FAIL'}")
