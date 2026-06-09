"""D8: Cross-analogue bridge — theta_i via RG / transfer-matrix / resolvent."""
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge

system = build_eh_beta_system(d=4)
fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
stab = analyze_stability(system, fp)

bridge = CrossAnalogueBridge(system, fp, stab)
result = bridge.verify_commutativity()
print("verify_commutativity ->")
if isinstance(result, dict):
    for k, v in result.items():
        print(f"  {k}: {v}")
else:
    print(" ", result)
