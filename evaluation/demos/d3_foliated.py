"""D3: Foliated EH (ADM) NGFP — lambda_ADM* = 1 diffeo restoration."""
from asymsafety.beta.foliated import build_foliated_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

system = build_foliated_eh_beta_system(d=4)
fp = FixedPointFinder(system).find_fixed_point(
    {"g": 0.96, "lambda": 0.2, "lambda_ADM": 1.0})
print("location:", {k: f"{v:.10g}" for k, v in fp.location.items()})
stab = analyze_stability(system, fp)
print("critical exponents:", [f"{t:.6g}" for t in fp.critical_exponents])

lam_adm = fp.location["lambda_ADM"]
print(f"VERDICT lambda_ADM* == 1: {'PASS' if abs(lam_adm - 1.0) < 1e-8 else 'FAIL'} "
      f"(lambda_ADM* = {lam_adm:.12g})")
print(f"lambda* below pole 1/2: {'PASS' if fp.location['lambda'] < 0.5 else 'FAIL'}")
print("manrique_2011 reference: g*~0.96, lambda*~0.20, lambda_ADM*=1.0")
