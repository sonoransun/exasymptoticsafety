"""D3b: global FP search in foliated EH — does the Manrique NGFP exist?"""
from asymsafety.beta.foliated import build_foliated_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

system = build_foliated_eh_beta_system(d=4)
finder = FixedPointFinder(system)
fps = finder.find_all_fixed_points(
    bounds={"g": (-0.5, 3.0), "lambda": (-0.45, 0.45), "lambda_ADM": (0.5, 1.5)},
    n_grid=7, n_random=300)
print(f"fixed points found: {len(fps)}")
for fp in fps:
    analyze_stability(system, fp)
    loc = {k: round(v, 6) for k, v in fp.location.items()}
    th = [f"{t:.4g}" for t in fp.critical_exponents]
    print(f"  {loc} | theta={th} | relevant={fp.relevant_directions} | gaussian={fp.is_gaussian}")
print("validation/manrique_2011 claims: g*~0.96, lambda*~0.20, lambda_ADM*=1.0")
