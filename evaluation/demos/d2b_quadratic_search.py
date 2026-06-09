"""D2b: global FP search in quadratic gravity — does the claimed NGFP exist?"""
from asymsafety.beta.quadratic import build_quadratic_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

system = build_quadratic_beta_system(d=4)
finder = FixedPointFinder(system)
fps = finder.find_all_fixed_points(
    bounds={"g": (-0.5, 2.0), "lambda": (-0.45, 0.45),
            "alpha": (-0.1, 0.1), "beta": (-0.1, 0.1)},
    n_grid=5, n_random=200)
print(f"fixed points found: {len(fps)}")
for fp in fps:
    analyze_stability(system, fp)
    loc = {k: round(v, 6) for k, v in fp.location.items()}
    th = [f"{t:.4g}" for t in fp.critical_exponents]
    print(f"  {loc} | theta={th} | relevant={fp.relevant_directions} | gaussian={fp.is_gaussian}")
print("validation/codello_2009 claims: g*~0.97, lambda*~0.14, alpha*~0.006, beta*~0.002, n_relevant=4")
