"""HV-12c defense: does the toolkit's OWN matter system lose its NGFP near N_s=12 / N_v=6?
If yes, the dict values could be self-derived (still misattributed). If no, they are unsourced."""
from asymsafety.actions.matter import MatterContent
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.beta.matter import build_eh_matter_beta_system, build_gravity_matter_fp_system

def scan(builder, kind, nmax):
    print(f"--- {builder.__name__}, varying {kind} ---")
    guess = {"g": 0.7, "lambda": 0.14}
    last_ok = None
    for n in range(0, nmax + 1):
        mc = MatterContent(**{kind: n})
        syst = builder(mc, 4)
        fp = None
        for gu in (guess, {"g": 0.7, "lambda": 0.14}, {"g": 1.2, "lambda": 0.05},
                   {"g": 0.3, "lambda": 0.25}, {"g": 2.0, "lambda": -0.2}):
            fp = FixedPointFinder(syst).find_fixed_point(gu)
            if fp is not None and fp.location.get("g", 0) > 1e-4:
                break
            fp = None
        if fp is None:
            print(f"  N={n}: NO NGFP")
        else:
            loc = fp.location
            print(f"  N={n}: g*={loc['g']:.4f} lam*={loc['lambda']:.4f}")
            guess = dict(loc)
            last_ok = n
    print(f"  => last N with NGFP: {last_ok}")

for b in (build_eh_matter_beta_system, build_gravity_matter_fp_system):
    scan(b, "n_scalars", 30)
    scan(b, "n_vectors", 10)
