"""D1: Einstein-Hilbert NGFP + stability analysis (core API demo)."""
import numpy as np
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability

system = build_eh_beta_system(d=4)
fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})
print("location:", {k: f"{v:.16g}" for k, v in fp.location.items()})

stab = analyze_stability(system, fp)
print(stab.summary())
print("critical_exponents:", [f"{t:.10g}" for t in fp.critical_exponents])

residual = max(abs(v) for v in system.evaluate(fp.location).values())
print(f"beta residual at FP: {residual:.3e}")

pin_g, pin_l = 0.6936584729648413, 0.14228896515894982
g, l = fp.location["g"], fp.location["lambda"]
ok = abs(g - pin_g) / pin_g < 1e-6 and abs(l - pin_l) / pin_l < 1e-6
print(f"VERDICT pin-match rtol 1e-6: {'PASS' if ok else 'FAIL'} (g*={g:.12g}, lambda*={l:.12g})")

# literature comparison (Litim-cutoff EH: g*~0.707, lambda*~0.193, theta~1.475+-3.043i)
print(f"vs literature: dg={abs(g-0.707)/0.707:.1%}, dl={abs(l-0.193)/0.193:.1%}, "
      f"product g*l*={g*l:.4f} vs 0.137 ({abs(g*l-0.137)/0.137:.1%})")
