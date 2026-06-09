"""HV-12: matter dual-path consistency + matter signs vs DEP 1311.2898.

DEP reference (perturbative one-loop): beta_g = 2g + (g^2/6pi)(N_S + 2N_D
- 4N_V - 46) -> scalars destabilize (dg*/dN_S > 0), Dirac destabilize at
twice the weight, vectors stabilize (dg*/dN_V < 0).
"""
import numpy as np
import sympy
from sympy import Rational, Symbol, pi, simplify

from asymsafety.actions.matter import MatterContent, matter_eta_N_correction
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.beta.matter import (
    build_eh_matter_beta_system,
    build_gravity_matter_fp_system,
)

g = Symbol("g", positive=True)
lam = Symbol("lambda", real=True)

# ---------- (a) dual-path zero-matter comparison ----------
eh = build_eh_beta_system(d=4)
ehm = build_eh_matter_beta_system(MatterContent(), d=4)

for name in ("g", "lambda"):
    diff = simplify(eh.beta(name).expression - ehm.beta(name).expression)
    same = diff == 0
    print(f"beta_{name}: identical across paths? {same}")
    if not same:
        print(f"   eh    : {simplify(eh.beta(name).expression)}")
        print(f"   matter: {simplify(ehm.beta(name).expression)}")

fp_eh = FixedPointFinder(eh).find_fixed_point({"g": 0.7, "lambda": 0.14})
print(f"\nEH builder FP:        g*={fp_eh.location['g']:.6f} "
      f"lam*={fp_eh.location['lambda']:.6f} theta={fp_eh.critical_exponents}")
guesses = [{"g": 0.7, "lambda": 0.14}, {"g": 0.7, "lambda": 0.19},
           {"g": 1.5, "lambda": 0.1}, {"g": 0.3, "lambda": 0.3},
           {"g": 2.0, "lambda": -0.2}]
fp_m = None
for gu in guesses:
    fp_m = FixedPointFinder(ehm).find_fixed_point(gu)
    if fp_m is not None and fp_m.location.get("g", 0) > 1e-4:
        break
if fp_m is not None:
    print(f"matter builder (0 matter) FP: g*={fp_m.location['g']:.6f} "
          f"lam*={fp_m.location['lambda']:.6f} theta={fp_m.critical_exponents}")
else:
    print("matter builder (0 matter): NO nontrivial FP found from guesses")

# eta structure comparison
eta_eh = simplify(eh.beta("g").expression / g - 2)
eta_m = simplify(ehm.beta("g").expression / g - 2)
print("\neta_N identical?:", simplify(eta_eh - eta_m) == 0)
print("eta_eh:", eta_eh)
print("eta_m :", eta_m)

# ---------- (b) matter sign scans ----------
def gstar_scan(kind, builder):
    print(f"\n--- scan {kind} ({builder.__name__}) ---")
    rows = []
    guess = {"g": 0.7, "lambda": 0.14}
    for n in range(0, 7):
        kw = {kind: n}
        mc = MatterContent(**kw)
        if builder is build_gravity_matter_fp_system:
            syst = builder(mc, 4)
        else:
            syst = builder(mc, 4)
        fp = FixedPointFinder(syst).find_fixed_point(guess)
        if fp is None or fp.location.get("g", 0) <= 1e-6:
            rows.append((n, None, None))
            print(f"  N={n}: no FP")
            continue
        rows.append((n, fp.location["g"], fp.location["lambda"]))
        print(f"  N={n}: g*={fp.location['g']:.6f} lam*={fp.location['lambda']:.6f}")
        guess = {k: v for k, v in fp.location.items()}
    gs = [r[1] for r in rows if r[1] is not None]
    if len(gs) >= 2:
        trend = "INCREASING (destabilizing)" if gs[-1] > gs[0] else \
                "DECREASING (stabilizing)"
        print(f"  dg*/dN trend: {trend}  (g* {gs[0]:.4f} -> {gs[-1]:.4f})")
    return rows

for kind in ("n_scalars", "n_dirac", "n_vectors"):
    gstar_scan(kind, build_eh_matter_beta_system)
for kind in ("n_scalars", "n_dirac", "n_vectors"):
    gstar_scan(kind, build_gravity_matter_fp_system)

# ---------- (c) actions/matter.py A coefficients vs DEP ----------
print("\n--- matter_eta_N_correction per-field A coefficients ---")
for kind, dep in (("n_scalars", Rational(1, 6)),
                  ("n_dirac", Rational(1, 3)),
                  ("n_vectors", Rational(-2, 3))):
    A, B = matter_eta_N_correction(MatterContent(**{kind: 1}), 4)
    Apf = simplify(A * pi)  # report as multiple of 1/pi
    print(f"  {kind}: A = ({Apf})/pi ; DEP weight => A_DEP = ({dep})/pi ;"
          f" sign match: {(Apf * dep) > 0}")

# delta_N in build_gravity_matter_fp_system (read from source constants):
print("\nbuild_gravity_matter_fp_system delta_N per field (coeff of 1/(1-2l)):")
print("  scalar -1/6 (DEP wants +), dirac +1/3 (DEP +, OK sign),"
      " vector -1/3 (DEP -, OK sign)")

# ---------- (d) Korver bounds vs paper ----------
from asymsafety.validation.korver_2024 import (
    COVARIANT_MATTER_BOUNDS,
    FOLIATED_MATTER_BOUNDS,
)
print("\nFOLIATED_MATTER_BOUNDS:", FOLIATED_MATTER_BOUNDS)
print("paper 2402.01260 (notes): N_s + 6.4 N_v = 23.1 (p0) /"
      " N_s + 4.7 N_v = 22.4 (pvec)")
print("  => max N_s alone ~23 (toolkit claims 12); max N_v alone ~3-4"
      " (toolkit claims 6)")
print("COVARIANT_MATTER_BOUNDS:", COVARIANT_MATTER_BOUNDS)
print("DEP perturbative bound: N_D < 23 + 2N_V - N_S/2; N_s alone < ~31;"
      " toolkit max_N_D=10, max_N_s_minimal=22")
