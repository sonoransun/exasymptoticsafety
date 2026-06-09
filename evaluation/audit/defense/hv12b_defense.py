"""HV-12b defense: matter trends end-to-end, both builders, vs DEP."""
import numpy as np
from scipy.optimize import fsolve
from asymsafety.actions.matter import MatterContent, matter_eta_N_correction
from asymsafety.beta.matter import build_eh_matter_beta_system, build_gravity_matter_fp_system
import sympy

print("=== matter_eta_N_correction per-field A (d=4) ===")
for kw, name in [({"n_scalars":1},"scalar"),({"n_dirac":1},"dirac"),({"n_vectors":1},"vector")]:
    A,B = matter_eta_N_correction(MatterContent(**kw))
    print(f"  {name}: A = {sympy.nsimplify(A)} = {float(A):+.6f}")
print(f"  DEP eq(38): scalar +1/(6pi)={1/(6*np.pi):+.6f}, dirac +1/(3pi)={1/(3*np.pi):+.6f}, vector -2/(3pi)={-2/(3*np.pi):+.6f}")

def scan(builder, label):
    print(f"\n=== {label} ===")
    for fname in ["n_scalars","n_dirac","n_vectors"]:
        x0 = np.array([0.65, 0.14])
        row = []
        for n in range(0,5):
            sys_ = builder(MatterContent(**{fname:n}))
            f = sys_.rhs_vector()
            sol, info, ier, msg = fsolve(lambda y: f(0.0, y), x0, full_output=True)
            if ier == 1 and sol[0] > 1e-4:
                row.append((n, sol[0], sol[1]))
                x0 = sol
            else:
                row.append((n, None, None))
        print(f"  {fname}: " + ", ".join(
            f"N={n}:(g*={g:.4f},l*={l:.4f})" if g is not None else f"N={n}:none"
            for n,g,l in row))
        gs = [g for _,g,_ in row if g is not None]
        if len(gs) >= 2:
            print(f"    -> g* trend with N: {'INCREASING (destabilizing, DEP-like)' if gs[-1]>gs[0] else 'DECREASING (stabilizing, anti-DEP)'}")

scan(lambda m: build_eh_matter_beta_system(m), "build_eh_matter_beta_system")
scan(lambda m: build_gravity_matter_fp_system(m), "build_gravity_matter_fp_system")

print("\n=== DEP perturbative one-loop g* = 12pi/(46 - N_S - 2N_D + 4N_V) ===")
print("  N_S=0..4:", ", ".join(f"{12*np.pi/(46-ns):.4f}" for ns in range(5)), "(increasing)")
