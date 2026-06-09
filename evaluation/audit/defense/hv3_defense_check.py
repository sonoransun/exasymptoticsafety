"""HV-3 defense check: hunt for ANY convention under which (l+1)^2(l+2)^2/4
is a legitimate S^4 scalar Laplacian degeneracy."""
import numpy as np
from math import comb

code = lambda l: (l+1)**2 * (l+2)**2 // 4
exact = lambda l: (2*l+3)*(l+1)*(l+2)//6   # Rubin-Ordonez D_l

print("l : code  exact  cumulative-exact  dim(l,l)SO5  [(l+1)(l+2)/2]^2")
for l in range(7):
    cum = sum(exact(m) for m in range(l+1))
    # SO(5) irrep dim with Dynkin labels (a,b): (a+1)(b+1)(a+b+2)(2a+b+3)/6
    dim_ll = (l+1)*(l+1)*(2*l+2)*(3*l+3)//6
    print(l, ":", code(l), exact(l), cum, dim_ll, ((l+1)*(l+2)//2)**2)

# Defense A: mode relabeling l -> l+s for any integer shift s in -3..3
print("\nrelabel test: does code(l) == exact(l+s) for any fixed s?")
for s in range(-3, 4):
    ok = all(code(l) == exact(l+s) for l in range(max(0,-s), 10))
    print(f"  s={s}: {ok}")

# Defense B: cumulative mode count N(l) = sum_{m<=l} D_m (Weyl counting fn)
print("\ncumulative test: code(l) == sum_{m<=l} exact(m)?",
      all(code(l) == sum(exact(m) for m in range(l+1)) for l in range(10)))
# known closed form: N(l) = C(l+4,4) + C(l+3,4)
print("closed-form cumulative:", [comb(l+4,4)+comb(l+3,4) for l in range(5)])

# Defense C: ANY SO(5) irrep (a,b) with dim == code(l) for each l? scan
def dim_so5(a, b):
    return (a+1)*(b+1)*(a+b+2)*(2*a+b+3)//6
print("\nSO(5) irrep scan for code values:")
for l in range(1, 6):
    hits = [(a,b) for a in range(60) for b in range(60) if dim_so5(a,b) == code(l)]
    print(f"  l={l}, code={code(l)}: irreps {hits}")

# Defense D: overcount factor in trace_on_sphere, W=1/(1+z), R=12 (a^2=1)
for lmax in (128, 512):
    ls = np.arange(0, lmax+1)
    lam = ls*(ls+3.0)  # a^2 = d(d-1)/R = 1
    W = 1.0/(1.0+lam)
    t_code = np.sum(np.array([code(l) for l in ls])*W)
    t_true = np.sum(np.array([exact(l) for l in ls])*W)
    print(f"\nl_max={lmax}: trace code={t_code:.3f} exact={t_true:.3f} ratio={t_code/t_true:.1f}")

# Defense E: heat-kernel small-t scaling -> effective dimension
for t in (1e-2, 1e-3, 1e-4):
    ls = np.arange(0, 4000)
    lam = ls*(ls+3.0)
    K_code = np.sum(np.array([code(l) for l in ls])*np.exp(-t*lam))
    K_true = np.sum(np.array([exact(l) for l in ls])*np.exp(-t*lam))
    # K ~ t^{-d_eff/2}
    print(f"t={t}: K_code*t^2.5={K_code*t**2.5:.4f}  K_true*t^2={K_true*t**2:.4f}")
