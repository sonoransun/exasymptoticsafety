import sys
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
import numpy as np

# Self-contained check of the root algebra and both branches' physics.
def roots(N, eps=1.0, alpha=None):
    a = eps / N if alpha is None else alpha
    A, B, C = N + 4, eps + 6 * a, 9 * a**2
    disc = B**2 - 4 * A * C
    return a, (B - disc**0.5) / (2 * A), (B + disc**0.5) / (2 * A)

print("alpha->0 limit at N=30, eps=1 (which root connects to WF u=eps/(N+4)?):")
for al in [0.03, 0.003, 3e-4, 3e-6]:
    a, um, up = roots(30, alpha=al)
    print(f"  alpha={al:8.1e}: u-={um:.6e}  u+={up:.6e}   (WF=eps/(N+4)={1/34:.6e}, Gaussian=0)")

print("\nStability (theta = -dbeta/du at fixed alpha=alpha*) and nu for both branches:")
for N in (30, 40, 60, 100, 1000):
    eps = 1.0
    a, um, up = roots(N, eps)
    for tag, u in (("u-", um), ("u+", up)):
        dbdu = -eps + 2*(N+4)*u - 6*a          # d beta_u / du
        th_u = -dbdu                            # toolkit convention theta=-eig
        th_r = 2.0 - (N+2)*u + 6*a
        nrel = (th_u > 0) + (th_r > 0) + 0      # alpha direction: dbeta_a/da = -eps+2Na* = +eps>0 -> th_a=-eps<0 irrelevant
        nu = 1.0/th_r
        print(f"  N={N:5d} {tag}: u*={u:.6f} theta_u={th_u:+.4f} theta_r={th_r:.4f} "
              f"#relevant(u,r)={nrel} nu={nu:.4f}  [1-9.727/N={1-9.727/N:.4f}]")

print("\nDoes module docstring 'u* = eps/(N+4)(1+O(1/N))' match either branch?")
for N in (30, 100, 1000):
    a, um, up = roots(N)
    wf = 1.0/(N+4)
    print(f"  N={N:5d}: u-/wf = {um/wf:.4f}   u+/wf = {up/wf:.4f}  (docstring claims ratio -> 1)")
