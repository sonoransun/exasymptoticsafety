"""D9b: numpy vs jax batch-evaluator consistency on an EH grid."""
import numpy as np
from asymsafety.beta.einstein_hilbert import build_eh_beta_system

system = build_eh_beta_system(d=4)
g = np.linspace(0.01, 1.5, 40)
lam = np.linspace(-0.4, 0.4, 40)
G, L = np.meshgrid(g, lam, indexing="ij")
points = np.column_stack([G.ravel(), L.ravel()])

ev_np = system.batch_evaluator("numpy")
ev_jax = system.batch_evaluator("jax")
out_np = np.asarray(ev_np.evaluate_batch(points))
out_jax = np.asarray(ev_jax.evaluate_batch(points))
print("numpy out:", out_np.shape, out_np.dtype, "| jax out:", out_jax.shape, out_jax.dtype)

rel = np.abs(out_np - out_jax) / (np.abs(out_np) + 1e-300)
finite = np.isfinite(out_np) & np.isfinite(out_jax)
print(f"max relative deviation (finite entries): {rel[finite].max():.3e}")
print(f"non-finite entries numpy/jax: {np.sum(~np.isfinite(out_np))}/{np.sum(~np.isfinite(out_jax))}")
tol = 1e-6  # jax may run float32 by default
verdict = "PASS" if rel[finite].max() < tol else "FAIL"
print(f"VERDICT (rtol {tol}): {verdict}")
