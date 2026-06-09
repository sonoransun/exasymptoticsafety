"""HV-15d-b defense probe.

(1) Show compare_with_stability returns max_dev=0 / agrees=True for ANY
    StabilityAnalysis input -- even a fabricated garbage one -- proving the
    check is vacuous (cannot fail).
(2) Strongest defense test: the theoretical identity. Run the repo's own
    compute_edmd on EH trajectories near the NGFP and check whether
    log(eig K)/dt actually reproduces the stability eigenvalues mu(M).
    If yes, the *numbers* the method returns coincide with what a genuine
    EDMD comparison would have produced -- the defense that the substitution
    is 'exact by theorem', not just lazy.
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.linear.koopman import ClassicalKoopmanOperator

system = build_eh_beta_system()
fp = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14}, tol=1e-10)
assert fp is not None
stab = analyze_stability(system, fp)
mu = np.sort_complex(stab.eigenvalues)
print("mu(M) =", np.round(mu, 10))

ck = ClassicalKoopmanOperator(system, fp)

# (1) vacuity: garbage stability object still 'agrees'
class FakeStab:
    eigenvalues = np.array([1e6, -42.0, 3.14])  # nonsense, wrong dimension even
cmp_garbage = ck.compare_with_stability(FakeStab())
print("(1) garbage input -> max_dev =", cmp_garbage.max_deviation,
      " agrees =", cmp_garbage.agrees,
      " method_a =", cmp_garbage.method_a)

cmp_real = ck.compare_with_stability(stab)
print("    real input    -> max_dev =", cmp_real.max_deviation,
      " agrees =", cmp_real.agrees)
print("    values_a is values_b content:", np.array_equal(cmp_real.values_a, cmp_real.values_b))

# (2) the genuine comparison the method claims to perform.
# mu(M) has a strongly unstable eigenvalue, so use many SHORT two-snapshot
# trajectories (one EDMD pair each) with tiny displacements to stay linear.
integrator = FlowIntegrator(system)
dt = 0.01
trajs = []
rng = np.random.default_rng(1)
for _ in range(40):
    dg, dl = rng.normal(scale=1e-5, size=2)
    ic = {"g": fp.location["g"] + dg, "lambda": fp.location["lambda"] + dl}
    trajs.append(integrator.integrate(ic, t_span=(0, dt),
                                      t_eval=np.array([0.0, dt])))

for deg in (1, 2):
    res = ck.compute_edmd(trajs, dictionary_degree=deg)
    lam = res.eigenvalues
    # continuous-time exponents from discrete Koopman eigenvalues
    mu_K = np.log(lam[np.abs(lam) > 1e-12]) / dt
    # match leading n to mu(M)
    n = system.dimension
    # pick the mu_K closest to each true mu
    matched = []
    for m in mu:
        matched.append(mu_K[np.argmin(np.abs(mu_K - m))])
    matched = np.sort_complex(np.array(matched))
    dev = np.max(np.abs(matched - mu))
    print(f"(2) EDMD degree={deg}: dict_size={res.dictionary_size}, "
          f"matched log(eigK)/dt = {np.round(matched, 8)}, "
          f"max|.-mu(M)| = {dev:.3e}")
