"""HV-15d defense probe: does any supported truncation make tol=0.1 load-bearing?

The transfer path error scales ~ eps * exp(spread(mu)*dt)/dt. If stiff
truncations (quadratic, foliated) have |mu| spreads such that the expm
round-trip at the hardcoded dt=0.1 approaches 0.1, the tolerance is a
deliberate worst-case budget, not slack.
"""
import sys
import numpy as np

sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")

from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.transforms.bridge.cross_analogue import CrossAnalogueBridge

cases = []

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
cases.append(("eh", build_eh_beta_system(), {"g": 0.7, "lambda": 0.14}))

try:
    from asymsafety.beta.quadratic import build_quadratic_beta_system
    sys_q = build_quadratic_beta_system()
    # typical NGFP guess for quadratic gravity
    guess = {n: 0.2 for n in sys_q.coupling_names}
    for trial in ({"g": 0.7, "lambda": 0.2, "b": 0.01, "w": -0.005},
                  guess):
        try:
            cases.append(("quadratic", sys_q, trial))
            break
        except Exception:
            pass
except Exception as e:
    print("quadratic build failed:", e)

try:
    from asymsafety.beta.foliated import build_foliated_eh_beta_system
    sys_f = build_foliated_eh_beta_system()
    cases.append(("foliated", sys_f, {n: 0.3 for n in sys_f.coupling_names}))
except Exception as e:
    print("foliated build failed:", e)

for name, system, guess in cases:
    try:
        finder = FixedPointFinder(system)
        fp = finder.find_fixed_point(guess)
        if fp is None:
            print(f"{name}: no FP from guess {guess}")
            continue
        stab = analyze_stability(system, fp)
        mu = stab.eigenvalues
        spread = np.max(mu.real) - np.min(mu.real)
        br = CrossAnalogueBridge(system, fp, stab)
        res = br.verify_commutativity(tol=0.1)
        devs = {m: a["max_deviation"] for m, a in res["agreements"].items()}
        print(f"{name}: theta={np.round(stab.critical_exponents,4)}")
        print(f"   mu spread={spread:.2f}, spread*dt={spread*0.1:.2f}, "
              f"predicted expm err ~ {1e-16*np.exp(spread*0.1)/0.1:.2e}")
        print(f"   actual deviations: {devs}, all_agree={res['all_agree']}")
    except Exception as e:
        print(f"{name}: FAILED ({type(e).__name__}: {e})")
