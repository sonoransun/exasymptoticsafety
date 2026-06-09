import sys
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
import numpy as np
from asymsafety.transforms.bridge.gauge_higgs import GaugeHiggsAnalogue
for N in (30, 60):
    ga = GaugeHiggsAnalogue(N=N)
    th = ga.stability.critical_exponents
    print(f"N={N}: fp={ {k: round(v,6) for k,v in ga.fixed_point.location.items()} } "
          f"theta={np.round(th.real,4)} relevant={int((th.real>0).sum())} nu={ga.nu:.4f}")
