"""HV-14f defense: is the crossing check a disclosed manifest-symmetry
certification (standard bootstrap practice) rather than a misleading claim?"""
import sys
import numpy as np
sys.path.insert(0, "/root/cdev/exasymptoticsafety/tests")
from conftest import make_as_trajectory

from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.form_factor import GravitonFormFactor
from asymsafety.scattering import consistency as C
from asymsafety.scattering.bootstrap import StringAmplitude

traj = make_as_trajectory()
ff = GravitonFormFactor(traj)
amp = GravitonMediatedAmplitude(ff)

# (1) in-toolkit: bit-identical (auditor's fact)
s, t = 5.0, -2.0
u = -s - t
d1, d2 = complex(amp.eval(s, t, u)), complex(amp.eval(t, s, u))
print("[1] toolkit AS amp: eval(s,t,u)==eval(t,s,u) bitwise:", d1 == d2,
      " residual:", abs(d1 - d2))
# include_xi variant also symmetric (xi term is per-channel)
amp_xi = GravitonMediatedAmplitude(ff, include_xi=True)
d1x, d2x = complex(amp_xi.eval(s, t, u)), complex(amp_xi.eval(t, s, u))
print("    include_xi=True bitwise:", d1x == d2x)

# (2) the SAME is true of the string side certified in bridge.py:
vs = StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
sv1, sv2 = vs.eval(2.1, -1.3, -2.8), vs.eval(-1.3, 2.1, -2.8)
print("[2] Virasoro-Shapiro: eval(s,t,u)==eval(t,s,u) bitwise:", sv1 == sv2,
      " residual:", abs(sv1 - sv2))
ven = StringAmplitude(kind="veneziano", alphap=0.25)
v1, v2 = ven.eval(2.1, -1.3, -2.8), ven.eval(-1.3, 2.1, -2.8)
print("    Veneziano bitwise:", v1 == v2, " residual:", abs(v1 - v2))

# (3) crossing() DOES discriminate over the API: a global k^2=s
# identification (the crossing-VIOLATING scheme found in parts of the
# RG-improvement literature) fails the same check.
class GlobalSDressing:
    """RG-improve with k^2 = s globally (not per channel)."""
    def __init__(self, inner):
        self.inner = inner
        self.form_factor = inner.form_factor
    def eval(self, s_, t_, u_, dressed=True):
        s_a = np.asarray(s_, dtype=float)
        GN = self.form_factor.newton_constant()
        Gs = np.asarray(self.form_factor.G_of_psq(s_a), dtype=float)
        ratio = (Gs / GN) if dressed else 1.0
        return ratio * self.inner.eval(s_, t_, u_, dressed=False)

bad = GlobalSDressing(amp)
out_bad = C.crossing(bad, s=5.0, t=-2.0)
out_good = C.crossing(amp, s=5.0, t=-2.0)
print("[3] crossing() on global-s scheme:", out_bad["passed"],
      " rel_resid:", f"{out_bad['relative_residual']:.3e}")
print("    crossing() on per-channel scheme:", out_good["passed"],
      " rel_resid:", f"{out_good['relative_residual']:.3e}")

# (4) classical GR itself is 'manifestly' crossing symmetric the same way
gr_pass = C.crossing(amp, s=5.0, t=-2.0)  # dressed
d1g = complex(amp.eval(s, t, u, dressed=False))
d2g = complex(amp.eval(t, s, u, dressed=False))
print("[4] undressed GR bitwise:", d1g == d2g,
      "(saying 'GR passes crossing' is equally by-construction)")
