"""Defense probe for HV-15c-b: can ANY threshold-function convention,
regulator, or eta-sign convention give the foliated system an NGFP at
g*~0.96, lambda*~0.20?

Structure of the implemented system:
  A_fol = (1/3pi)[5 Phi^1_1(w) - 4 Phi^1_1(0) + 6 Phi^2_1(w)],  w=-2*lambda
  B_fol = -(1/6pi)[5 tPhi^1_1(w) - 4 tPhi^1_1(0) + 6 tPhi^2_1(w)]
  eta = g A/(1-gB);  beta_g = (2+eta) g

Claim to test: for lambda in (-1/2, 1/2), A_fol>0 and B_fol<0 for EVERY
admissible regulator (Phi positive, monotonically decreasing in w),
hence eta>0 for g>0 and beta_g>2g>0: no NGFP. Verify for Litim AND
exponential regulator numerically.
"""
import sys
import numpy as np
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
from asymsafety.frg.threshold import ThresholdFunctions
import inspect
import asymsafety.frg.threshold as th

print("ThresholdFunctions signature:", inspect.signature(ThresholdFunctions.__init__))
src = inspect.getsource(th)
print("--- regulator options in threshold.py (grep 'def Phi' / class) ---")
for line in src.splitlines():
    if line.strip().startswith(("def ", "class ", "regulator", "self.regulator")) :
        print(" ", line.rstrip())
