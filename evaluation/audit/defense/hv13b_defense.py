"""Defense check for HV-13b: is there ANY convention under which
veneziano_residue's -(-1)^n/n! * prod(a_t+k) is the residue of the module's
own Veneziano amplitude at alpha(s)=n?"""
import numpy as np
import mpmath as mp
mp.mp.dps = 40
from asymsafety.scattering import bootstrap as B

def ven(s, t, a0=1.0, ap=1.0):
    a_s, a_t = a0 + ap*mp.mpc(s), a0 + ap*mp.mpc(t)
    return mp.gamma(-a_s)*mp.gamma(-a_t)/mp.gamma(-a_s-a_t)

print("== 1. residues of the module's own amplitude, candidate conventions ==")
print(" n   t     lim(a_s-n)A   lim(n-a_s)A   lim(s-s_n)A   code")
for n in range(0, 5):
    for t in (-2.7, -0.6):
        sn = n - 1.0  # a0=1, ap=1
        # high-precision residue via mpmath limit
        eps = mp.mpf('1e-20')
        r_as = complex((eps)*ven(sn+eps, t)).real        # w.r.t. a_s (= w.r.t. s for ap=1)
        code = B.veneziano_residue(n, t)
        print(f" {n}  {t:+.1f}  {r_as:+.8f}   {-r_as:+.8f}   {r_as:+.8f}   {code:+.8f}")

print()
print("== 2. candidate closed forms vs true residue (w.r.t. a_s) ==")
from math import factorial
for n in range(0, 5):
    t = -2.7
    a_t = 1.0 + t
    poch = 1.0
    for k in range(1, n+1):
        poch *= (a_t + k)
    true = complex(mp.mpf('1e-20')*ven(n-1.0+mp.mpf('1e-20'), t)).real
    cand = {
        "-(1/n!)prod(a_t+k)      ": -poch/factorial(n),
        "+(1/n!)prod(a_t+k)      ": +poch/factorial(n),
        "-(-1)^n/n! prod (CODE)  ": -((-1)**n)/factorial(n)*poch,
        "+(-1)^n/n! prod         ": +((-1)**n)/factorial(n)*poch,
    }
    print(f" n={n}: true={true:+.8f}  " + "  ".join(f"{k.strip()}={v:+.8f}{'<==MATCH' if abs(v-true)<1e-9 else ''}" for k,v in cand.items()))

print()
print("== 3. exotic defense: polynomial in a_u under tachyonic constraint ==")
print("   (a_s+a_t+a_u=-1, i.e. s+t+u=4m^2=-4 for external tachyons, a0=ap=1)")
# If the code's argument were u, code(n,u) = -(-1)^n/n! prod(a_u+k).
# Under the tachyon constraint a_t = -1-n-a_u at a_s=n,
# -(1/n!)prod(a_t+k) = -(-1)^n/n! prod(a_u+k). Check identity:
for n in range(1, 5):
    a_u = 0.37
    a_t = -1.0 - n - a_u
    lhs = -1.0/factorial(n)
    rhs = -((-1)**n)/factorial(n)
    for k in range(1, n+1):
        lhs *= (a_t+k); rhs *= (a_u+k)
    print(f" n={n}: -(1/n!)prod(a_t+k)={lhs:+.6f}  -(-1)^n/n! prod(a_u+k)={rhs:+.6f}  equal={abs(lhs-rhs)<1e-12}")
# But the module uses MASSLESS kinematics (StringAmplitude m=0 -> s+t+u=0 -> a_s+a_t+a_u=3):
print("   massless (module's own) kinematics: a_s+a_t+a_u = 3 (a0=1,ap=1):")
for n in range(1, 4):
    a_u = 0.37
    a_t = 3.0 - n - a_u
    lhs = -1.0/factorial(n)
    rhs = -((-1)**n)/factorial(n)
    for k in range(1, n+1):
        lhs *= (a_t+k); rhs *= (a_u+k)
    print(f" n={n}: -(1/n!)prod(a_t+k)={lhs:+.6f}  -(-1)^n/n! prod(a_u+k)={rhs:+.6f}  equal={abs(lhs-rhs)<1e-12}")

print()
print("== 4. GSW-style pole expansion check: A = sum_n R_n/(a_s-n) ==")
# reconstruct A(s,t) from -(1/n!)prod(a_t+k)/(a_s-n) truncated, compare to true at s left of all used poles? 
# Instead verify single-pole dominance numerically very close to pole:
s0, t0 = 0.0 + 1e-8, -2.7   # a_s = 1+1e-8, n=1
A = complex(ven(s0, t0))
a_t = 1.0 + t0
R1_true = -(a_t+1)/1.0
print(f" near n=1: A={A.real:.4e}; R/(a_s-1) with R=-(1/1!)(a_t+1)={R1_true/1e-8:.4e}; with code R={-(-1)**1*(a_t+1)/1e-8:.4e}")

print()
print("== 5. zeros (sign-insensitive) and downstream usage sanity ==")
print(" residue_zeros(3) =", B.residue_zeros(3))
for tz in B.residue_zeros(3):
    print(f"   code residue(3,{tz:+.1f}) = {B.veneziano_residue(3, tz):+.2e}")
