"""HV-6b defense check.

(1) Brute-force tr(Omega_mu_nu Omega^mu_nu) on the Sym^2 (and traceless Sym^2_0)
    bundle over unit S^4, where (Omega_mn T)_ab = -R^c_amn T_cb - R^c_bmn T_ac.
    Unit S^4: R_abcd = (g_ac g_bd - g_ad g_bc), R = 12, Riem^2 = 24.
    Claimed truth: tr(Om^2)|_Sym2_0 = -(d+2) Riem^2 = -144.
    Code's implied value: +2(d-1) Riem^2 = +144 (sign flip).
    Note sign of Omega convention is irrelevant (quadratic).

(2) Scan: is there ANY constant endomorphism c*R such that exact TT b4 on S^4
    equals the code's 155/432 R^2?  b4(c) = R^2(-1/432 + 5c/6 + 5c^2/2).

(3) Vector cross-check of the brute force: tr(Om^2)|_vector should be -Riem^2 = -24.
"""
import itertools
import numpy as np

d = 4
g = np.eye(d)
# Riemann of unit S^4 (R = d(d-1) = 12): R_{abcd} = g_ac g_bd - g_ad g_bc
Riem = np.zeros((d, d, d, d))
for a, b, c, e in itertools.product(range(d), repeat=4):
    Riem[a, b, c, e] = g[a, c]*g[b, e] - g[a, e]*g[b, c]
Riem2 = np.einsum('abcd,abcd->', Riem, Riem)
print("Riem^2 (unit S^4) =", Riem2, " (expect 24)")

# --- vector bundle: (Omega_mn V)^a = R^a_{b mn} V^b ---
trOm2_vec = 0.0
for m, n in itertools.product(range(d), repeat=2):
    Om = Riem[:, :, m, n]            # (Om)^a_b
    trOm2_vec += np.trace(Om @ Om)   # indices raised with delta
print("tr(Om^2) vector       =", trOm2_vec, " (truth -Riem^2 = -24; code uses +Riem^2)")

# --- Sym^2 bundle (10-dim), and traceless Sym^2_0 (9-dim) ---
# basis of Sym^2: pairs (a,b) a<=b
pairs = [(a, b) for a in range(d) for b in range(a, d)]
P = len(pairs)

def omega_matrix(m, n):
    M = np.zeros((P, P))
    for i, (a, b) in enumerate(pairs):
        # (Om T)_{ab} = R^c_{a mn} T_{cb} + R^c_{b mn} T_{ac}  (overall sign irrelevant)
        for c in range(d):
            # T_{cb} component -> source pair sorted(c,b)
            j = pairs.index(tuple(sorted((c, b))))
            M[i, j] += Riem[c, a, m, n]
            j = pairs.index(tuple(sorted((a, c))))
            M[i, j] += Riem[c, b, m, n]
    return M

trOm2_sym = 0.0
M_sum = np.zeros((P, P))
for m, n in itertools.product(range(d), repeat=2):
    M = omega_matrix(m, n)
    trOm2_sym += np.trace(M @ M)
    M_sum += M @ M
print("tr(Om^2) Sym^2 (10-d) =", trOm2_sym, " (truth -(d+2)Riem^2 = -144)")

# traceless projection: trace part is the singlet T_{ab} ~ g_{ab}; Om annihilates it,
# but project explicitly to be safe.
# Build metric vector in pair basis (with symmetric-tensor inner product weights):
# inner product <S,T> = S_{ab} T^{ab}; off-diagonal pairs carry weight 2.
w = np.array([1.0 if a == b else 2.0 for (a, b) in pairs])
gvec = np.array([1.0 if a == b else 0.0 for (a, b) in pairs])
# projector onto trace part in component basis:
norm = np.sum(w * gvec * gvec)
Ptrace = np.outer(gvec, w * gvec) / norm
Ptl = np.eye(P) - Ptrace
trOm2_sym0 = np.trace(Ptl @ M_sum @ Ptl)
print("tr(Om^2) Sym^2_0 (9-d)=", trOm2_sym0, " (truth -(d+2)Riem^2 = -144; code +2(d-1)Riem^2 = +144)")

# --- (2) scan for constant endomorphism matching code's 155/432 ---
from fractions import Fraction
import math
target = Fraction(155, 432)
# b4(c) = -1/432 + (5/6)c + (5/2)c^2  must equal 155/432
# 5/2 c^2 + 5/6 c - 156/432 = 0  -> c = [-5/6 +- sqrt(25/36 + 10*156/432)]/5
disc = Fraction(25, 36) + 4*Fraction(5, 2)*Fraction(156, 432)
print("\nscan: discriminant =", disc, "= ", float(disc), "; sqrt =", math.sqrt(float(disc)))
r = math.sqrt(float(disc))
c1 = (-5/6 + r)/5
c2 = (-5/6 - r)/5
print("roots c =", c1, c2, " (rational/natural? no — sqrt(%s) is irrational)" % disc)
for name, c in [("-D^2 (c=0)", 0.0), ("code E=-R/6 (c=-1/6)", -1/6),
                ("Lichnerowicz (c=2/3)", 2/3)]:
    b4 = -1/432 + (5/6)*c + (5/2)*c**2
    print(f"  exact TT b4 for {name}: {b4*432:.4f}/432  (code: 155/432)")
