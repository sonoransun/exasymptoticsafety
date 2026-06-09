"""Independent de Donder / S=1+iT sign check of the t-channel graviton
exchange between massless scalars, done numerically with explicit
4-vectors in BOTH signatures, to test whether any signature choice can
flip the overall sign of M(s,t,u).

Conventions tested:
  (A) mostly-minus (P&S): vertex tau^{mu nu} = -i(kappa/2)[p^mu p'^nu + p^nu p'^mu
      - eta^{mu nu}(p.p' - m^2)], propagator iP/k^2,
      P = 1/2(eta eta + eta eta - eta eta).
  (B) mostly-plus (Srednicki/DKRS): same T^{mu nu} structure with
      eta -> diag(-1,1,1,1), propagator (1/i) P / k^2 i.e. -iP/k^2
      (massless: 1/[i(k^2 - i eps)] with k^2 = +vec k^2 - k0^2),
      vertex +i(kappa/2)T (sign squared cancels anyway).
Then compare against +/- (kappa^2/4) su/t.
"""
import numpy as np

kappa2_over4 = 1.0  # set 8 pi G = 1

def mandelstam(p1,p2,p3,p4,eta):
    dot = lambda a,b: a @ eta @ b
    sgn = 1.0 if eta[0,0] > 0 else -1.0   # s = (p1+p2)^2 (mostly-minus) or -(p1+p2)^2 (mostly-plus)
    s = sgn*dot(p1+p2,p1+p2); t = sgn*dot(p1-p3,p1-p3); u = sgn*dot(p1-p4,p1-p4)
    return s,t,u

def T_munu(p, pp, eta):
    """matrix element of stress tensor between incoming p, outgoing pp (massless)."""
    inv = np.linalg.inv(eta)
    pu = inv @ p; ppu = inv @ pp          # raised-index vectors
    dot = p @ inv @ pp
    return np.outer(pu, ppu) + np.outer(ppu, pu) - inv * dot

def amp_t_channel(p1,p2,p3,p4,eta,prop_sign):
    """i M_t = [i_v (k/2) T1] [prop_sign * i P /k^2] [i_v (k/2) T2]; vertex i's squared
    give -1 regardless, so M_t = prop_sign * (k^2/4) T1.P.T2 / k_t^2 * (-1) * (1/i->...)"""
    inv = np.linalg.inv(eta)
    q = p1 - p3
    k2 = q @ inv @ q
    T1 = T_munu(p1, p3, eta)   # contravariant T1^{mu nu}
    T2 = T_munu(p2, p4, eta)
    # contract with P_{mu nu rho si} = 1/2(eta_mr eta_ns + eta_ms eta_nr - eta_mn eta_rs)
    T1d = eta @ T1 @ eta       # lower indices
    TT = np.einsum('mn,mn->', T1d, T2)
    tr1 = np.einsum('mn,mn->', eta, T1)
    tr2 = np.einsum('mn,mn->', eta, T2)
    contr = TT - 0.5*tr1*tr2
    # iM = (vertex i factor)^2 * (kappa/2)^2 * [prop_sign * i * P/k2] => M = -prop_sign*(kappa^2/4)*contr/k2
    return -prop_sign * kappa2_over4 * contr / k2

# physical massless kinematics: E=1, scattering angle th
th = 1.234
c, s_ = np.cos(th), np.sin(th)
E = 1.0
p1 = np.array([E, 0, 0,  E]); p2 = np.array([E, 0, 0, -E])
p3 = np.array([E, E*s_, 0, E*c]); p4 = np.array([E, -E*s_, 0, -E*c])

for name, eta, prop_sign in (("mostly-minus (+---), prop +iP/k^2", np.diag([1.,-1,-1,-1]), +1),
                             ("mostly-plus (-+++), prop -iP/k^2 (=1/i /(k^2))", np.diag([-1.,1,1,1]), -1)):
    s,t,u = mandelstam(p1,p2,p3,p4,eta)
    Mt = amp_t_channel(p1,p2,p3,p4,eta,prop_sign)
    print(f"{name}: s={s:+.4f} t={t:+.4f} u={u:+.4f}")
    print(f"   M_t = {Mt:+.6f};  +su/t = {s*u/t:+.6f};  -su/t = {-s*u/t:+.6f}")
    print(f"   => sign matches {'+8piG su/t' if abs(Mt - s*u/t) < 1e-9 else ('-8piG su/t' if abs(Mt + s*u/t)<1e-9 else 'NEITHER')}")
