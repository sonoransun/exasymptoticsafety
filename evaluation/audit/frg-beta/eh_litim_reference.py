"""HV-8a: Independent EH-Litim reference system (d=4, harmonic gauge alpha=1).

Implements the literature closed forms (Reuter hep-th/9605030, Lauscher-Reuter
hep-th/0108040, CPR 0805.2909 type-Ia + optimized cutoff) from the blind
derivation notes, with NO repo imports. Targets:
    lambda* = 0.193201, g* = 0.707321, theta = 1.475302 +- 3.043206 i,
    eta_N(FP) = -2.
"""
import numpy as np
from scipy.optimize import fsolve

PI = np.pi


def eta_N(g, lam):
    A = 1.0 / (1.0 - 2.0 * lam)
    B1 = (1.0 / (3.0 * PI)) * (5.0 * A - 9.0 * A**2 - 7.0)
    B2 = -(1.0 / (12.0 * PI)) * (5.0 * A - 6.0 * A**2)
    return g * B1 / (1.0 - g * B2)


def betas(u):
    g, lam = u
    A = 1.0 / (1.0 - 2.0 * lam)
    eta = eta_N(g, lam)
    beta_g = (2.0 + eta) * g
    beta_lam = -(2.0 - eta) * lam + (g / (2.0 * PI)) * (
        5.0 * A - 4.0 - (5.0 / 6.0) * eta * A
    )
    return np.array([beta_g, beta_lam])


def jacobian(u, h=1e-7):
    J = np.zeros((2, 2))
    for j in range(2):
        up = u.copy(); um = u.copy()
        up[j] += h; um[j] -= h
        J[:, j] = (betas(up) - betas(um)) / (2 * h)
    return J


def main():
    # Semi-analytic cross-check: A* is the real root >1 of 24A^3-59A^2+41A-14
    roots = np.roots([24.0, -59.0, 41.0, -14.0])
    A_star = float([r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 1][0])
    lam_semi = (A_star - 1.0) / (2.0 * A_star)
    g_semi = 12.0 * PI / (24.0 * A_star**2 - 15.0 * A_star + 14.0)
    print(f"semi-analytic: A*={A_star:.7f} lam*={lam_semi:.6f} g*={g_semi:.6f}")

    sol = fsolve(betas, [0.7, 0.19], full_output=True, xtol=1e-13)
    u_star, info, ier, msg = sol
    assert ier == 1, msg
    g_star, lam_star = u_star
    print(f"fsolve FP:     g*={g_star:.6f} lam*={lam_star:.6f} "
          f"g*lam*={g_star*lam_star:.6f}")
    print(f"eta_N(FP) = {eta_N(g_star, lam_star):+.12f}  (must be -2)")

    M = jacobian(u_star)
    theta = -np.linalg.eigvals(M)
    print(f"theta = {theta[0]:.6f}, {theta[1]:.6f}")

    # Compare against published targets
    ok = (abs(lam_star - 0.193201) < 1e-2 and abs(g_star - 0.707321) < 1e-2
          and abs(theta[0].real - 1.475302) < 1e-2
          and abs(abs(theta[0].imag) - 3.043206) < 1e-2
          and abs(eta_N(g_star, lam_star) + 2) < 1e-9)
    print("REFERENCE VALIDATED:", ok)

    # Identity checks (notes section 4)
    for lam in (-0.3, 0.0, 0.2, 0.45):
        b = betas(np.array([0.0, lam]))
        assert abs(b[1] - (-2 * lam)) < 1e-14, (lam, b)
    print("beta_lambda(0,lam) == -2 lam : OK")
    J0 = jacobian(np.array([0.0, 0.0]))
    print("GFP Jacobian eigs:", np.linalg.eigvals(J0),
          " dbeta_lam/dg|_0 =", J0[1, 0], " (lit: 1/(2pi) =", 1/(2*PI), ")")


if __name__ == "__main__":
    main()
