"""Vectorized spectral sum evaluation.

Replaces the 100K-iteration Python double loop in SpectralSumEvaluator
with vectorized NumPy/JAX operations.
"""

from __future__ import annotations
import numpy as np


def trace_on_S1xS3_vectorized(
    W_func,
    field_type: str,
    R3_val: float,
    beta_period: float,
    n_matsubara: int = 100,
    l_max: int = 500,
) -> float:
    """Vectorized trace evaluation on S^1 x S^3.

    Replaces the nested Python loop with NumPy array operations.
    100K evaluations become a single vectorized operation.

    Args:
        W_func: Callable(z_spatial, omega_sq) -> float (must support array inputs)
        field_type: "scalar", "vector", or "TT"
        R3_val: Ricci scalar of S^3
        beta_period: Period of S^1
        n_matsubara: Maximum Matsubara index
        l_max: Maximum angular momentum
    """
    a_sq = 6.0 / R3_val  # R^(3) = 6/a^2 for S^3

    l_min = {"scalar": 0, "vector": 1, "TT": 2}.get(field_type, 0)

    # Build 2D grid of (n, l) values
    ns = np.arange(-n_matsubara, n_matsubara + 1)     # 201 values
    ls = np.arange(l_min, l_max + 1)                   # ~500 values
    N_grid, L_grid = np.meshgrid(ns, ls, indexing='ij')

    # Flatten for vectorized evaluation
    n_flat = N_grid.ravel()
    l_flat = L_grid.ravel()

    # Compute Matsubara frequencies: omega^2 = (2*pi*n/beta)^2
    omega_sq = (2 * np.pi * n_flat / beta_period)**2

    # Compute eigenvalues on S^3
    # For S^3 (d=3): eigenvalue = l(l+2)/a^2
    base = l_flat * (l_flat + 2)
    shift = {"scalar": 0, "vector": 1, "TT": 2}.get(field_type, 0)
    lam_l = (base - shift) / a_sq

    # Compute multiplicities on S^3
    d_l = _S3_multiplicity_vectorized(field_type, l_flat)

    # Evaluate W at all points
    try:
        w_vals = W_func(lam_l, omega_sq)
    except Exception:
        # Fallback: evaluate point-by-point
        w_vals = np.array([W_func(float(z), float(w)) for z, w in zip(lam_l, omega_sq)])

    # Handle NaN/Inf
    w_vals = np.nan_to_num(w_vals, nan=0.0, posinf=0.0, neginf=0.0)

    return float(np.sum(d_l * w_vals))


def _S3_multiplicity_vectorized(field_type: str, l: np.ndarray) -> np.ndarray:
    """Vectorized multiplicities on S^3.

    Scalars: (l+1)^2 (l >= 0); transverse vectors: 2l(l+2) (l >= 1);
    TT tensors: 2(l-1)(l+3) (l >= 2).
    Reference: Rubin & Ordonez, J. Math. Phys. 25 (1984) 2888 (d=3).
    """
    if field_type == "scalar":
        return (l + 1).astype(float)**2
    elif field_type == "vector":
        result = 2.0 * l * (l + 2)
        result[l < 1] = 0
        return result
    elif field_type == "TT":
        result = 2.0 * (l - 1) * (l + 3)
        result[l < 2] = 0
        return result
    return np.ones_like(l, dtype=float)


def trace_on_sphere_vectorized(
    W_func,
    field_type: str,
    R_bar_val: float,
    d: int = 4,
    l_max: int = 500,
) -> float:
    """Vectorized trace evaluation on S^d.

    Replaces the Python loop with a single NumPy operation.
    """
    a_sq = d * (d - 1) / R_bar_val
    l_min = {"scalar": 0, "vector": 1, "TT": 2}.get(field_type, 0)

    ls = np.arange(l_min, l_max + 1, dtype=float)

    # Eigenvalues
    base = ls * (ls + d - 1)
    shift = {"scalar": 0, "vector": 1, "TT": 2}.get(field_type, 0)
    lam_l = (base - shift) / a_sq

    # Multiplicities (S^4 specific for d=4)
    # Rubin & Ordonez, J. Math. Phys. 25 (1984) 2888:
    #   scalar (2l+3)(l+1)(l+2)/6; transverse vector l(l+3)(2l+3)/2;
    #   TT 5(2l+3)(l-1)(l+4)/6.
    if d == 4:
        if field_type == "scalar":
            d_l = (2 * ls + 3) * (ls + 1) * (ls + 2) / 6
        elif field_type == "vector":
            d_l = ls * (ls + 3) * (2 * ls + 3) / 2
            d_l[ls < 1] = 0
        elif field_type == "TT":
            d_l = 5 * (2 * ls + 3) * (ls - 1) * (ls + 4) / 6
            d_l[ls < 2] = 0
        else:
            d_l = np.ones_like(ls)
    else:
        raise NotImplementedError(f"Multiplicities for S^{d}")

    # Evaluate
    try:
        w_vals = W_func(lam_l)
    except Exception:
        w_vals = np.array([W_func(float(z)) for z in lam_l])

    w_vals = np.nan_to_num(w_vals, nan=0.0, posinf=0.0, neginf=0.0)

    return float(np.sum(d_l * w_vals))
