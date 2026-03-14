"""Batched Newton solver for parallel root-finding.

Finds roots of beta(x) = 0 for multiple initial guesses simultaneously.
Works with NumPy; when JAX is available, can be GPU-accelerated.
"""

from __future__ import annotations
import numpy as np


def batched_newton(
    f: callable,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    jacobian_fd_eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched Newton's method for finding roots of f(x) = 0.

    Solves N independent root-finding problems simultaneously.

    Args:
        f: Callable (N, d) -> (N, d), the function to find roots of.
        x0: (N, d) array of initial guesses.
        tol: Convergence tolerance on |f(x)|.
        max_iter: Maximum iterations.
        jacobian_fd_eps: Step size for finite-difference Jacobian.

    Returns:
        roots: (N, d) array of found roots.
        converged: (N,) boolean mask of converged solutions.
        residuals: (N,) array of final |f(x)| values.
    """
    N, d = x0.shape
    x = x0.copy()
    converged = np.zeros(N, dtype=bool)

    for iteration in range(max_iter):
        # Evaluate f at all current points
        fx = f(x)  # (N, d)
        residuals = np.linalg.norm(fx, axis=1)  # (N,)

        # Mark converged
        newly_converged = (residuals < tol) & ~converged
        converged |= newly_converged

        # Stop if all converged
        if converged.all():
            break

        # Compute Jacobian via finite differences for non-converged points
        active = ~converged
        n_active = active.sum()
        if n_active == 0:
            break

        # Jacobian: J[i,j] = df_i/dx_j
        J = np.zeros((N, d, d))
        for j in range(d):
            x_plus = x.copy()
            x_plus[:, j] += jacobian_fd_eps
            fx_plus = f(x_plus)
            J[:, :, j] = (fx_plus - fx) / jacobian_fd_eps

        # Newton step: dx = -J^{-1} f(x)
        # Only for non-converged points
        for i in range(N):
            if not converged[i]:
                try:
                    dx = np.linalg.solve(J[i], -fx[i])
                    x[i] += dx
                except np.linalg.LinAlgError:
                    # Singular Jacobian, mark as failed
                    pass

    residuals = np.linalg.norm(f(x), axis=1)
    converged = residuals < tol

    return x, converged, residuals
