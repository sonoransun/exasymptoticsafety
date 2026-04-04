"""Discrete RG transfer matrix.

When the RG flow is discretised with step dt, the linearised map
becomes

    delta_g_{n+1} = T delta_g_n

where the transfer matrix T = exp(M dt) (exact) or T = I + dt M
(Euler approximation).

The eigenvalues of T satisfy

    lambda_T = exp(-theta_i dt)

so the critical exponents can be recovered as

    theta_i = -log(lambda_T) / dt

This provides an independent check on the stability analysis and
connects the continuous RG to lattice / discrete RG blocking schemes.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.beta.system import BetaFunctionSystem


class DiscreteRGTransferMatrix:
    """Transfer matrix for a discretised RG step.

    Args:
        system: The beta function system.
        fp: The fixed point at which to linearise.
    """

    def __init__(
        self, system: BetaFunctionSystem, fp: FixedPoint
    ) -> None:
        self._system = system
        self._fp = fp
        self._M = system.jacobian_numerical(fp.location)
        self._n = self._M.shape[0]

    @property
    def stability_matrix(self) -> np.ndarray:
        """The underlying continuous stability matrix M."""
        return self._M

    def compute(
        self, dt: float, method: str = "exact"
    ) -> np.ndarray:
        """Compute the transfer matrix for a step of size dt.

        Args:
            dt: Discrete RG step size.
            method: ``"exact"`` uses the matrix exponential exp(M dt);
                ``"euler"`` uses the first-order approximation I + dt M.

        Returns:
            Transfer matrix T of shape (n, n).

        Raises:
            ValueError: If *method* is not ``"exact"`` or ``"euler"``.
        """
        if method == "exact":
            return expm(self._M * dt)
        elif method == "euler":
            return np.eye(self._n) + dt * self._M
        else:
            raise ValueError(
                f"Unknown method '{method}'; use 'exact' or 'euler'."
            )

    def eigenvalues(self, dt: float) -> np.ndarray:
        """Eigenvalues of the transfer matrix T(dt).

        These satisfy lambda_T = exp(-theta_i dt).

        Args:
            dt: Discrete RG step size.

        Returns:
            Array of transfer matrix eigenvalues.
        """
        T = self.compute(dt)
        return np.linalg.eigvals(T)

    def critical_exponents_from_transfer(
        self, dt: float
    ) -> np.ndarray:
        """Recover critical exponents from the transfer matrix.

        Computes theta_i = -log(lambda_T) / dt which should agree
        with the values obtained from direct stability analysis.

        Args:
            dt: Discrete RG step size.

        Returns:
            Array of critical exponents.
        """
        lam = self.eigenvalues(dt)
        return -np.log(lam) / dt

    def power_iteration(
        self,
        n_steps: int,
        initial: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Iterate the transfer matrix: delta_g_n = T^n delta_g_0.

        This provides a discrete-time trajectory of the linearised
        flow, suitable for comparison with continuous integration or
        lattice RG blocking.

        Args:
            n_steps: Number of discrete RG steps to perform.
            initial: Initial displacement from the fixed point, shape
                (n_couplings,).
            dt: Discrete RG step size.

        Returns:
            Array of shape (n_steps + 1, n_couplings) containing the
            trajectory from step 0 through step n_steps.
        """
        T = self.compute(dt)
        n_couplings = len(initial)
        trajectory = np.empty((n_steps + 1, n_couplings))
        trajectory[0] = initial

        current = initial.copy()
        for step in range(n_steps):
            current = T @ current
            trajectory[step + 1] = current

        return trajectory
