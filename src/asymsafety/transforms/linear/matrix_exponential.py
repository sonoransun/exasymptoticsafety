"""Matrix exponential propagation of linearised RG flow.

Near a fixed point g*, the flow decomposes as g(t) = g* + delta_g(t)
where the deviation satisfies

    d/dt delta_g = M delta_g + O(delta_g^2)

with M_ij = partial beta_i / partial g_j |_{g*} the stability matrix.
The exact linear solution is

    delta_g(t) = exp(M t) delta_g(0)

which is diagonalised by the eigenvectors of M:

    delta_g(t) = V diag(exp(-theta_i t)) V^{-1} delta_g(0)

where theta_i = -eigenvalue_i are the critical exponents.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.analysis.stability import StabilityAnalysis
from asymsafety.transforms._types import ComparisonResult


class MatrixExponentialFlow:
    """Propagate perturbations around a fixed point via exp(M t).

    This provides the exact solution to the linearised RG flow equation
    and serves as a reference against which nonlinear trajectories can
    be compared.

    Args:
        stability: A completed stability analysis at the fixed point.
    """

    def __init__(self, stability: StabilityAnalysis) -> None:
        self._stability = stability
        self._M = stability.stability_matrix

    @property
    def stability_matrix(self) -> np.ndarray:
        """The stability matrix M_ij = partial beta_i / partial g_j."""
        return self._M

    def propagate(
        self, delta_g0: np.ndarray, t_values: np.ndarray
    ) -> np.ndarray:
        """Propagate an initial perturbation through the linearised flow.

        Computes delta_g(t) = expm(M * t) @ delta_g0 for each t in
        ``t_values``.

        Args:
            delta_g0: Initial displacement from the fixed point, shape
                (n_couplings,).
            t_values: Array of RG times at which to evaluate, shape
                (n_times,).

        Returns:
            Array of shape (n_times, n_couplings) with the linearised
            trajectory.
        """
        n_couplings = len(delta_g0)
        n_times = len(t_values)
        result = np.empty((n_times, n_couplings))

        for i, t in enumerate(t_values):
            result[i] = expm(self._M * t) @ delta_g0

        return result

    def eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition of the stability matrix.

        The stability matrix factorises as

            M = V @ diag(-theta) @ V^{-1}

        where theta_i are the critical exponents and V contains the
        right eigenvectors as columns.

        Returns:
            Tuple (V, Theta) where V is the eigenvector matrix of shape
            (n, n) and Theta is the array of critical exponents of shape
            (n,).
        """
        V = self._stability.eigenvectors
        Theta = self._stability.critical_exponents
        return V, Theta

    def compare_with_flow(
        self, trajectory: RGTrajectory
    ) -> ComparisonResult:
        """Compare linearised propagation with an actual RG trajectory.

        Computes the initial displacement of the trajectory from the
        fixed point, propagates it with exp(M t), and measures the
        deviation from the full nonlinear trajectory.

        Args:
            trajectory: An RGTrajectory computed from the full nonlinear
                beta function system.

        Returns:
            ComparisonResult quantifying the agreement.
        """
        fp_loc = self._stability.fixed_point.location
        names = trajectory.coupling_names

        # Initial displacement from the fixed point
        delta_g0 = np.array([
            trajectory.coupling_values[name][0] - fp_loc[name]
            for name in names
        ])

        # Linearised trajectory
        linearised = self.propagate(delta_g0, trajectory.t_values)

        # Actual trajectory values as an array
        actual = np.column_stack([
            trajectory.coupling_values[name] for name in names
        ])

        # Shift actual trajectory to deviations from FP
        fp_array = np.array([fp_loc[name] for name in names])
        actual_delta = actual - fp_array[np.newaxis, :]

        max_deviation = float(np.max(np.abs(linearised - actual_delta)))

        # Agreement criterion: max deviation is small relative to the
        # perturbation magnitude (10% tolerance by default)
        perturbation_scale = max(np.max(np.abs(delta_g0)), 1e-15)
        agrees = max_deviation < 0.1 * perturbation_scale

        return ComparisonResult(
            method_a="matrix_exponential",
            method_b="nonlinear_trajectory",
            values_a=linearised.ravel(),
            values_b=actual_delta.ravel(),
            max_deviation=max_deviation,
            agrees=agrees,
        )
