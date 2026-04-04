"""Mellin transform of RG flow trajectories.

Computes the Mellin transform

    M[g(k)](s) = integral  g(k)  k^{s-1}  dk

where k = exp(t) is the RG momentum scale (t = log(k/k_0)).  Poles in
the Mellin s-plane correspond to scaling dimensions of the operators
dual to the couplings.

Evaluating along the critical line Re(s) = 1/2 gives a spectral
decomposition analogous to the Riemann zeta function, where the
imaginary parts of the poles encode the oscillatory scaling behaviour.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.transforms._types import MellinResult


class RGFlowMellin:
    """Mellin transform of RG flow coupling trajectories.

    The RG time t = log(k/k_0) is converted to momentum scale
    k = exp(t) so that the Mellin integral is evaluated in k-space.

    Args:
        trajectory: Integrated RG flow trajectory.
    """

    def __init__(self, trajectory: RGTrajectory) -> None:
        self._trajectory = trajectory
        self._t = trajectory.t_values
        self._k = np.exp(trajectory.t_values)
        self._names = trajectory.coupling_names
        self._signals = np.array(
            [trajectory.coupling_values[name] for name in self._names]
        )

    def transform(self, s_values: np.ndarray) -> MellinResult:
        """Evaluate the Mellin transform on a set of complex *s* values.

        For each coupling:

            M_i(s) = integral g_i(k) k^{s-1} dk

        integrated numerically via the trapezoidal rule over the k-grid
        derived from the trajectory's t-values.

        Args:
            s_values: 1-D array of (possibly complex) Mellin variable
                values.  Should have positive real part for convergence.

        Returns:
            MellinResult with transform values of shape
            ``(n_couplings, len(s_values))``, and detected scaling
            dimensions.
        """
        k = self._k
        n_couplings = len(self._names)
        n_s = len(s_values)

        transform_values = np.empty((n_couplings, n_s), dtype=np.complex128)

        for j, s in enumerate(s_values):
            kernel = k ** (s - 1)
            for i in range(n_couplings):
                integrand = self._signals[i] * kernel
                transform_values[i, j] = trapezoid(integrand, k)

        scaling_dims = self.scaling_dimensions()

        return MellinResult(
            domain_values=s_values,
            transform_values=transform_values,
            coupling_names=list(self._names),
            scaling_dimensions=scaling_dims,
        )

    def scaling_dimensions(self, n_search: int = 100) -> np.ndarray:
        """Detect scaling dimensions from the Mellin transform.

        Evaluates the Mellin transform on the critical line
        Re(s) = 1/2 + i*t for a range of imaginary parts, and detects
        poles as local maxima of |M(s)|.  The imaginary parts of the
        detected poles correspond to scaling dimensions Delta of the
        operators.

        Args:
            n_search: Number of evaluation points along the critical
                line.

        Returns:
            1-D array of detected scaling dimensions (imaginary parts
            of the poles on the critical line), sorted in ascending
            order.
        """
        k = self._k

        # Scan the critical line Re(s) = 0.5
        t_imag = np.linspace(0.1, 20.0, n_search)
        s_line = 0.5 + 1j * t_imag

        # Compute aggregate magnitude across all couplings
        magnitudes = np.zeros(n_search)
        for j, s in enumerate(s_line):
            kernel = k ** (s - 1)
            for i in range(len(self._names)):
                integrand = self._signals[i] * kernel
                magnitudes[j] += np.abs(trapezoid(integrand, k))

        # Detect local maxima
        dims: list[float] = []
        for j in range(1, n_search - 1):
            if magnitudes[j] > magnitudes[j - 1] and magnitudes[j] > magnitudes[j + 1]:
                dims.append(t_imag[j])

        return np.sort(np.array(dims))
