"""Laplace transform of RG flow trajectories.

Computes the one-sided Laplace transform

    G_i(s) = integral_0^T  g_i(t) exp(-s t) dt

of each coupling trajectory g_i(t).  Poles in the Laplace domain at
s = -theta_j recover the critical exponents of the fixed point from
which the trajectory emanates.

A rational (transfer-function) approximation P(s)/Q(s) can also be
fitted to capture the leading-pole structure.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.transforms._types import LaplaceResult


class RGFlowLaplace:
    """Laplace transform of RG flow coupling trajectories.

    Args:
        trajectory: Integrated RG flow trajectory.
    """

    def __init__(self, trajectory: RGTrajectory) -> None:
        self._trajectory = trajectory
        self._t = trajectory.t_values
        self._names = trajectory.coupling_names
        self._signals = np.array(
            [trajectory.coupling_values[name] for name in self._names]
        )

    def transform(self, s_values: np.ndarray) -> LaplaceResult:
        """Evaluate the numerical Laplace transform on a set of *s* values.

        For each coupling and each value of *s* the integral

            G_i(s) = integral g_i(t) exp(-s t) dt

        is evaluated via the trapezoidal rule over the trajectory time
        grid.

        Args:
            s_values: 1-D array of real, positive Laplace variable values.

        Returns:
            LaplaceResult with transform values of shape
            ``(n_couplings, len(s_values))``.
        """
        t = self._t
        n_couplings = len(self._names)
        n_s = len(s_values)

        transform_values = np.empty((n_couplings, n_s), dtype=np.complex128)

        for j, s in enumerate(s_values):
            kernel = np.exp(-s * t)
            for i in range(n_couplings):
                integrand = self._signals[i] * kernel
                transform_values[i, j] = trapezoid(integrand, t)

        poles = self.detect_poles()

        return LaplaceResult(
            domain_values=s_values,
            transform_values=transform_values,
            coupling_names=list(self._names),
            poles=poles,
        )

    def detect_poles(
        self,
        n_poles: int = 5,
        s_range: tuple[float, float] = (0.1, 10.0),
        n_search: int = 200,
    ) -> dict[str, np.ndarray]:
        """Detect poles of G_i(s) as local maxima of |G_i(s)|.

        The Laplace transform of a sum of exponentials has poles where
        the denominator vanishes; numerically these appear as peaks in
        |G(s)| on the real *s* axis.

        Poles at s = -theta_j recover critical exponents from the
        Laplace domain.

        Args:
            n_poles: Maximum number of poles to return per coupling.
            s_range: (s_min, s_max) search interval.
            n_search: Number of grid points for the search.

        Returns:
            Dictionary mapping coupling names to arrays of detected
            pole locations, sorted by magnitude of G at the pole
            (strongest first).
        """
        s_grid = np.linspace(s_range[0], s_range[1], n_search)
        t = self._t

        result: dict[str, np.ndarray] = {}

        for idx, name in enumerate(self._names):
            signal = self._signals[idx]
            magnitudes = np.empty(n_search)
            for j, s in enumerate(s_grid):
                kernel = np.exp(-s * t)
                val = trapezoid(signal * kernel, t)
                magnitudes[j] = np.abs(val)

            # Find local maxima
            pole_indices: list[int] = []
            for j in range(1, n_search - 1):
                if magnitudes[j] > magnitudes[j - 1] and magnitudes[j] > magnitudes[j + 1]:
                    pole_indices.append(j)

            # Sort by magnitude (strongest first) and keep up to n_poles
            pole_indices.sort(key=lambda k: magnitudes[k], reverse=True)
            pole_indices = pole_indices[:n_poles]

            result[name] = np.array([s_grid[k] for k in pole_indices])

        return result

    def transfer_function(
        self,
        n_poles: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a rational approximation P(s)/Q(s) to the Laplace data.

        The approximation uses polynomial least-squares fitting to
        determine numerator and denominator coefficients of a rational
        transfer function that captures the leading-pole structure.

        The denominator is constrained to be monic (leading coefficient
        one).

        Args:
            n_poles: Order of the denominator polynomial (number of
                poles in the approximation).

        Returns:
            ``(numerator_coeffs, denominator_coeffs)`` where each is a
            1-D array of polynomial coefficients in descending power
            order (highest degree first), suitable for use with
            ``numpy.polyval``.
        """
        # Evaluate the transform on a well-spaced grid
        n_pts = max(20, 4 * (n_poles + 1))
        s_grid = np.linspace(0.5, 10.0, n_pts)
        t = self._t

        # Use the first coupling for the rational fit
        signal = self._signals[0]
        g_vals = np.empty(n_pts)
        for j, s in enumerate(s_grid):
            kernel = np.exp(-s * t)
            g_vals[j] = float(np.real(trapezoid(signal * kernel, t)))

        # Fit P(s)/Q(s) where deg(P) = n_poles - 1, deg(Q) = n_poles.
        # Rearrange as  G(s) * Q(s) = P(s)  and solve in least-squares
        # sense.  Q(s) is monic: Q(s) = s^n_poles + q_{n-1} s^{n-1} + ...
        n_num = n_poles  # numerator degree = n_poles - 1 => n_poles coeffs
        n_den = n_poles  # number of free denominator coefficients (excluding leading 1)

        # Build Vandermonde-like columns
        # Unknowns: [q_{n-1}, ..., q_0, p_{n-1}, ..., p_0]
        n_unknowns = n_den + n_num
        A = np.empty((n_pts, n_unknowns))

        for j, s in enumerate(s_grid):
            # Denominator columns: -G(s) * s^k for k in [n_poles-1, ..., 0]
            for k in range(n_den):
                power = n_den - 1 - k
                A[j, k] = -g_vals[j] * s**power

            # Numerator columns: s^k for k in [n_num-1, ..., 0]
            for k in range(n_num):
                power = n_num - 1 - k
                A[j, n_den + k] = s**power

        # RHS: G(s) * s^n_poles  (from the monic leading term)
        rhs = np.array([g_vals[j] * s_grid[j] ** n_poles for j in range(n_pts)])

        coeffs, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        denominator_coeffs = np.concatenate(([1.0], coeffs[:n_den]))
        numerator_coeffs = coeffs[n_den:]

        return numerator_coeffs, denominator_coeffs
