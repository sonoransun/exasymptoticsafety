"""Resolvent operator of the RG stability matrix.

The resolvent R(s) = (s I - M)^{-1} is a matrix-valued meromorphic
function of the complex variable s with poles at the eigenvalues of M.
Since the eigenvalues satisfy lambda_i = -theta_i, the poles of the
resolvent directly encode the critical exponents.

The epsilon-pseudospectrum

    sigma_epsilon(M) = { s in C : ||R(s)|| > 1/epsilon }

extends the spectrum to capture non-normal sensitivity: even when all
eigenvalues are well-separated, a highly non-normal stability matrix
can produce transient growth that is invisible from eigenvalues alone.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import inv, svdvals

from asymsafety.analysis.stability import StabilityAnalysis


class ResolventOperator:
    """Resolvent of the stability matrix and pseudospectral analysis.

    Args:
        stability: A completed stability analysis at the fixed point.
    """

    def __init__(self, stability: StabilityAnalysis) -> None:
        self._stability = stability
        self._M = stability.stability_matrix
        self._n = self._M.shape[0]

    def evaluate(self, s: complex) -> np.ndarray:
        """Evaluate the resolvent R(s) = (s I - M)^{-1}.

        Args:
            s: Complex spectral parameter.

        Returns:
            Resolvent matrix of shape (n, n).

        Raises:
            numpy.linalg.LinAlgError: If s is an eigenvalue of M
                (the resolvent has a pole there).
        """
        return inv(s * np.eye(self._n) - self._M)

    def resolvent_norm(self, s: complex) -> float:
        """Operator 2-norm of the resolvent at s.

        ||R(s)||_2 = sigma_max(R(s)), the largest singular value.

        Args:
            s: Complex spectral parameter.

        Returns:
            The operator 2-norm (a positive float).
        """
        R = self.evaluate(s)
        return float(svdvals(R)[0])

    def pseudospectrum(
        self,
        epsilon: float,
        real_range: tuple[float, float],
        imag_range: tuple[float, float],
        n_grid: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the epsilon-pseudospectrum on a grid.

        The epsilon-pseudospectrum is the set

            sigma_epsilon = { s in C : ||R(s)||_2 > 1/epsilon }

        which reduces to the exact spectrum as epsilon -> 0.

        Args:
            epsilon: Pseudospectral level.
            real_range: (re_min, re_max) for the real axis grid.
            imag_range: (im_min, im_max) for the imaginary axis grid.
            n_grid: Number of grid points per axis.

        Returns:
            Tuple (real_grid, imag_grid, norm_grid) where real_grid and
            imag_grid are 1-D arrays of length n_grid and norm_grid is a
            2-D array of shape (n_grid, n_grid) containing ||R(s)||_2
            at each grid point s = re + i*im.
        """
        real_grid = np.linspace(real_range[0], real_range[1], n_grid)
        imag_grid = np.linspace(imag_range[0], imag_range[1], n_grid)
        norm_grid = np.empty((n_grid, n_grid))

        I = np.eye(self._n)  # noqa: E741

        for i, re in enumerate(real_grid):
            for j, im in enumerate(imag_grid):
                s = complex(re, im)
                A = s * I - self._M
                # Minimum singular value gives 1/||R(s)||
                svals = svdvals(A)
                sigma_min = svals[-1]
                # ||R(s)|| = 1/sigma_min(sI - M)
                if sigma_min < 1e-15:
                    norm_grid[i, j] = 1e15
                else:
                    norm_grid[i, j] = 1.0 / sigma_min

        return real_grid, imag_grid, norm_grid

    def poles(self) -> np.ndarray:
        """Poles of the resolvent (eigenvalues of M).

        These equal -theta_i where theta_i are the critical exponents.

        Returns:
            Array of eigenvalues of the stability matrix.
        """
        return self._stability.eigenvalues
