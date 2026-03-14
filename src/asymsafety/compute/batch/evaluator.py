"""Batch evaluation of beta functions over arrays of coupling points.

The core primitive for GPU acceleration.  Converts the per-point
``rhs_vector()`` callable into a vectorised
``f(points: ndarray) -> ndarray`` that evaluates beta functions at
*N* points simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from sympy import lambdify

if TYPE_CHECKING:
    from asymsafety.beta.system import BetaFunctionSystem


@runtime_checkable
class BatchEvaluator(Protocol):
    """Protocol for batch beta-function evaluation."""

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate beta functions at *N* points.

        Args:
            points: ``(N, n_couplings)`` array of coupling values.

        Returns:
            ``(N, n_couplings)`` array of beta-function values.
        """
        ...

    def evaluate_norms(self, points: np.ndarray) -> np.ndarray:
        """Evaluate |beta(x)| at *N* points.

        Returns:
            ``(N,)`` array of Euclidean norms.
        """
        ...


class NumpyBatchEvaluator:
    """Batch evaluator using NumPy broadcasting.

    The SymPy lambdified expressions are element-wise rational functions
    of the couplings, so they naturally support array inputs via NumPy
    broadcasting when each coupling is passed as a 1-D array.
    """

    def __init__(self, system: BetaFunctionSystem) -> None:
        from asymsafety.beta.system import BetaFunctionSystem  # noqa: WPS433

        self._system = system
        self._symbols = system.coupling_symbols
        self._names = system.coupling_names
        self._n_couplings = system.dimension

        # Create lambdified functions for each beta
        self._funcs: list = []
        for name in self._names:
            beta = system.beta(name)
            fn = lambdify(self._symbols, beta.expression, modules="numpy")
            self._funcs.append(fn)

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate beta functions at *N* points using NumPy broadcasting.

        Args:
            points: ``(N, n_couplings)`` array.

        Returns:
            ``(N, n_couplings)`` array.
        """
        N = points.shape[0]
        result = np.zeros((N, self._n_couplings))

        # Extract each coupling as a 1-D array
        args = [points[:, i] for i in range(self._n_couplings)]

        for j, fn in enumerate(self._funcs):
            try:
                result[:, j] = fn(*args)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                # Handle singularities point-by-point
                for i in range(N):
                    try:
                        result[i, j] = fn(
                            *[points[i, k] for k in range(self._n_couplings)]
                        )
                    except Exception:  # noqa: BLE001
                        result[i, j] = np.nan

        return result

    def evaluate_norms(self, points: np.ndarray) -> np.ndarray:
        """Evaluate |beta(x)| at *N* points."""
        betas = self.evaluate_batch(points)
        return np.linalg.norm(betas, axis=1)
