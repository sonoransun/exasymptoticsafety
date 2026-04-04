"""Koopman operator via EDMD from RG flow snapshots.

Lifts the nonlinear RG flow to a linear evolution on observables:

.. math::
    K^t[f](g) = f(\\phi^t(g))

The finite-dimensional approximation is obtained through Extended
Dynamic Mode Decomposition (EDMD) using a monomial dictionary lifted
from trajectory snapshot data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.analysis.flow import RGTrajectory
    from asymsafety.beta.system import BetaFunctionSystem


class KoopmanOperator:
    """Koopman operator via EDMD from RG flow snapshots.

    Parameters
    ----------
    system:
        The RG beta-function system whose flow is to be analysed.
    """

    def __init__(self, system: BetaFunctionSystem) -> None:
        self._system = system
        self._coupling_names = system.coupling_names
        self._n = system.dimension

    # ------------------------------------------------------------------
    # EDMD
    # ------------------------------------------------------------------

    def compute_edmd(
        self,
        trajectories: list[RGTrajectory],
        dictionary_degree: int = 4,
    ) -> dict:
        """Extended Dynamic Mode Decomposition.

        1. Collect snapshot pairs ``(x_k, x_{k+1})`` from trajectories.
        2. Lift each snapshot to the monomial dictionary of total degree
           ``<= dictionary_degree``.
        3. Solve the least-squares problem ``K = pinv(Psi_X) @ Psi_Y``.
        4. Eigendecompose ``K``.

        Parameters
        ----------
        trajectories:
            List of :class:`RGTrajectory` objects used as training data.
        dictionary_degree:
            Maximum total polynomial degree of the monomial dictionary.

        Returns
        -------
        dict
            Keys: ``eigenvalues``, ``eigenfunctions``, ``modes``,
            ``dictionary_size``.
        """
        exponents = self._build_dictionary(dictionary_degree)
        dict_size = len(exponents)

        psi_x_rows: list[np.ndarray] = []
        psi_y_rows: list[np.ndarray] = []

        for traj in trajectories:
            values = np.column_stack([
                traj.coupling_values[name]
                for name in self._coupling_names
            ])
            n_times = values.shape[0]

            for k in range(n_times - 1):
                psi_x_rows.append(
                    self._evaluate_dictionary(values[k], exponents)
                )
                psi_y_rows.append(
                    self._evaluate_dictionary(values[k + 1], exponents)
                )

        if not psi_x_rows:
            return {
                "eigenvalues": np.array([]),
                "eigenfunctions": np.array([]),
                "modes": np.array([]),
                "dictionary_size": dict_size,
            }

        Psi_X = np.array(psi_x_rows)
        Psi_Y = np.array(psi_y_rows)

        # Least-squares: K @ Psi_X^T ~ Psi_Y^T
        K, _, _, _ = np.linalg.lstsq(Psi_X, Psi_Y, rcond=None)

        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Sort by eigenvalue magnitude (largest first)
        sort_idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Koopman modes: project back to coupling observable space
        # (first n_couplings dictionary elements are the linear monomials)
        modes = eigenvectors[: self._n, :].T

        return {
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenvectors,
            "modes": modes,
            "dictionary_size": dict_size,
        }

    # ------------------------------------------------------------------
    # Dictionary construction
    # ------------------------------------------------------------------

    def _build_dictionary(self, degree: int) -> list[tuple[int, ...]]:
        """Multi-index tuples with ``sum <= degree``.

        Returns all tuples ``(a_1, ..., a_n)`` with non-negative
        entries whose sum does not exceed *degree*, ordered by total
        degree.
        """
        n = self._n
        result: list[tuple[int, ...]] = []
        for total_deg in range(degree + 1):
            result.extend(self._partitions(n, total_deg))
        return result

    @staticmethod
    def _partitions(n_vars: int, total: int) -> list[tuple[int, ...]]:
        """All n_vars-tuples of non-negative integers summing to total."""
        if n_vars == 1:
            return [(total,)]
        result: list[tuple[int, ...]] = []
        for first in range(total + 1):
            for rest in KoopmanOperator._partitions(n_vars - 1, total - first):
                result.append((first, *rest))
        return result

    # ------------------------------------------------------------------
    # Dictionary evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_dictionary(
        point: np.ndarray,
        exponents: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Evaluate all monomials at a point.

        For exponent tuple ``alpha = (a_1, ..., a_n)``, the monomial is
        ``prod_i  x_i^{a_i}``.

        Parameters
        ----------
        point:
            Coupling values, shape ``(n_couplings,)``.
        exponents:
            List of exponent tuples.

        Returns
        -------
        np.ndarray
            Monomial values, shape ``(dictionary_size,)``.
        """
        result = np.empty(len(exponents))
        for k, alpha in enumerate(exponents):
            result[k] = np.prod(point ** np.array(alpha))
        return result
