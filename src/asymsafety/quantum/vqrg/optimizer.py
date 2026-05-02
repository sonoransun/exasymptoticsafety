"""Quantum natural gradient optimiser for VQRG.

Implements the update rule

    theta  <-  theta - lr * G^{-1} @ grad(C)

where ``G`` is the quantum Fisher information matrix and ``grad(C)``
is the parameter-shift gradient of the cost function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as la

from asymsafety.analysis.fixed_points import FixedPoint

if TYPE_CHECKING:
    from asymsafety.quantum.vqrg.cost import VQRGCostFunction
    from asymsafety.quantum.vqrg.fisher import QuantumFisherInformation

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a VQRG optimisation run.

    ``parameters_history`` records a snapshot of the circuit parameters at
    every iteration; consumed by
    ``asymsafety.quantum.visualization.vqrg_plots.plot_vqrg_optimization_trajectory``
    to draw the parameter-trajectory inset alongside the cost curve.
    """

    parameters: np.ndarray
    cost_history: list[float] = field(default_factory=list)
    parameters_history: list[np.ndarray] = field(default_factory=list)
    fixed_point: FixedPoint | None = None
    n_iterations: int = 0
    converged: bool = False


class QuantumNaturalGradient:
    """Optimisation using quantum natural gradient.

    ``theta <- theta - lr * G^{-1} @ grad(C)``

    where *G* is the quantum Fisher information matrix and *grad(C)* is
    the parameter-shift gradient.
    """

    def __init__(
        self,
        cost_fn: VQRGCostFunction,
        fisher: QuantumFisherInformation,
        learning_rate: float = 0.1,
        max_iterations: int = 200,
        tol: float = 1e-8,
        regularisation: float = 1e-4,
    ) -> None:
        """Initialise the optimiser.

        Args:
            cost_fn: VQRG cost function.
            fisher: Quantum Fisher information computer.
            learning_rate: Step size ``lr``.
            max_iterations: Maximum number of gradient steps.
            tol: Convergence tolerance on the cost function.
            regularisation: Tikhonov regularisation added to the QFI
                diagonal before inversion (``G + eps * I``).
        """
        self._cost_fn = cost_fn
        self._fisher = fisher
        self._lr = learning_rate
        self._max_iter = max_iterations
        self._tol = tol
        self._reg = regularisation

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self, initial_parameters: np.ndarray) -> OptimizationResult:
        """Run quantum natural gradient optimisation.

        At convergence, extracts the fixed point from the optimised
        circuit parameters.

        Args:
            initial_parameters: 1-D array of starting circuit parameters.

        Returns:
            :class:`OptimizationResult` with the final parameters,
            cost history, and (if converged) the extracted
            :class:`~asymsafety.analysis.fixed_points.FixedPoint`.
        """
        params = np.asarray(initial_parameters, dtype=float).copy()
        n_params = params.size
        cost_history: list[float] = []
        parameters_history: list[np.ndarray] = []

        converged = False

        for iteration in range(self._max_iter):
            cost = self._cost_fn.evaluate(params)
            cost_history.append(cost)
            parameters_history.append(params.copy())

            logger.debug(
                "Iteration %d / %d  |  cost = %.3e",
                iteration + 1,
                self._max_iter,
                cost,
            )

            if cost < self._tol:
                converged = True
                logger.info(
                    "Converged at iteration %d with cost %.3e",
                    iteration + 1,
                    cost,
                )
                break

            # Gradient via parameter-shift rule
            grad = self._cost_fn.gradient(params)

            # QFI matrix (+ Tikhonov regularisation)
            qfi = self._fisher.compute(params)
            qfi_reg = qfi + self._reg * np.eye(n_params)

            # Natural gradient step: G^{-1} @ grad
            try:
                nat_grad = la.solve(qfi_reg, grad, assume_a="pos")
            except la.LinAlgError:
                # Fall back to pseudo-inverse if QFI is singular
                nat_grad = la.lstsq(qfi_reg, grad)[0]

            params = params - self._lr * nat_grad

        else:
            # Reached max_iterations without convergence
            cost = self._cost_fn.evaluate(params)
            cost_history.append(cost)
            parameters_history.append(params.copy())
            converged = cost < self._tol

        fixed_point = self._extract_fixed_point(params) if converged else None

        return OptimizationResult(
            parameters=params,
            cost_history=cost_history,
            parameters_history=parameters_history,
            fixed_point=fixed_point,
            n_iterations=len(cost_history),
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Fixed-point extraction
    # ------------------------------------------------------------------

    def _extract_fixed_point(self, parameters: np.ndarray) -> FixedPoint:
        """Extract a :class:`FixedPoint` from optimised circuit parameters.

        1. Convert circuit parameters to coupling values via the
           :class:`CouplingAngleMap`.
        2. Compute the Hessian of the cost function.
        3. Eigendecompose the Hessian to obtain critical exponents.

        Args:
            parameters: Optimised circuit parameters.

        Returns:
            A :class:`FixedPoint` instance.
        """
        coupling_map = self._cost_fn.coupling_map
        system = self._cost_fn.system

        # 1. Get coupling values at the minimum
        z_exp = self._cost_fn.circuit.extract_expectation(parameters)
        angles = np.arccos(np.clip(z_exp, -1.0, 1.0))
        location = coupling_map.angles_to_couplings(angles)

        # 2. Compute stability matrix from the beta-function Jacobian
        jacobian = system.jacobian_numerical(location)
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        critical_exponents = -eigenvalues

        return FixedPoint(
            location=location,
            eigenvalues=eigenvalues,
            critical_exponents=critical_exponents,
            eigenvectors=eigenvectors,
        )
