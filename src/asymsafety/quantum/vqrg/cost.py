"""VQRG cost function and its derivatives.

The cost function is

    C(theta) = (1/2) sum_i beta_i(g(theta))^2

where g(theta) is obtained from the circuit expectation values via a
:class:`~asymsafety.quantum.vqrg.mapping.CouplingAngleMap`.

Fixed points of the RG flow correspond to global minima C = 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.beta.system import BetaFunctionSystem
    from asymsafety.quantum.vqrg.circuit import VQRGCircuit
    from asymsafety.quantum.vqrg.mapping import CouplingAngleMap


class VQRGCostFunction:
    """Cost function C(theta) = (1/2) sum_i beta_i(g(theta))^2.

    Fixed points correspond to C = 0.
    """

    def __init__(
        self,
        system: BetaFunctionSystem,
        coupling_map: CouplingAngleMap,
        circuit: VQRGCircuit,
    ) -> None:
        self._system = system
        self._coupling_map = coupling_map
        self._circuit = circuit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def system(self) -> BetaFunctionSystem:
        """The underlying beta-function system."""
        return self._system

    @property
    def coupling_map(self) -> CouplingAngleMap:
        """The coupling-to-angle map."""
        return self._coupling_map

    @property
    def circuit(self) -> VQRGCircuit:
        """The variational circuit."""
        return self._circuit

    def _expectations_to_couplings(
        self, parameters: np.ndarray
    ) -> dict[str, float]:
        """Convert circuit expectations <Z_i> to coupling values.

        The mapping is:  theta_i = arccos(<Z_i>)  in [0, pi],
        then theta_i is converted to g_i via the CouplingAngleMap.
        """
        z_expectations = self._circuit.extract_expectation(parameters)
        # Map <Z_i> in [-1, 1] to theta_i in [0, pi]
        angles = np.arccos(np.clip(z_expectations, -1.0, 1.0))
        return self._coupling_map.angles_to_couplings(angles)

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def evaluate(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function at given circuit parameters.

        Args:
            parameters: 1-D array of circuit parameters.

        Returns:
            Scalar cost ``C(theta) = (1/2) sum_i beta_i^2``.
        """
        couplings = self._expectations_to_couplings(parameters)
        betas = self._system.evaluate(couplings)
        beta_vals = np.array([betas[n] for n in self._coupling_map.coupling_names])
        return 0.5 * float(np.dot(beta_vals, beta_vals))

    # ------------------------------------------------------------------
    # Parameter-shift rule gradient
    # ------------------------------------------------------------------

    def gradient(
        self,
        parameters: np.ndarray,
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """Parameter-shift rule gradient.

        .. math::

            \\frac{\\partial C}{\\partial \\theta_k}
            = \\frac{C(\\theta_k + \\pi/2) - C(\\theta_k - \\pi/2)}{2}

        Args:
            parameters: 1-D array of circuit parameters.
            shift: Shift amount (default ``pi/2`` for standard gates).

        Returns:
            Gradient array of shape ``(n_parameters,)``.
        """
        params = np.asarray(parameters, dtype=float)
        n_params = params.size
        grad = np.empty(n_params)

        for k in range(n_params):
            shifted_plus = params.copy()
            shifted_plus[k] += shift

            shifted_minus = params.copy()
            shifted_minus[k] -= shift

            c_plus = self.evaluate(shifted_plus)
            c_minus = self.evaluate(shifted_minus)
            grad[k] = (c_plus - c_minus) / (2 * np.sin(shift))

        return grad

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------

    def hessian(
        self,
        parameters: np.ndarray,
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """Hessian of the cost function.

        At a fixed point (``C = 0``) the Hessian factorises as
        ``H = M^T M`` where ``M`` is the stability matrix; its
        eigenvalues give ``|theta_i|^2``.

        Uses the parameter-shift rule applied twice:

        .. math::

            H_{jk} = \\frac{1}{4 \\sin^2 s}
            \\bigl[
              C(+s_j, +s_k) - C(+s_j, -s_k) - C(-s_j, +s_k) + C(-s_j, -s_k)
            \\bigr]

        Args:
            parameters: 1-D array of circuit parameters.
            shift: Shift amount.

        Returns:
            Hessian matrix of shape ``(n_parameters, n_parameters)``.
        """
        params = np.asarray(parameters, dtype=float)
        n_params = params.size
        hess = np.empty((n_params, n_params))
        sin2 = np.sin(shift) ** 2

        for j in range(n_params):
            for k in range(j, n_params):
                p_pp = params.copy()
                p_pp[j] += shift
                p_pp[k] += shift

                p_pm = params.copy()
                p_pm[j] += shift
                p_pm[k] -= shift

                p_mp = params.copy()
                p_mp[j] -= shift
                p_mp[k] += shift

                p_mm = params.copy()
                p_mm[j] -= shift
                p_mm[k] -= shift

                val = (
                    self.evaluate(p_pp)
                    - self.evaluate(p_pm)
                    - self.evaluate(p_mp)
                    + self.evaluate(p_mm)
                ) / (4.0 * sin2)

                hess[j, k] = val
                hess[k, j] = val

        return hess
