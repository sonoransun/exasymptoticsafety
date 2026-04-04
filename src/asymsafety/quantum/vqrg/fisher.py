"""Quantum Fisher Information matrix.

The QFI matrix serves as the Zamolodchikov metric on the space of
variational parameters and, when transformed to coupling space, yields
the Zamolodchikov metric on the theory space.

.. math::

    G_{ij} = 4 \\,\\mathrm{Re}\\bigl[
        \\langle \\partial_i \\psi | \\partial_j \\psi \\rangle
        - \\langle \\partial_i \\psi | \\psi \\rangle
          \\langle \\psi | \\partial_j \\psi \\rangle
    \\bigr]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.quantum.vqrg.circuit import VQRGCircuit
    from asymsafety.quantum.vqrg.mapping import CouplingAngleMap


class QuantumFisherInformation:
    """Quantum Fisher Information matrix -> Zamolodchikov metric.

    .. math::

        G_{ij} = 4 \\,\\mathrm{Re}\\bigl[
            \\langle \\partial_i \\psi | \\partial_j \\psi \\rangle
            - \\langle \\partial_i \\psi | \\psi \\rangle
              \\langle \\psi | \\partial_j \\psi \\rangle
        \\bigr]
    """

    def __init__(self, circuit: VQRGCircuit) -> None:
        self._circuit = circuit

    @property
    def circuit(self) -> VQRGCircuit:
        return self._circuit

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        parameters: np.ndarray,
        shift: float = 0.01,
    ) -> np.ndarray:
        """Compute the QFI matrix using finite differences on the statevector.

        The derivative is approximated as:

        .. math::

            |\\partial_i \\psi\\rangle \\approx
            \\frac{|\\psi(\\theta + e_i h)\\rangle
                  - |\\psi(\\theta - e_i h)\\rangle}{2h}

        Args:
            parameters: 1-D array of circuit parameters.
            shift: Step size *h* for finite differences.

        Returns:
            QFI matrix of shape ``(n_params, n_params)``.
        """
        params = np.asarray(parameters, dtype=float)
        n_params = params.size

        psi = self._circuit.get_statevector(params)

        # Compute all partial derivatives |d_i psi>
        dpsi = np.empty((n_params, psi.size), dtype=complex)
        for i in range(n_params):
            p_plus = params.copy()
            p_plus[i] += shift
            p_minus = params.copy()
            p_minus[i] -= shift

            psi_plus = self._circuit.get_statevector(p_plus)
            psi_minus = self._circuit.get_statevector(p_minus)
            dpsi[i] = (psi_plus - psi_minus) / (2.0 * shift)

        # G_ij = 4 Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
        qfi = np.empty((n_params, n_params))
        for i in range(n_params):
            for j in range(i, n_params):
                overlap_ij = np.vdot(dpsi[i], dpsi[j])           # <d_i|d_j>
                overlap_i_psi = np.vdot(dpsi[i], psi)            # <d_i|psi>
                overlap_psi_j = np.vdot(psi, dpsi[j])            # <psi|d_j>

                g_ij = 4.0 * np.real(
                    overlap_ij - overlap_i_psi * overlap_psi_j
                )
                qfi[i, j] = g_ij
                qfi[j, i] = g_ij

        return qfi

    # ------------------------------------------------------------------
    # Transformation to coupling space
    # ------------------------------------------------------------------

    def to_coupling_space(
        self,
        qfi: np.ndarray,
        coupling_map: CouplingAngleMap,
        parameters: np.ndarray,  # noqa: ARG002 -- reserved for non-linear maps
    ) -> np.ndarray:
        """Transform QFI from parameter space to coupling space.

        For the linear map ``theta = f(g)`` the Zamolodchikov metric is:

        .. math::

            G^{\\text{coupling}}_{ab}
            = J^T_{ai} \\, G^{\\theta}_{ij} \\, J_{jb}

        where ``J = d(theta)/d(g)`` is the Jacobian of the
        :class:`CouplingAngleMap`.

        Note: this truncates the QFI to the first ``n_couplings``
        parameters (the ones that directly encode couplings).  Higher
        layers' parameters contribute indirectly through the circuit
        but are not part of the coupling-space metric.

        Args:
            qfi: QFI matrix in parameter space, shape
                ``(n_params, n_params)``.
            coupling_map: The coupling-angle mapping.
            parameters: Circuit parameters (reserved for future
                non-linear maps; currently unused).

        Returns:
            Zamolodchikov metric in coupling space, shape
            ``(n_couplings, n_couplings)``.
        """
        n_c = coupling_map.n_couplings
        # Jacobian d(theta)/d(g)
        j_mat = coupling_map.jacobian_couplings_to_angles()  # (n_c, n_c)

        # Extract the coupling-relevant block of the QFI
        qfi_block = qfi[:n_c, :n_c]

        # G_coupling = J^T @ G_theta @ J
        return j_mat.T @ qfi_block @ j_mat
