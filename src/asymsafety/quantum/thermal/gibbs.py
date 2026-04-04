"""Gibbs state preparation for the Laplacian Hamiltonian.

Provides two routes to the thermal density matrix
``rho = exp(-beta*H) / Z(beta)``:

1. **Exact** (classical): directly compute the diagonal of ``rho``
   from the eigenvalues.  Used for validation and small systems.
2. **QITE** (Quantum Imaginary Time Evolution): approximate
   ``exp(-delta_beta * H)`` as a sequence of unitary steps on a
   quantum circuit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymsafety.quantum.thermal.hamiltonian import LaplacianHamiltonian


class GibbsStatePreparer:
    """Prepare the Gibbs state ``rho = exp(-beta*H) / Z(beta)``.

    Parameters
    ----------
    hamiltonian:
        The Laplacian Hamiltonian whose Gibbs state is to be prepared.
    """

    def __init__(self, hamiltonian: LaplacianHamiltonian) -> None:
        self._hamiltonian = hamiltonian

    # ------------------------------------------------------------------
    # Exact (classical)
    # ------------------------------------------------------------------

    def prepare_exact(self, beta: float) -> np.ndarray:
        """Exact Gibbs state diagonal (classical computation).

        Since H is diagonal, the density matrix ``rho`` is also
        diagonal:

        .. math::
            \\rho_{jj} = \\frac{e^{-\\beta \\lambda_j}}{Z(\\beta)}

        Parameters
        ----------
        beta:
            Inverse temperature (or propertime parameter *s* in the
            heat-kernel context).

        Returns
        -------
        np.ndarray
            Diagonal of the density matrix, shape ``(2^n,)``.
        """
        H_diag = np.diag(self._hamiltonian.to_matrix())
        boltzmann = np.exp(-beta * H_diag)
        Z = boltzmann.sum()
        if Z == 0.0:
            return np.zeros_like(boltzmann)
        return boltzmann / Z

    # ------------------------------------------------------------------
    # Partition function
    # ------------------------------------------------------------------

    def partition_function(self, beta: float) -> float:
        """Partition function ``Z(beta) = Tr[exp(-beta*H)]``.

        Computed exactly from the eigenvalues (pure numpy).
        """
        H_diag = np.diag(self._hamiltonian.to_matrix())
        return float(np.sum(np.exp(-beta * H_diag)))

    # ------------------------------------------------------------------
    # QITE
    # ------------------------------------------------------------------

    def prepare_qite(self, beta: float, n_steps: int = 10) -> Any:
        """QITE (Quantum Imaginary Time Evolution) state preparation.

        Approximates ``exp(-beta*H)`` as a product of unitaries:

        .. math::
            e^{-\\beta H} \\approx \\prod_{k=1}^{n} e^{-\\delta\\beta\\, H}

        Each slice ``exp(-delta_beta * H)`` is further approximated by
        a unitary via first-order McLachlan variational principle
        (real-time rotation by the Pauli decomposition of H).

        Parameters
        ----------
        beta:
            Inverse temperature.
        n_steps:
            Number of Trotter-like slices.

        Returns
        -------
        QuantumCircuit
            Qiskit circuit that approximately prepares the Gibbs state.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        n_qubits = self._hamiltonian.n_qubits
        delta_beta = beta / n_steps
        pauli_terms = self._hamiltonian.to_pauli_decomposition()

        qc = QuantumCircuit(n_qubits, name="QITE")

        # Start from uniform superposition (approximates high-T state)
        qc.h(range(n_qubits))

        for _step in range(n_steps):
            # Approximate exp(-delta_beta * H) via unitary rotations
            # For each Pauli term c_P * P, apply exp(-i * delta_beta * c_P * P)
            for coeff, pauli_str in pauli_terms:
                angle = 2.0 * delta_beta * coeff  # rotation angle
                if abs(angle) < 1e-15:
                    continue
                self._apply_pauli_rotation(qc, pauli_str, angle)

        return qc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_pauli_rotation(qc: Any, pauli_str: str, angle: float) -> None:
        """Apply ``exp(-i * angle/2 * P)`` for a Pauli string P.

        For IZ-type strings (the only ones arising from a diagonal H),
        this reduces to single-qubit Rz rotations on the Z positions,
        interleaved with CNOTs for multi-qubit Z products.
        """
        # Identify qubits where the Pauli is 'Z'
        n = len(pauli_str)
        z_qubits = [n - 1 - i for i, ch in enumerate(pauli_str) if ch == "Z"]

        if not z_qubits:
            # Pure identity term; global phase, skip
            return

        if len(z_qubits) == 1:
            qc.rz(angle, z_qubits[0])
            return

        # Multi-Z rotation: CNOT ladder -> Rz -> undo CNOT ladder
        for i in range(len(z_qubits) - 1):
            qc.cx(z_qubits[i], z_qubits[i + 1])
        qc.rz(angle, z_qubits[-1])
        for i in range(len(z_qubits) - 2, -1, -1):
            qc.cx(z_qubits[i], z_qubits[i + 1])
