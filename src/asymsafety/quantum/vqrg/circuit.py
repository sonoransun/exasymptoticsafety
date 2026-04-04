"""Parameterized variational circuit for RG flow.

For Einstein--Hilbert (n = 2 couplings):
    - 2 qubits, L layers, 5 parameters per layer
    - Each layer:
        Ry(theta_0) + Ry(theta_1)  -->  CNOT(0,1)  -->  CNOT(1,0)
        -->  Rz(phi)  -->  Ry(alpha)

All qiskit imports are lazy (inside method bodies), following the
JAX-evaluator pattern from ``asymsafety.compute.batch.jax_evaluator``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VQRGCircuit:
    """Parameterized variational circuit for RG flow.

    The ansatz consists of *n_layers* identical layers.  Within each
    layer the structure is:

    1. ``Ry`` rotations on every qubit (``n_qubits`` parameters).
    2. Entangling ``CNOT`` ladder: ``CNOT(i, i+1)`` for even *i*,
       then ``CNOT(i, i+1)`` for odd *i*, plus a wrap-around
       ``CNOT(n-1, 0)`` when ``n_qubits > 1``.
    3. ``Rz`` rotation on qubit 0 (1 parameter).
    4. ``Ry`` rotations on every qubit (``n_qubits`` parameters).

    Total parameters per layer: ``2 * n_qubits + 1``.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 10) -> None:
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._params_per_layer = 2 * n_qubits + 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_parameters(self) -> int:
        """Total number of variational parameters."""
        return self._n_layers * self._params_per_layer

    # ------------------------------------------------------------------
    # Circuit construction (lazy qiskit import)
    # ------------------------------------------------------------------

    def build_circuit(self, parameters: np.ndarray) -> Any:
        """Build a qiskit ``QuantumCircuit`` with the given parameters.

        Args:
            parameters: 1-D array of length :attr:`n_parameters`.

        Returns:
            A ``qiskit.QuantumCircuit`` instance.

        Raises:
            ImportError: If qiskit is not installed.
            ValueError: If *parameters* has the wrong length.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        params = np.asarray(parameters, dtype=float)
        if params.size != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, got {params.size}"
            )

        qc = QuantumCircuit(self._n_qubits)
        idx = 0

        for _ in range(self._n_layers):
            # --- Ry rotations on each qubit ---
            for q in range(self._n_qubits):
                qc.ry(params[idx], q)
                idx += 1

            # --- Entangling CNOT ladder ---
            if self._n_qubits > 1:
                for q in range(0, self._n_qubits - 1, 2):
                    qc.cx(q, q + 1)
                for q in range(1, self._n_qubits - 1, 2):
                    qc.cx(q, q + 1)
                qc.cx(self._n_qubits - 1, 0)

            # --- Rz on qubit 0 ---
            qc.rz(params[idx], 0)
            idx += 1

            # --- Ry rotations on each qubit ---
            for q in range(self._n_qubits):
                qc.ry(params[idx], q)
                idx += 1

        return qc

    # ------------------------------------------------------------------
    # Statevector helpers
    # ------------------------------------------------------------------

    def get_statevector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector for given parameters.

        Uses ``qiskit.quantum_info.Statevector`` for exact simulation
        (no sampling noise).

        Args:
            parameters: 1-D array of length :attr:`n_parameters`.

        Returns:
            Complex state vector of length ``2**n_qubits``.
        """
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        qc = self.build_circuit(parameters)
        sv = Statevector.from_instruction(qc)
        return np.asarray(sv.data, dtype=complex)

    def extract_expectation(self, parameters: np.ndarray) -> np.ndarray:
        """Extract coupling-encoding expectation values from the circuit.

        Computes ``<Z_i>`` for each qubit *i*.  These expectation values
        lie in ``[-1, 1]`` and are later mapped through a
        :class:`~asymsafety.quantum.vqrg.mapping.CouplingAngleMap` to
        recover coupling constants.

        Args:
            parameters: 1-D array of length :attr:`n_parameters`.

        Returns:
            Array of ``n_qubits`` expectation values.
        """
        sv = self.get_statevector(parameters)
        probs = np.abs(sv) ** 2

        expectations = np.empty(self._n_qubits)
        for q in range(self._n_qubits):
            # Probability that qubit q is |0>
            p0 = 0.0
            for basis_state in range(2**self._n_qubits):
                # qiskit uses big-endian: qubit 0 is the most significant bit
                bit = (basis_state >> (self._n_qubits - 1 - q)) & 1
                if bit == 0:
                    p0 += probs[basis_state]
            # <Z_q> = P(0) - P(1) = 2*P(0) - 1
            expectations[q] = 2 * p0 - 1.0

        return expectations
