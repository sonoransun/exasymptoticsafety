"""Hamiltonian simulation of exp(iMt) via Trotterisation.

For the Einstein-Hilbert truncation the stability matrix ``M`` is 2x2,
requiring only a single qubit (``M = aI + bX + cY + dZ``).  For
quadratic gravity (4x4 ``M``) two qubits suffice and the Suzuki-Trotter
product formula is used.

Quantum Phase Estimation (QPE) can then extract the eigenvalues of
``M``, which should agree with the critical exponents obtained
classically.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.analysis.stability import StabilityAnalysis


class StabilityMatrixSimulator:
    """Hamiltonian simulation of ``exp(iMt)`` for the stability matrix.

    Parameters
    ----------
    stability:
        A completed :class:`StabilityAnalysis`.
    """

    def __init__(self, stability: StabilityAnalysis) -> None:
        self._stability = stability
        self._M = stability.stability_matrix.copy()
        self._n = self._M.shape[0]
        self._n_qubits = math.ceil(math.log2(max(self._n, 2)))

    # ------------------------------------------------------------------
    # Pauli decomposition
    # ------------------------------------------------------------------

    def pauli_decomposition(self) -> list[tuple[complex, str]]:
        """Decompose M into the Pauli basis.

        .. math::
            M = \\sum_P c_P \\, P, \\qquad
            c_P = \\frac{1}{2^n} \\operatorname{Tr}(P \\cdot M)

        where ``P`` ranges over all ``n``-qubit Pauli strings.

        Returns
        -------
        list of (coefficient, pauli_string) pairs
            Only terms with ``|c_P| > 1e-15`` are included.
        """
        n = self._n_qubits
        dim = 1 << n

        # Pad M to dim x dim if necessary
        M_padded = np.zeros((dim, dim), dtype=complex)
        M_padded[: self._n, : self._n] = self._M

        # Pauli matrices
        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        paulis = {"I": I2, "X": X, "Y": Y, "Z": Z}
        pauli_labels = ["I", "X", "Y", "Z"]

        terms: list[tuple[complex, str]] = []

        # Iterate over all 4^n Pauli strings
        for idx in range(4 ** n):
            # Decode the index into a Pauli string
            label_chars: list[str] = []
            remainder = idx
            matrices: list[np.ndarray] = []
            for _q in range(n):
                p_idx = remainder % 4
                remainder //= 4
                label_chars.append(pauli_labels[p_idx])
                matrices.append(paulis[pauli_labels[p_idx]])

            # Tensor product (qubit 0 is rightmost in the string)
            label_chars.reverse()
            pauli_str = "".join(label_chars)

            # Build full Pauli matrix via Kronecker product
            P = matrices[-1]
            for k in range(len(matrices) - 2, -1, -1):
                P = np.kron(matrices[k], P)

            coeff = np.trace(P @ M_padded) / dim

            if abs(coeff) > 1e-15:
                terms.append((complex(coeff), pauli_str))

        return terms

    # ------------------------------------------------------------------
    # Trotter circuit
    # ------------------------------------------------------------------

    def build_trotter_circuit(self, t: float, n_trotter: int = 10) -> Any:
        """Build a Trotterised circuit for ``exp(iMt)``.

        .. math::
            e^{iMt} \\approx \\Bigl[\\prod_P e^{i c_P P \\,t/n}\\Bigr]^n

        Parameters
        ----------
        t:
            Evolution time.
        n_trotter:
            Number of Trotter steps.

        Returns
        -------
        QuantumCircuit
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        n_qubits = self._n_qubits
        pauli_terms = self.pauli_decomposition()
        dt = t / n_trotter

        qc = QuantumCircuit(n_qubits, name="Trotter_exp(iMt)")

        for _step in range(n_trotter):
            for coeff, pauli_str in pauli_terms:
                angle = 2.0 * dt * coeff.real  # rotation angle
                if abs(angle) < 1e-15:
                    continue
                self._apply_pauli_rotation(qc, pauli_str, angle)

        return qc

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def simulate_evolution(
        self,
        t: float,
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate ``exp(iMt)|initial>`` and return the final state.

        Parameters
        ----------
        t:
            Evolution time.
        initial_state:
            Initial statevector.  If ``None``, uses ``|0...0>``.

        Returns
        -------
        np.ndarray
            Final statevector.
        """
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        n_qubits = self._n_qubits
        dim = 1 << n_qubits

        if initial_state is None:
            init = np.zeros(dim, dtype=complex)
            init[0] = 1.0
        else:
            init = np.zeros(dim, dtype=complex)
            init[: len(initial_state)] = initial_state
            norm = np.linalg.norm(init)
            if norm > 1e-30:
                init /= norm

        sv = Statevector(init)
        circuit = self.build_trotter_circuit(t)
        sv = sv.evolve(circuit)
        return np.array(sv.data)

    # ------------------------------------------------------------------
    # QPE eigenvalue extraction
    # ------------------------------------------------------------------

    def extract_eigenvalues_qpe(self, n_ancilla: int = 4) -> np.ndarray:
        """Use Quantum Phase Estimation to extract eigenvalues of M.

        The eigenvalues of ``M`` appear as phases in the unitary
        ``U = exp(2*pi*i*M / norm)``.  QPE reads those phases into the
        ancilla register.

        Parameters
        ----------
        n_ancilla:
            Number of ancilla qubits (determines precision).

        Returns
        -------
        np.ndarray
            Estimated eigenvalues of M.
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        n_sys = self._n_qubits
        n_total = n_ancilla + n_sys
        dim_sys = 1 << n_sys

        # Normalise M so that eigenvalues fit in [0, 1)
        M_padded = np.zeros((dim_sys, dim_sys), dtype=complex)
        M_padded[: self._n, : self._n] = self._M
        eig_max = max(abs(np.linalg.eigvals(M_padded).real.max()), 1e-10)
        scale = 2.0 * eig_max  # eigenphases = eigenvalue / scale
        t_base = 2.0 * np.pi / scale

        # Build QPE circuit
        qc = QuantumCircuit(n_total, n_ancilla)

        # Hadamard on ancilla qubits
        for a in range(n_ancilla):
            qc.h(a)

        # Controlled-U^{2^k} for each ancilla qubit k
        for k in range(n_ancilla):
            power = 1 << k
            t_val = t_base * power
            trotter = self.build_trotter_circuit(t_val, n_trotter=max(10, power))
            ctrl_gate = trotter.to_gate().control(1)
            qc.append(ctrl_gate, [k] + list(range(n_ancilla, n_total)))

        # Inverse QFT on ancilla
        self._inverse_qft(qc, list(range(n_ancilla)))

        # Measure ancilla
        qc.measure(range(n_ancilla), range(n_ancilla))

        # Simulate
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = np.abs(sv.data) ** 2

        # Extract eigenvalue estimates from the measurement probabilities
        n_phases = 1 << n_ancilla
        phase_probs = np.zeros(n_phases)
        dim_total = 1 << n_total

        for state_idx in range(dim_total):
            ancilla_val = state_idx % n_phases
            phase_probs[ancilla_val] += probs[state_idx]

        # Find peaks
        threshold = 1.0 / (2 * n_phases)
        peak_indices = np.where(phase_probs > threshold)[0]

        estimated_eigenvalues: list[float] = []
        for idx in peak_indices:
            phase = idx / n_phases
            eigenvalue = phase * scale
            estimated_eigenvalues.append(eigenvalue)

        return np.array(estimated_eigenvalues)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_pauli_rotation(qc: Any, pauli_str: str, angle: float) -> None:
        """Apply ``exp(-i * angle/2 * P)`` for a Pauli string P.

        Handles arbitrary IZ, IX, IY, ... strings by rotating into the
        Z basis, applying a CNOT-ladder Rz rotation, and rotating back.
        """
        n = len(pauli_str)

        # Identify active qubits and their Pauli type
        active_qubits: list[int] = []
        pauli_types: list[str] = []
        for i, ch in enumerate(pauli_str):
            qubit = n - 1 - i  # MSB convention
            if ch != "I":
                active_qubits.append(qubit)
                pauli_types.append(ch)

        if not active_qubits:
            return

        # Rotate into Z basis
        for qubit, ptype in zip(active_qubits, pauli_types):
            if ptype == "X":
                qc.h(qubit)
            elif ptype == "Y":
                qc.sdg(qubit)
                qc.h(qubit)
            # Z and I need no basis change

        # CNOT ladder for multi-qubit Pauli
        if len(active_qubits) == 1:
            qc.rz(angle, active_qubits[0])
        else:
            for i in range(len(active_qubits) - 1):
                qc.cx(active_qubits[i], active_qubits[i + 1])
            qc.rz(angle, active_qubits[-1])
            for i in range(len(active_qubits) - 2, -1, -1):
                qc.cx(active_qubits[i], active_qubits[i + 1])

        # Undo basis rotation
        for qubit, ptype in zip(active_qubits, pauli_types):
            if ptype == "X":
                qc.h(qubit)
            elif ptype == "Y":
                qc.h(qubit)
                qc.s(qubit)

    @staticmethod
    def _inverse_qft(qc: Any, qubits: list[int]) -> None:
        """Apply inverse QFT on the given qubits in-place."""
        n = len(qubits)
        # Swap
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        # Inverse rotations
        for target_idx in range(n - 1, -1, -1):
            for control_idx in range(n - 1, target_idx, -1):
                k = control_idx - target_idx + 1
                qc.cp(-2 * np.pi / (1 << k), qubits[control_idx], qubits[target_idx])
            qc.h(qubits[target_idx])
