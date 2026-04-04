"""Oracle for Grover search that marks fixed-point states.

The oracle identifies computational-basis states ``|g>`` for which the
beta-function norm ``|beta(g)| < epsilon``.  In practice, the oracle is
built classically by pre-computing which grid points satisfy the
threshold condition.  The resulting circuit flips the phase of marked
states using multi-controlled Z gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.quantum.grover.encoding import CouplingGridEncoding


class BetaOracle:
    """Oracle that marks states ``|g>`` where ``|beta(g)| < epsilon``.

    Parameters
    ----------
    system:
        The RG beta-function system to evaluate.
    encoding:
        Grid encoding that maps qubit states to coupling values.
    epsilon:
        Threshold on the beta-function norm below which a point is
        considered an approximate fixed point.
    """

    def __init__(
        self,
        system: BetaFunctionSystem,
        encoding: CouplingGridEncoding,
        epsilon: float = 0.1,
    ) -> None:
        self._system = system
        self._encoding = encoding
        self._epsilon = epsilon
        self._marked_states: list[int] | None = None

    # ------------------------------------------------------------------
    # Classical pre-computation
    # ------------------------------------------------------------------

    def find_marked_states(self) -> list[int]:
        """Pre-compute which grid points satisfy ``|beta| < epsilon``.

        Evaluates the beta function at every grid point (classically)
        and returns the list of flat indices whose beta-function norm is
        below the threshold.
        """
        if self._marked_states is not None:
            return list(self._marked_states)

        marked: list[int] = []
        n_total = self._encoding.n_grid_points
        for idx in range(n_total):
            couplings = self._encoding.index_to_couplings(idx)
            try:
                beta_vals = self._system.evaluate(couplings)
                norm = np.sqrt(sum(v ** 2 for v in beta_vals.values()))
                if norm < self._epsilon:
                    marked.append(idx)
            except Exception:
                # Skip points where evaluation fails (e.g. singularity)
                continue

        self._marked_states = marked
        return list(marked)

    @property
    def n_marked(self) -> int:
        """Number of marked states (solutions)."""
        return len(self.find_marked_states())

    # ------------------------------------------------------------------
    # Oracle circuit
    # ------------------------------------------------------------------

    def build_oracle(self) -> Any:
        """Build the oracle circuit that flips phase of marked states.

        Uses multi-controlled Z gates for small grids.  Each marked
        state ``|m>`` gets a phase flip implemented as
        ``X`` on 0-bits -> multi-controlled-Z -> ``X`` undo.

        Returns
        -------
        QuantumCircuit
            Qiskit circuit implementing the oracle.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        marked = self.find_marked_states()
        n_qubits = self._encoding.n_qubits

        oracle_qc = QuantumCircuit(n_qubits, name="Oracle")

        for state_idx in marked:
            # Convert index to binary string of length n_qubits
            bitstring = format(state_idx, f"0{n_qubits}b")

            # Apply X gates to qubits where the bit is '0' so that the
            # multi-controlled Z fires only on |state_idx>.
            flip_qubits: list[int] = []
            for qubit_pos in range(n_qubits):
                # bit position: qubit 0 is the LSB (rightmost character)
                bit = bitstring[n_qubits - 1 - qubit_pos]
                if bit == "0":
                    oracle_qc.x(qubit_pos)
                    flip_qubits.append(qubit_pos)

            # Multi-controlled Z = H on target, MCX, H on target
            if n_qubits == 1:
                oracle_qc.z(0)
            else:
                target = n_qubits - 1
                controls = list(range(n_qubits - 1))
                oracle_qc.h(target)
                oracle_qc.mcx(controls, target)
                oracle_qc.h(target)

            # Undo X flips
            for qubit_pos in flip_qubits:
                oracle_qc.x(qubit_pos)

        return oracle_qc
