"""Qiskit Aer simulator backend.

All qiskit imports are lazy -- only loaded when methods are first called.
"""

from __future__ import annotations

from typing import Any


class AerSimulatorBackend:
    """Qiskit Aer simulator backend.

    All qiskit imports are lazy -- only loaded when methods are called.
    This follows the same pattern used by
    :class:`~asymsafety.compute.batch.jax_evaluator.JaxBatchEvaluator`.
    """

    def __init__(self) -> None:
        self._backend: Any | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        """Import and instantiate ``AerSimulator`` on first use."""
        if self._backend is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as exc:
                raise ImportError(
                    "Qiskit Aer is required for the simulator backend. "
                    "Install with: pip install asymsafety[quantum]"
                ) from exc
            self._backend = AerSimulator()

    # ------------------------------------------------------------------
    # QuantumBackend interface
    # ------------------------------------------------------------------

    def execute_circuit(self, circuit: Any, shots: int = 1024) -> dict[str, int]:
        """Execute *circuit* and return bitstring counts.

        The circuit must already contain measurements.  If it does not,
        a measurement of all qubits is appended automatically.
        """
        self._ensure_backend()

        try:
            from qiskit import transpile
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        transpiled = transpile(circuit, self._backend)
        result = self._backend.run(transpiled, shots=shots).result()
        return dict(result.get_counts())

    def statevector(self, circuit: Any) -> Any:
        """Return the statevector produced by *circuit*.

        Uses :class:`qiskit.quantum_info.Statevector` for an exact
        computation (no sampling noise).
        """
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        sv = Statevector.from_instruction(circuit)
        return sv

    @property
    def name(self) -> str:
        return "aer_simulator"

    @property
    def capabilities(self) -> set[str]:
        return {"simulator", "statevector"}
