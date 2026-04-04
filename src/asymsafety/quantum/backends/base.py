"""Core quantum-backend protocol.

Mirrors the :class:`~asymsafety.compute.backends.base.ComputeBackend`
protocol but specialised for quantum circuit execution.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QuantumBackend(Protocol):
    """Protocol for quantum computing backends."""

    def execute_circuit(self, circuit: Any, shots: int = 1024) -> dict[str, int]:
        """Execute a quantum circuit, return measurement counts."""
        ...

    def statevector(self, circuit: Any) -> Any:
        """Get the statevector from a circuit (no measurement)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    def capabilities(self) -> set[str]:
        """Set of capability tags (e.g. ``"simulator"``, ``"statevector"``)."""
        ...
