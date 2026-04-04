"""Grover search for RG fixed points with quadratic speedup.

Workflow
--------
1. Prepare uniform superposition over a discretised coupling-space grid.
2. Apply  ``[Oracle -> Diffusion]`` for the optimal number of iterations.
3. Measure to obtain an approximate fixed-point location.
4. Refine the result with classical root-finding (``scipy.optimize.fsolve``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import fsolve

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.quantum.grover.encoding import CouplingGridEncoding
from asymsafety.quantum.grover.oracle import BetaOracle


@dataclass
class GroverSearchResult:
    """Result of a Grover-amplified fixed-point search.

    Attributes
    ----------
    measured_couplings:
        Raw coupling values decoded from the Grover measurement.
    refined_fixed_point:
        Fixed point obtained after classical refinement (``None`` if
        refinement failed).
    n_grover_iterations:
        Number of Grover iterations applied.
    success_probability:
        Theoretical success probability ``sin^2((2k+1) * arcsin(sqrt(M/N)))``.
    """

    measured_couplings: dict[str, float]
    refined_fixed_point: FixedPoint | None
    n_grover_iterations: int
    success_probability: float


class GroverFixedPointSearch:
    """Grover search for fixed points with ``sqrt(N)`` speedup.

    Parameters
    ----------
    system:
        The RG beta-function system.
    encoding:
        Grid encoding for the coupling space.
    oracle:
        Pre-built oracle that marks approximate fixed-point states.
    """

    def __init__(
        self,
        system: BetaFunctionSystem,
        encoding: CouplingGridEncoding,
        oracle: BetaOracle,
    ) -> None:
        self._system = system
        self._encoding = encoding
        self._oracle = oracle

    # ------------------------------------------------------------------
    # Iteration count
    # ------------------------------------------------------------------

    def optimal_iterations(self) -> int:
        r"""Optimal Grover iteration count.

        .. math::
            k = \lfloor \frac{\pi}{4} \sqrt{N / M} \rfloor

        where *N* is the total grid size and *M* the number of marked
        states.  Returns at least 1 when ``M > 0``.
        """
        n_total = self._encoding.n_grid_points
        n_marked = self._oracle.n_marked
        if n_marked == 0:
            return 0
        return max(1, math.floor(math.pi / 4 * math.sqrt(n_total / n_marked)))

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> Any:
        """Build the full Grover circuit.

        Structure::

            H^{⊗n}  --  [Oracle · Diffusion]^k  --  Measure

        Returns
        -------
        QuantumCircuit
            The complete Grover circuit with measurements.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        n_qubits = self._encoding.n_qubits
        n_iter = self.optimal_iterations()
        if n_iter == 0:
            raise ValueError(
                "No marked states found; cannot build Grover circuit."
            )

        oracle_gate = self._oracle.build_oracle().to_gate()
        diffusion_gate = self._build_diffusion(n_qubits).to_gate()

        qc = QuantumCircuit(n_qubits, n_qubits)

        # Uniform superposition
        qc.h(range(n_qubits))

        # Grover iterations
        for _ in range(n_iter):
            qc.append(oracle_gate, range(n_qubits))
            qc.append(diffusion_gate, range(n_qubits))

        # Measurement
        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    # ------------------------------------------------------------------
    # Search execution
    # ------------------------------------------------------------------

    def search(
        self,
        backend: Any = None,
        shots: int = 1024,
    ) -> GroverSearchResult:
        """Execute Grover search and refine the result classically.

        Parameters
        ----------
        backend:
            Qiskit backend (simulator or hardware).  When ``None``,
            uses :class:`qiskit.quantum_info.Statevector` for exact
            statevector simulation.
        shots:
            Number of measurement shots.

        Returns
        -------
        GroverSearchResult
        """
        n_iter = self.optimal_iterations()
        if n_iter == 0:
            return GroverSearchResult(
                measured_couplings={},
                refined_fixed_point=None,
                n_grover_iterations=0,
                success_probability=0.0,
            )

        qc = self.build_circuit()

        # --- Execute ---
        if backend is None:
            counts = self._run_statevector(qc, shots)
        else:
            counts = self._run_backend(qc, backend, shots)

        # Most frequent measurement
        best_bitstring = max(counts, key=counts.get)  # type: ignore[arg-type]
        measured_couplings = self._encoding.bitstring_to_couplings(best_bitstring)

        # Compute theoretical success probability
        success_prob = self._success_probability(n_iter)

        # Classical refinement via fsolve
        refined_fp = self._refine(measured_couplings)

        return GroverSearchResult(
            measured_couplings=measured_couplings,
            refined_fixed_point=refined_fp,
            n_grover_iterations=n_iter,
            success_probability=success_prob,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_diffusion(n_qubits: int) -> Any:
        """Build the Grover diffusion operator ``2|s><s| - I``."""
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        qc = QuantumCircuit(n_qubits, name="Diffusion")
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))

        # Multi-controlled Z
        if n_qubits == 1:
            qc.z(0)
        else:
            target = n_qubits - 1
            controls = list(range(n_qubits - 1))
            qc.h(target)
            qc.mcx(controls, target)
            qc.h(target)

        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    @staticmethod
    def _run_statevector(qc: Any, shots: int) -> dict[str, int]:
        """Exact statevector simulation with sampling."""
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        # Remove measurements for statevector sim, then sample
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        counts = sv.sample_counts(shots)
        return dict(counts)

    @staticmethod
    def _run_backend(qc: Any, backend: Any, shots: int) -> dict[str, int]:
        """Run on a qiskit backend (simulator or hardware)."""
        try:
            from qiskit import transpile
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        return dict(result.get_counts())

    def _success_probability(self, n_iter: int) -> float:
        r"""Theoretical Grover success probability.

        .. math::
            P = \sin^2\!\bigl((2k+1)\,\arcsin\!\sqrt{M/N}\bigr)
        """
        n_total = self._encoding.n_grid_points
        n_marked = self._oracle.n_marked
        if n_marked == 0 or n_total == 0:
            return 0.0
        theta = math.asin(math.sqrt(n_marked / n_total))
        return math.sin((2 * n_iter + 1) * theta) ** 2

    def _refine(self, measured: dict[str, float]) -> FixedPoint | None:
        """Refine the Grover measurement with classical root-finding."""
        names = self._system.coupling_names
        rhs = self._system.rhs_vector()

        y0 = np.array([measured.get(name, 0.0) for name in names])

        def residual(y: np.ndarray) -> list[float]:
            return rhs(0, y)

        try:
            sol, info, ier, _msg = fsolve(residual, y0, full_output=True)
            if ier != 1:
                return None
            max_residual = float(np.max(np.abs(info["fvec"])))
            if max_residual > 1e-6:
                return None

            location = {name: float(sol[i]) for i, name in enumerate(names)}

            # Compute stability information
            jac = self._system.jacobian_numerical(location)
            eigenvalues, eigenvectors = np.linalg.eig(jac)
            critical_exponents = -eigenvalues

            return FixedPoint(
                location=location,
                eigenvalues=eigenvalues,
                critical_exponents=critical_exponents,
                eigenvectors=eigenvectors,
            )
        except Exception:
            return None
