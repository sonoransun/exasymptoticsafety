"""Entanglement measures for quantum RG flow circuits.

All functions are pure NumPy / SciPy -- no qiskit dependency.

The von Neumann entropy of a subsystem maps to the anomalous dimension
eta_N at the fixed-point circuit.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la


def reduced_density_matrix(
    statevector: np.ndarray,
    n_qubits: int,
    keep_qubits: list[int],
) -> np.ndarray:
    """Compute the reduced density matrix by tracing out complement qubits.

    Given a pure state ``|psi>`` on ``n_qubits`` qubits, compute

        rho_A = Tr_B(|psi><psi|)

    where *A* consists of the qubits listed in *keep_qubits* and *B*
    is the complement.

    Args:
        statevector: Complex state vector of length ``2**n_qubits``.
        n_qubits: Total number of qubits.
        keep_qubits: Indices of the qubits to *keep* (subsystem A).

    Returns:
        Reduced density matrix of shape ``(2**n_A, 2**n_A)`` where
        ``n_A = len(keep_qubits)``.
    """
    psi = np.asarray(statevector, dtype=complex).reshape([2] * n_qubits)

    # Determine qubits to trace out (complement of keep_qubits)
    all_qubits = list(range(n_qubits))
    trace_qubits = sorted(set(all_qubits) - set(keep_qubits))

    n_keep = len(keep_qubits)
    n_trace = len(trace_qubits)

    # Reorder axes so that keep_qubits come first, trace_qubits last
    perm = list(keep_qubits) + trace_qubits
    psi_perm = np.transpose(psi, perm)

    # Reshape into (dim_A, dim_B)
    dim_a = 2**n_keep
    dim_b = 2**n_trace
    psi_bipartite = psi_perm.reshape(dim_a, dim_b)

    # rho_A = Tr_B(|psi><psi|) = sum_j (psi @ psi^dagger) over B indices
    rho = psi_bipartite @ psi_bipartite.conj().T

    return rho


def von_neumann_entropy(
    statevector: np.ndarray,
    n_qubits: int,
    subsystem_qubits: list[int],
) -> float:
    """Compute von Neumann entropy S(rho_A) of a subsystem.

    The procedure is:
        1. Reshape the statevector into bipartite form.
        2. Compute reduced density matrix ``rho_A = Tr_B(|psi><psi|)``.
        3. ``S = -Tr(rho_A log2(rho_A))``.

    This maps to the anomalous dimension ``eta_N`` at the fixed-point
    circuit.

    Args:
        statevector: Complex state vector of length ``2**n_qubits``.
        n_qubits: Total number of qubits.
        subsystem_qubits: Indices of the qubits forming subsystem A.

    Returns:
        Von Neumann entropy ``S(rho_A)`` in bits (log base 2).
    """
    rho = reduced_density_matrix(statevector, n_qubits, subsystem_qubits)

    # Eigenvalues of rho (Hermitian, so use eigh for stability)
    eigenvalues = la.eigvalsh(rho)

    # Filter out zero / negligible eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    # S = -sum(lambda_i * log2(lambda_i))
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)
