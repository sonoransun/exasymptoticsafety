"""Tests for thermal state / heat kernel (no qiskit required)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.geometry.decomposition import ModeSpectrum
from asymsafety.quantum.thermal.hamiltonian import LaplacianHamiltonian
from asymsafety.quantum.thermal.gibbs import GibbsStatePreparer
from asymsafety.quantum.thermal.partition import PartitionFunctionEstimator


@pytest.fixture
def spectrum():
    return ModeSpectrum(d=4)


@pytest.fixture
def hamiltonian(spectrum):
    return LaplacianHamiltonian(spectrum, field_type="scalar", l_max=3)


@pytest.fixture
def gibbs(hamiltonian):
    return GibbsStatePreparer(hamiltonian)


# ---------------------------------------------------------------------------
# LaplacianHamiltonian
# ---------------------------------------------------------------------------

def test_laplacian_hamiltonian_eigenvalues(hamiltonian, spectrum):
    evals = hamiltonian.eigenvalues()
    # Check that eigenvalues match the spectrum for scalar field on S^4
    import sympy
    a_sq = sympy.Rational(1)
    for i, l_val in enumerate(hamiltonian._l_values):
        expected = float(spectrum.eigenvalue("scalar", l_val, a_sq))
        assert_allclose(evals[i], expected, rtol=1e-10)


def test_laplacian_hamiltonian_matrix_shape(hamiltonian):
    H = hamiltonian.to_matrix()
    n_qubits = hamiltonian.n_qubits
    dim = 2**n_qubits
    assert H.shape == (dim, dim)


def test_pauli_decomposition(hamiltonian):
    H = hamiltonian.to_matrix()
    terms = hamiltonian.to_pauli_decomposition()
    assert len(terms) > 0

    # Reconstruct H from Pauli decomposition
    n = hamiltonian.n_qubits
    dim = 2**n
    H_reconstructed = np.zeros((dim, dim), dtype=complex)

    I2 = np.eye(2, dtype=complex)
    Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

    for coeff, pauli_str in terms:
        P = np.array([[1.0]], dtype=complex)
        for ch in pauli_str:
            if ch == "I":
                P = np.kron(P, I2)
            elif ch == "Z":
                P = np.kron(P, Z2)
        H_reconstructed += coeff * P

    assert_allclose(H_reconstructed.real, H, atol=1e-10)


# ---------------------------------------------------------------------------
# Gibbs state
# ---------------------------------------------------------------------------

def test_gibbs_state_normalization(gibbs):
    rho_diag = gibbs.prepare_exact(beta=1.0)
    # Sum of diagonal elements should be 1
    assert_allclose(np.sum(rho_diag), 1.0, rtol=1e-10)


def test_partition_function_positive(gibbs):
    for beta in [0.1, 0.5, 1.0, 2.0]:
        Z = gibbs.partition_function(beta)
        assert Z > 0, f"Z({beta}) = {Z} is not positive"


# ---------------------------------------------------------------------------
# Partition function estimator
# ---------------------------------------------------------------------------

def test_partition_function_estimator(gibbs):
    estimator = PartitionFunctionEstimator(gibbs, d=4)
    coeffs = estimator.estimate_coefficients(
        beta_values=np.linspace(0.01, 0.5, 15),
        n_coeffs=2,
    )
    assert "b0" in coeffs
    assert "b2" in coeffs
    assert np.isfinite(coeffs["b0"])
    assert np.isfinite(coeffs["b2"])
