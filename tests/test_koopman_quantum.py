"""Tests for quantum Koopman operator (no qiskit required)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.analysis.flow import FlowIntegrator
from asymsafety.analysis.stability import analyze_stability
from asymsafety.quantum.koopman.operator import KoopmanOperator
from asymsafety.quantum.koopman.matrix_exp import StabilityMatrixSimulator


@pytest.fixture
def eh_system():
    return build_eh_beta_system(d=4)


@pytest.fixture
def ngfp(eh_system):
    finder = FixedPointFinder(eh_system)
    fp = finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
    assert fp is not None, "NGFP search failed"
    return fp


@pytest.fixture
def stability(eh_system, ngfp):
    return analyze_stability(eh_system, ngfp)


@pytest.fixture
def trajectories(eh_system, ngfp):
    integrator = FlowIntegrator(eh_system)
    trajs = []
    for dg in [0.01, -0.01]:
        ic = {
            "g": ngfp.location["g"] + dg,
            "lambda": ngfp.location["lambda"] + dg * 0.2,
        }
        traj = integrator.integrate(ic, t_span=(0, 1), t_eval=np.linspace(0, 1, 50))
        trajs.append(traj)
    return trajs


# ---------------------------------------------------------------------------
# KoopmanOperator (pure numpy EDMD)
# ---------------------------------------------------------------------------

def test_koopman_operator_edmd(eh_system, trajectories):
    koop = KoopmanOperator(eh_system)
    result = koop.compute_edmd(trajectories, dictionary_degree=3)
    assert "eigenvalues" in result
    assert len(result["eigenvalues"]) > 0
    assert all(np.isfinite(result["eigenvalues"]))


def test_koopman_dictionary(eh_system):
    koop = KoopmanOperator(eh_system)
    exponents = koop._build_dictionary(degree=2)
    # For 2 vars, degree 2: C(2,2)+C(3,2)+C(4,2) = 1+3+6... wait
    # degree 0: (0,0) -> 1
    # degree 1: (1,0), (0,1) -> 2
    # degree 2: (2,0), (1,1), (0,2) -> 3
    # Total = 1 + 2 + 3 = 6
    assert len(exponents) == 6


# ---------------------------------------------------------------------------
# StabilityMatrixSimulator (Pauli decomposition, pure numpy)
# ---------------------------------------------------------------------------

def test_stability_matrix_pauli_decomposition(stability):
    sim = StabilityMatrixSimulator(stability)
    terms = sim.pauli_decomposition()
    assert len(terms) > 0
    # Each term should be (coefficient, pauli_string)
    for coeff, pstr in terms:
        assert isinstance(pstr, str)
        assert all(c in "IXYZ" for c in pstr)


def test_pauli_decomposition_reconstruction(stability):
    sim = StabilityMatrixSimulator(stability)
    terms = sim.pauli_decomposition()

    n = sim._n_qubits
    dim = 1 << n

    # Pauli matrices
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {"I": I2, "X": X, "Y": Y, "Z": Z}

    # Reconstruct from Pauli decomposition
    M_reconstructed = np.zeros((dim, dim), dtype=complex)
    for coeff, pauli_str in terms:
        P = np.array([[1.0]], dtype=complex)
        for ch in pauli_str:
            P = np.kron(P, pauli_map[ch])
        M_reconstructed += coeff * P

    # The original matrix (padded to dim x dim)
    M_padded = np.zeros((dim, dim), dtype=complex)
    M_padded[:sim._n, :sim._n] = sim._M

    assert_allclose(M_reconstructed, M_padded, atol=1e-10)
