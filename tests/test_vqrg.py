"""Tests for variational quantum RG (no qiskit required - test pure math parts)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asymsafety.quantum.vqrg.mapping import CouplingAngleMap
from asymsafety.quantum.utils.qubit_encoding import (
    float_to_binary,
    binary_to_float,
    gray_code,
)
from asymsafety.quantum.utils.entanglement import (
    von_neumann_entropy,
    reduced_density_matrix,
)


# ---------------------------------------------------------------------------
# CouplingAngleMap
# ---------------------------------------------------------------------------

def test_coupling_angle_map_roundtrip():
    cam = CouplingAngleMap({"g": (0.0, 2.0), "lambda": (-0.5, 0.5)})
    couplings = {"g": 0.7, "lambda": 0.14}
    angles = cam.couplings_to_angles(couplings)
    recovered = cam.angles_to_couplings(angles)
    assert_allclose(recovered["g"], couplings["g"], rtol=1e-10)
    assert_allclose(recovered["lambda"], couplings["lambda"], rtol=1e-10)


def test_coupling_angle_map_bounds():
    cam = CouplingAngleMap({"g": (0.0, 2.0), "lambda": (-0.5, 0.5)})
    # At g_min, angle should be 0; at g_max, angle should be pi
    angles_min = cam.couplings_to_angles({"g": 0.0, "lambda": -0.5})
    angles_max = cam.couplings_to_angles({"g": 2.0, "lambda": 0.5})
    assert_allclose(angles_min[0], 0.0, atol=1e-10)
    assert_allclose(angles_min[1], 0.0, atol=1e-10)
    assert_allclose(angles_max[0], np.pi, rtol=1e-10)
    assert_allclose(angles_max[1], np.pi, rtol=1e-10)


# ---------------------------------------------------------------------------
# Qubit encoding
# ---------------------------------------------------------------------------

def test_float_to_binary_roundtrip():
    n_bits = 8
    value_range = (0.0, 1.0)
    original = 0.75
    bits = float_to_binary(original, n_bits, value_range)
    recovered = binary_to_float(bits, value_range)
    # Precision limited by 2^n_bits bins
    assert abs(recovered - original) < 1.0 / (2**n_bits)


def test_gray_code_length():
    for n in range(1, 5):
        gc = gray_code(n)
        assert len(gc) == 2**n


# ---------------------------------------------------------------------------
# Entanglement measures
# ---------------------------------------------------------------------------

def test_von_neumann_entropy_pure_state():
    # |00> is a product state -> entropy of subsystem A (qubit 0) is 0
    psi = np.array([1, 0, 0, 0], dtype=complex)
    S = von_neumann_entropy(psi, n_qubits=2, subsystem_qubits=[0])
    assert_allclose(S, 0.0, atol=1e-10)


def test_von_neumann_entropy_bell_state():
    # Bell state: (|00> + |11>) / sqrt(2)
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    S = von_neumann_entropy(psi, n_qubits=2, subsystem_qubits=[0])
    # Maximally entangled -> S = 1 bit
    assert_allclose(S, 1.0, atol=1e-6)


def test_reduced_density_matrix_trace():
    # Any pure state's reduced DM should have Tr(rho) = 1
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = reduced_density_matrix(psi, n_qubits=2, keep_qubits=[0])
    assert_allclose(np.trace(rho).real, 1.0, rtol=1e-10)
    assert rho.shape == (2, 2)
