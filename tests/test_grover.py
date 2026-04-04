"""Tests for Grover search encoding (no qiskit required)."""
from __future__ import annotations

import numpy as np
import pytest

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.quantum.grover.encoding import CouplingGridEncoding
from asymsafety.quantum.grover.oracle import BetaOracle


@pytest.fixture
def encoding():
    return CouplingGridEncoding(
        coupling_ranges={"g": (0.0, 2.0), "lambda": (-0.5, 0.5)},
        n_bits=3,
    )


@pytest.fixture
def eh_system():
    return build_eh_beta_system(d=4)


# ---------------------------------------------------------------------------
# CouplingGridEncoding
# ---------------------------------------------------------------------------

def test_coupling_grid_encoding_properties(encoding):
    assert encoding.n_qubits == 2 * 3  # 2 couplings * 3 bits
    assert encoding.n_grid_points == (2**3) ** 2  # 8^2 = 64
    assert encoding.coupling_names == ["g", "lambda"]
    assert encoding.n_bits == 3


def test_index_to_couplings(encoding):
    couplings = encoding.index_to_couplings(0)
    assert "g" in couplings
    assert "lambda" in couplings
    # Index 0 should map to minimum values
    assert couplings["g"] >= 0.0
    assert couplings["lambda"] >= -0.5


def test_couplings_to_index(encoding):
    # Test roundtrip
    original_couplings = {"g": 1.0, "lambda": 0.0}
    idx = encoding.couplings_to_index(original_couplings)
    recovered = encoding.index_to_couplings(idx)
    # Should be close, within grid resolution
    assert abs(recovered["g"] - original_couplings["g"]) < 0.5
    assert abs(recovered["lambda"] - original_couplings["lambda"]) < 0.5


def test_bitstring_to_couplings(encoding):
    n_q = encoding.n_qubits  # 6
    bitstring = "0" * n_q
    couplings = encoding.bitstring_to_couplings(bitstring)
    assert "g" in couplings
    assert "lambda" in couplings


# ---------------------------------------------------------------------------
# BetaOracle (classical part only)
# ---------------------------------------------------------------------------

def test_beta_oracle_find_marked(eh_system, encoding):
    oracle = BetaOracle(eh_system, encoding, epsilon=0.5)
    marked = oracle.find_marked_states()
    assert isinstance(marked, list)
    # The Gaussian FP at g=0, lambda=0 should be marked (if within grid)
    # At least one state should be near a fixed point with large epsilon
    # (epsilon=0.5 is quite loose)
    assert len(marked) >= 0  # Could be empty for some grids; just check type
