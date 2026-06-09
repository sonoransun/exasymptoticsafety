"""Tests for spectral-sum trace evaluation on S^4 and S¹ × S³.

Degeneracy ground truth: Rubin & Ordóñez, J. Math. Phys. 25 (1984) 2888
(d=4 and d=3 spheres).
"""

import math

import numpy as np
import pytest

from asymsafety.compute.batch.spectral import (
    _S3_multiplicity_vectorized,
    trace_on_S1xS3_vectorized,
    trace_on_sphere_vectorized,
)
from asymsafety.frg.spectral import SpectralSumEvaluator


# Ground-truth degeneracies, written independently of the implementation.
def _deg_S4(field_type, l):
    if field_type == "scalar":
        return (2 * l + 3) * (l + 1) * (l + 2) // 6
    if field_type == "vector":
        return l * (l + 3) * (2 * l + 3) // 2 if l >= 1 else 0
    if field_type == "TT":
        return 5 * (2 * l + 3) * (l - 1) * (l + 4) // 6 if l >= 2 else 0
    raise ValueError(field_type)


def _deg_S3(field_type, l):
    if field_type == "scalar":
        return (l + 1) ** 2
    if field_type == "vector":
        return 2 * l * (l + 2) if l >= 1 else 0
    if field_type == "TT":
        return 2 * (l - 1) * (l + 3) if l >= 2 else 0
    raise ValueError(field_type)


class TestTraceOnSphere:
    """Tr[W(-D²)] on S^4 against an independent reference sum."""

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_trace_matches_reference(self, field_type):
        l_max = 64
        ss = SpectralSumEvaluator(d=4, l_max=l_max)
        W = lambda z: 1.0 / (1.0 + z)
        shift = {"scalar": 0, "vector": 1, "TT": 2}[field_type]
        l_min = {"scalar": 0, "vector": 1, "TT": 2}[field_type]
        # Unit S^4: R_bar = 12, a² = 1
        ref = sum(
            _deg_S4(field_type, l) * W(l * (l + 3) - shift)
            for l in range(l_min, l_max + 1)
        )
        got = ss.trace_on_sphere(W, field_type, 12.0)
        assert got == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_vectorized_matches_loop(self, field_type):
        l_max = 64
        ss = SpectralSumEvaluator(d=4, l_max=l_max)
        W = lambda z: np.exp(-0.05 * z)
        loop = ss.trace_on_sphere(W, field_type, 12.0)
        vec = trace_on_sphere_vectorized(
            W, field_type, 12.0, d=4, l_max=l_max,
        )
        assert vec == pytest.approx(loop, rel=1e-12)

    def test_heat_trace_b0_scalar(self):
        """6 t² Tr e^{tD²} → b0 = 1 as t → 0 on unit S^4 (Weyl law).

        This is the check the old degeneracy (l+1)²(l+2)²/4 failed:
        it grew as l⁴ and the rescaled heat trace diverged ~ t^{-1/2}.
        """
        t = 0.01
        ss = SpectralSumEvaluator(d=4, l_max=200)
        tr = ss.trace_on_sphere(lambda z: math.exp(-t * z), "scalar", 12.0)
        assert 6 * t**2 * tr == pytest.approx(1.0, abs=0.05)


class TestTraceOnS1xS3:
    """End-to-end foliated trace on S¹ × S³ (regression for the
    always-true hasattr guard that made every call raise
    NotImplementedError)."""

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_trace_matches_reference(self, field_type):
        l_max, n_mats = 32, 5
        beta_period = 2 * math.pi
        ss = SpectralSumEvaluator(d=4, l_max=l_max)
        W = lambda z, om: 1.0 / (1.0 + z + om)
        l_min = {"scalar": 0, "vector": 1, "TT": 2}[field_type]
        shift = {"scalar": 0, "vector": 1, "TT": 2}[field_type]
        # Unit S³: R^(3) = 6, a² = 1
        ref = sum(
            _deg_S3(field_type, l)
            * W(l * (l + 2) - shift, (2 * math.pi * n / beta_period) ** 2)
            for n in range(-n_mats, n_mats + 1)
            for l in range(l_min, l_max + 1)
        )
        got = ss.trace_on_S1xS3(
            W, field_type, 6.0, beta_period, n_matsubara=n_mats,
        )
        assert got == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_vectorized_matches_loop(self, field_type):
        l_max, n_mats = 32, 5
        ss = SpectralSumEvaluator(d=4, l_max=l_max)
        W = lambda z, om: 1.0 / (1.0 + z + om) ** 2
        loop = ss.trace_on_S1xS3(W, field_type, 6.0, 2.0, n_matsubara=n_mats)
        vec = trace_on_S1xS3_vectorized(
            W, field_type, 6.0, 2.0, n_matsubara=n_mats, l_max=l_max,
        )
        assert vec == pytest.approx(loop, rel=1e-12)


class TestS3Multiplicities:
    """S³ degeneracies: scalar (l+1)², vector 2l(l+2), TT 2(l-1)(l+3)."""

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_loop_implementation(self, field_type):
        for l in range(10):
            got = SpectralSumEvaluator._S3_multiplicity(field_type, l)
            assert got == _deg_S3(field_type, l)

    @pytest.mark.parametrize("field_type", ["scalar", "vector", "TT"])
    def test_vectorized_implementation(self, field_type):
        ls = np.arange(0, 10)
        got = _S3_multiplicity_vectorized(field_type, ls)
        expected = [_deg_S3(field_type, int(l)) for l in ls]
        assert np.allclose(got, expected)

    def test_TT_l2_anchor(self):
        """TT degeneracy at l=2 on S³ is 10 (the old formula gave the
        non-integer 25/3)."""
        assert SpectralSumEvaluator._S3_multiplicity("TT", 2) == 10
