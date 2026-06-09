"""Tests for the York decomposition and S^4 mode spectra.

Degeneracy ground truth: Rubin & Ordóñez, J. Math. Phys. 25 (1984) 2888.
Excluded-mode ground truth: Lauscher & Reuter, Phys. Rev. D 65 (2002) 025013.
"""

import pytest
import sympy
from sympy import Rational, Symbol

from asymsafety.geometry.decomposition import ModeSpectrum, YorkDecomposition


@pytest.fixture
def spectrum():
    return ModeSpectrum(d=4)


@pytest.fixture
def york():
    return YorkDecomposition(d=4)


class TestS4Multiplicities:
    """Exact SO(5) representation dimensions on S^4."""

    def test_scalar_multiplicities(self, spectrum):
        """Scalar: d_l = (2l+3)(l+1)(l+2)/6 -> 1, 5, 14, 30, 55, 91, 140."""
        expected = [1, 5, 14, 30, 55, 91, 140]
        got = [spectrum.multiplicity("scalar", l) for l in range(7)]
        assert got == expected

    def test_scalar_l1_is_five(self, spectrum):
        """Anchor: 5 Cartesian coordinate functions of S^4 ⊂ R^5."""
        assert spectrum.multiplicity("scalar", 1) == 5

    def test_vector_multiplicities(self, spectrum):
        """Transverse vector: d_l = l(l+3)(2l+3)/2 -> 10, 35, 81, 154."""
        expected = [0, 10, 35, 81, 154]
        got = [spectrum.multiplicity("vector", l) for l in range(5)]
        assert got == expected

    def test_vector_l1_killing_vectors(self, spectrum):
        """Anchor: l=1 transverse vectors are the 10 = dim SO(5) Killing
        vectors of S^4."""
        assert spectrum.multiplicity("vector", 1) == 10

    def test_TT_multiplicities(self, spectrum):
        """TT tensor: d_l = 5(2l+3)(l-1)(l+4)/6 -> 35, 105, 220, 390."""
        expected = [0, 0, 35, 105, 220, 390]
        got = [spectrum.multiplicity("TT", l) for l in range(6)]
        assert got == expected

    def test_weyl_law_degree(self):
        """Degeneracies on a 4-manifold must grow as l^{d-1} = l³."""
        l = Symbol("l")
        scalar = (2 * l + 3) * (l + 1) * (l + 2) / 6
        vector = l * (l + 3) * (2 * l + 3) / 2
        TT = 5 * (2 * l + 3) * (l - 1) * (l + 4) / 6
        for expr in (scalar, vector, TT):
            assert sympy.Poly(expr, l).degree() == 3

    def test_multiplicities_are_integers(self, spectrum):
        """The closed forms divide exactly for every l."""
        for l in range(50):
            for ft in ("scalar", "vector", "TT"):
                d_l = spectrum.multiplicity(ft, l)
                assert isinstance(d_l, int)
                assert d_l >= 0

    def test_non_d4_raises(self):
        with pytest.raises(NotImplementedError):
            ModeSpectrum(d=3).multiplicity("scalar", 2)


class TestS4Eigenvalues:
    """Laplacian eigenvalues λ_l = [l(l+3) - shift]/a² on S^4."""

    def test_eigenvalues(self, spectrum):
        a_sq = Symbol("a2", positive=True)
        for ft, shift in [("scalar", 0), ("vector", 1), ("TT", 2)]:
            ev = spectrum.eigenvalue(ft, 3, a_sq)
            assert sympy.simplify(ev - (3 * 6 - shift) / a_sq) == 0

    def test_min_l(self, spectrum):
        assert spectrum.min_l("scalar") == 0
        assert spectrum.min_l("vector") == 1
        assert spectrum.min_l("TT") == 2


class TestYorkDecomposition:
    """Degrees of freedom and excluded modes."""

    def test_dof_bookkeeping(self, york):
        """TT(5) + vector(3) + scalar(2) = 10 = d(d+1)/2 components."""
        assert york.dof_TT == 5
        assert york.dof_vector == 3
        assert york.dof_scalar == 2
        assert york.total_dof == 10
        assert york.dof_TT + york.dof_vector + york.dof_scalar == york.total_dof

    def test_excluded_modes_vector_killing(self, york):
        """ξ l=1 (the 10 Killing vectors, D_μξ_ν + D_νξ_μ = 0) are
        excluded from the physical spectrum (Lauscher-Reuter PRD 65,
        025013)."""
        assert york.excluded_modes("vector") == [1]

    def test_excluded_modes_sigma(self, york):
        """σ l=0 (constant) and l=1 (conformal Killing scalars) are
        annihilated by D_μD_ν - (1/4)g_μν D²."""
        assert york.excluded_modes("scalar_sigma") == [0, 1]

    def test_excluded_modes_TT_and_h(self, york):
        """No zero modes for TT (the tower starts at l=2) or for h."""
        assert york.excluded_modes("TT") == []
        assert york.excluded_modes("scalar_h") == []
