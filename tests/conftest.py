"""Shared pytest fixtures for asymptotic safety tests."""

import pytest
import sympy
from sympy import Symbol, Rational

from asymsafety.frg.regulator import LitimRegulator, ExponentialRegulator
from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.geometry.curvature import MaxSymBackground


@pytest.fixture
def litim():
    """Litim regulator instance."""
    return LitimRegulator()


@pytest.fixture
def exponential():
    """Exponential regulator instance."""
    return ExponentialRegulator()


@pytest.fixture
def threshold_litim():
    """Threshold functions with Litim regulator."""
    return ThresholdFunctions(LitimRegulator())


@pytest.fixture
def R_bar():
    """Background Ricci scalar symbol."""
    return Symbol("R_bar", positive=True)


@pytest.fixture
def z():
    """Laplacian eigenvalue symbol."""
    return Symbol("z", positive=True)


@pytest.fixture
def k():
    """RG scale symbol."""
    return Symbol("k", positive=True)


@pytest.fixture
def g():
    """Dimensionless Newton coupling symbol."""
    return Symbol("g", positive=True)


@pytest.fixture
def lam():
    """Dimensionless cosmological constant symbol."""
    return Symbol("lambda", real=True)


@pytest.fixture
def bg_S4(R_bar):
    """Maximally symmetric S^4 background."""
    return MaxSymBackground(d=4, R_bar=R_bar)
