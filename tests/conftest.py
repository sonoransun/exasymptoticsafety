"""Shared pytest fixtures for asymptotic safety tests."""

import numpy as np
import pytest
import sympy
from sympy import Symbol, Rational

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.frg.regulator import LitimRegulator, ExponentialRegulator
from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.geometry.curvature import MaxSymBackground


def make_as_trajectory(
    g_star: float = 0.7,
    lam_star: float = 0.2,
    xi_const: float | None = None,
    t_range: tuple[float, float] = (-15.0, 15.0),
    n: int = 601,
) -> RGTrajectory:
    """Synthetic asymptotically-safe RG trajectory for scattering tests.

    Uses the closed-form crossover ``g(t) = g* e^{2t}/(1 + e^{2t})``, which
    interpolates the IR Gaussian regime ``g ∝ k²`` (so the dimensionful
    ``G = g/k² → g*/k0²`` is constant — a finite Newton constant) and the
    UV non-Gaussian fixed point ``g → g*`` (so ``G ∝ 1/k²``).  ``lambda`` is
    held constant; ``xi`` is added only when ``xi_const`` is given.
    """
    t = np.linspace(t_range[0], t_range[1], n)
    e2t = np.exp(2.0 * t)
    g_vals = g_star * e2t / (1.0 + e2t)
    coupling_values = {"g": g_vals, "lambda": np.full_like(t, lam_star)}
    names = ["g", "lambda"]
    if xi_const is not None:
        coupling_values["xi"] = np.full_like(t, xi_const)
        names.append("xi")
    return RGTrajectory(
        t_values=t, coupling_values=coupling_values, coupling_names=names,
    )


@pytest.fixture
def as_trajectory():
    """Synthetic asymptotically-safe trajectory (g*, lambda* fixed)."""
    return make_as_trajectory()


@pytest.fixture
def make_trajectory():
    """Factory fixture returning :func:`make_as_trajectory`."""
    return make_as_trajectory


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
