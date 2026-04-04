"""Result types for RG flow transforms.

Each transform produces a structured result that captures the transformed
data along with domain information and any derived quantities (poles,
scaling dimensions, dominant frequencies, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TransformResult:
    """Base result for frequency-domain transforms of RG trajectories.

    Attributes:
        domain_values: Values in the transform domain (e.g., frequencies,
            Laplace variable s, Mellin variable s).
        transform_values: Complex-valued transform output, shape
            (n_couplings, n_domain) or (n_domain,) for scalar signals.
        coupling_names: Ordered list of coupling names corresponding
            to the first axis of ``transform_values``.
    """

    domain_values: np.ndarray
    transform_values: np.ndarray
    coupling_names: list[str]


@dataclass
class LaplaceResult(TransformResult):
    """Result of a Laplace transform of RG flow data.

    The Laplace transform F(s) = integral_0^infty f(t) e^{-st} dt
    converts the RG time-domain trajectory into a meromorphic function
    whose poles encode the critical exponents.

    Attributes:
        poles: Detected poles in the s-plane, keyed by coupling name.
            Each value is an array of complex pole locations.
    """

    poles: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def s_values(self) -> np.ndarray:
        """Laplace variable values (alias for ``domain_values``)."""
        return self.domain_values


@dataclass
class MellinResult(TransformResult):
    """Result of a Mellin transform of RG flow data.

    The Mellin transform M(s) = integral_0^infty t^{s-1} f(t) dt
    is natural for extracting scaling dimensions from power-law
    behaviour near fixed points.

    Attributes:
        scaling_dimensions: Extracted scaling dimensions Delta_i from
            the pole structure of M(s).
    """

    scaling_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    @property
    def s_values(self) -> np.ndarray:
        """Mellin variable values (alias for ``domain_values``)."""
        return self.domain_values


@dataclass
class WaveletResult:
    """Result of a continuous wavelet transform of RG flow data.

    The wavelet transform provides scale-position decomposition of
    the RG trajectory, revealing transient behaviour and crossover
    phenomena that are invisible to Fourier/Laplace methods.

    Attributes:
        scales: Wavelet scale parameter values.
        positions: Translation parameter values (RG time positions).
        coefficients: Wavelet coefficients, shape
            (n_couplings, n_scales, n_positions).
        coupling_names: Ordered list of coupling names.
    """

    scales: np.ndarray
    positions: np.ndarray
    coefficients: np.ndarray
    coupling_names: list[str]


@dataclass
class FourierResult(TransformResult):
    """Result of a Fourier transform of RG flow data.

    Useful for detecting periodic or quasi-periodic structures in the
    RG flow (e.g., limit cycles, spiral flow near complex critical
    exponents).

    Attributes:
        power_spectrum: |F(omega)|^2 for each coupling, shape
            (n_couplings, n_frequencies).
        dominant_frequencies: Frequencies with the largest spectral
            weight, shape (n_dominant,).
    """

    power_spectrum: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    dominant_frequencies: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency values (alias for ``domain_values``)."""
        return self.domain_values


@dataclass
class KoopmanResult:
    """Result of a Koopman operator analysis of RG flow.

    The Koopman operator K acts on observables (functions of coupling
    space) and advances them by one RG step.  Its spectral decomposition
    provides a global linear representation of the nonlinear flow.

    Attributes:
        eigenvalues: Koopman eigenvalues mu_i.
        eigenfunctions: Koopman eigenfunctions in the dictionary basis,
            shape (dictionary_size, n_modes).
        modes: Koopman modes in observable space, shape
            (n_modes, n_observables).
        dictionary_size: Number of dictionary elements used in the
            Extended DMD approximation.
    """

    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    modes: np.ndarray
    dictionary_size: int


@dataclass
class ComparisonResult:
    """Quantitative comparison between two analysis methods.

    Attributes:
        method_a: Label for the first method.
        method_b: Label for the second method.
        values_a: Representative values from method A.
        values_b: Representative values from method B.
        max_deviation: Maximum absolute deviation between matched values.
        agrees: True if the methods agree within expected tolerance.
    """

    method_a: str
    method_b: str
    values_a: np.ndarray
    values_b: np.ndarray
    max_deviation: float
    agrees: bool
