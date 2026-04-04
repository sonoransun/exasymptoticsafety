"""Continuous wavelet transform of RG flow trajectories.

Performs a continuous wavelet transform (CWT) of each coupling
trajectory g_i(t) to produce a time-scale decomposition:

    W_i(a, b) = (1 / sqrt(a)) integral g_i(t) psi*((t - b) / a) dt

The resulting scalogram |W(a, b)|^2 reveals the energy distribution
across RG scales and flow times, making it possible to identify
transient crossover regimes and multi-scale structure in the flow.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.transforms._types import WaveletResult


def _morlet(t: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """Morlet wavelet: psi(t) = pi^{-1/4} exp(i omega0 t) exp(-t^2/2)."""
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)


def _mexican_hat(t: np.ndarray) -> np.ndarray:
    """Mexican hat wavelet: psi(t) = (2/sqrt(3)) pi^{-1/4} (1 - t^2) exp(-t^2/2)."""
    return (2.0 / np.sqrt(3.0)) * np.pi ** (-0.25) * (1 - t**2) * np.exp(-t**2 / 2)


class RGFlowWavelet:
    """Continuous wavelet transform of RG flow coupling trajectories.

    Args:
        trajectory: Integrated RG flow trajectory.
    """

    def __init__(self, trajectory: RGTrajectory) -> None:
        self._trajectory = trajectory
        self._t = trajectory.t_values
        self._names = trajectory.coupling_names
        self._signals = np.array(
            [trajectory.coupling_values[name] for name in self._names]
        )

    def transform(
        self,
        scales: np.ndarray,
        wavelet_type: str = "morlet",
        omega0: float = 6.0,
    ) -> WaveletResult:
        """Compute the continuous wavelet transform of each coupling.

        For each coupling, scale *a*, and position *b*:

            W(a, b) = (1/sqrt(a)) integral g(t) psi*((t - b) / a) dt

        The integration is performed via the trapezoidal rule using
        the trajectory time grid.

        Supported wavelets:
            - ``"morlet"``: Complex Morlet wavelet with centre frequency
              *omega0*.
            - ``"mexican_hat"``: Real-valued Mexican hat (Ricker)
              wavelet.

        Args:
            scales: 1-D array of positive wavelet scales.
            wavelet_type: Name of the mother wavelet.
            omega0: Centre frequency for the Morlet wavelet (ignored
                for other types).

        Returns:
            WaveletResult with coefficients of shape
            ``(n_couplings, n_scales, n_positions)``.

        Raises:
            ValueError: If *wavelet_type* is not supported.
        """
        t = self._t
        positions = t.copy()
        n_couplings = len(self._names)
        n_scales = len(scales)
        n_positions = len(positions)

        coefficients = np.empty(
            (n_couplings, n_scales, n_positions), dtype=np.complex128,
        )

        for si, a in enumerate(scales):
            inv_sqrt_a = 1.0 / np.sqrt(a)
            for bi, b in enumerate(positions):
                eta = (t - b) / a

                if wavelet_type == "morlet":
                    psi = _morlet(eta, omega0=omega0)
                elif wavelet_type == "mexican_hat":
                    psi = _mexican_hat(eta)
                else:
                    raise ValueError(
                        f"Unsupported wavelet type: {wavelet_type!r}. "
                        "Choose 'morlet' or 'mexican_hat'."
                    )

                # Conjugate the wavelet for the inner product
                psi_conj = np.conj(psi)

                for ci in range(n_couplings):
                    integrand = self._signals[ci] * psi_conj
                    coefficients[ci, si, bi] = inv_sqrt_a * trapezoid(
                        integrand, t,
                    )

        return WaveletResult(
            scales=scales,
            positions=positions,
            coefficients=coefficients,
            coupling_names=list(self._names),
        )

    def scalogram(
        self,
        scales: np.ndarray | None = None,
        wavelet_type: str = "morlet",
        omega0: float = 6.0,
        n_scales: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the wavelet scalogram |W(a, b)|^2.

        The scalogram gives the energy density in the time-scale plane,
        useful for identifying crossover regimes where the flow
        transitions between fixed points.

        Args:
            scales: Wavelet scales.  If *None*, a logarithmically
                spaced set of *n_scales* scales is generated from the
                trajectory time extent.
            wavelet_type: Mother wavelet name.
            omega0: Centre frequency for Morlet.
            n_scales: Number of scales when *scales* is not provided.

        Returns:
            ``(scales, positions, energy)`` where *energy* has shape
            ``(n_couplings, n_scales, n_positions)``.
        """
        if scales is None:
            t_extent = float(self._t[-1] - self._t[0])
            dt = float(self._t[1] - self._t[0])
            s_min = 2 * dt
            s_max = t_extent / 2
            scales = np.geomspace(s_min, s_max, n_scales)

        result = self.transform(scales, wavelet_type=wavelet_type, omega0=omega0)

        energy = np.abs(result.coefficients) ** 2

        return scales, result.positions, energy
