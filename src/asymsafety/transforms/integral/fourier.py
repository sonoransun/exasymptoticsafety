"""Fourier analysis of RG flow trajectories.

Computes the discrete Fourier transform of coupling trajectories g_i(t)
to identify oscillatory behaviour associated with complex critical
exponents.  Peaks in the power spectral density at angular frequency
omega = Im(theta) reveal the imaginary parts of the stability matrix
eigenvalues at a non-trivial fixed point.
"""

from __future__ import annotations

import numpy as np

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.transforms._types import FourierResult


class RGFlowFourier:
    """Fourier transform of RG flow coupling trajectories.

    Given a trajectory g_i(t), computes

        G_hat_i(omega) = sum_n g_i(t_n) exp(-i omega t_n)

    via the FFT and derives the power spectral density |G_hat_i|^2.

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

    def transform(self) -> FourierResult:
        """Compute the FFT of each coupling trajectory.

        Returns:
            FourierResult containing frequencies, complex FFT
            coefficients, the power spectrum, and dominant frequencies.
        """
        n_freq = len(self._t)
        dt = float(self._t[1] - self._t[0])

        frequencies = np.fft.fftfreq(n_freq, d=dt)

        # Shape: (n_couplings, n_freq)
        fft_coeffs = np.array([np.fft.fft(sig) for sig in self._signals])

        power = np.abs(fft_coeffs) ** 2

        dominant = self.dominant_frequencies()

        return FourierResult(
            domain_values=frequencies,
            transform_values=fft_coeffs,
            coupling_names=list(self._names),
            power_spectrum=power,
            dominant_frequencies=dominant,
        )

    def dominant_frequencies(self, threshold: float = 0.1) -> np.ndarray:
        """Detect dominant frequencies in the power spectral density.

        A frequency is considered dominant if the corresponding PSD peak
        exceeds *threshold* times the maximum PSD value across all
        couplings.  Only positive frequencies are returned.

        Peaks at omega = Im(theta) for complex critical exponents.

        Args:
            threshold: Fraction of the global PSD maximum used as the
                detection floor.

        Returns:
            Sorted 1-D array of dominant angular frequencies.
        """
        frequencies, psd = self.power_spectral_density()

        # Restrict to positive frequencies
        pos_mask = frequencies > 0
        pos_freq = frequencies[pos_mask]
        pos_psd = psd[:, pos_mask]

        global_max = pos_psd.max()
        if global_max == 0.0:
            return np.array([])

        cutoff = threshold * global_max

        # Union of peaks across all couplings
        dominant_set: set[float] = set()
        for row in pos_psd:
            for i in range(1, len(row) - 1):
                if row[i] > cutoff and row[i] > row[i - 1] and row[i] > row[i + 1]:
                    dominant_set.add(float(pos_freq[i]))

        return np.sort(np.array(list(dominant_set)))

    def power_spectral_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the power spectral density of each coupling.

        Returns:
            (frequencies, psd) where *psd* has shape
            ``(n_couplings, n_freq)``.
        """
        n_freq = len(self._t)
        dt = float(self._t[1] - self._t[0])

        frequencies = np.fft.fftfreq(n_freq, d=dt)
        fft_coeffs = np.array([np.fft.fft(sig) for sig in self._signals])
        psd = np.abs(fft_coeffs) ** 2

        return frequencies, psd
