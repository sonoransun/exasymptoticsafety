"""QFT spectral decomposition of RG trajectories.

Encodes the time-series of a single coupling ``g(t_j)`` as the
amplitudes of a quantum state, applies the Quantum Fourier Transform,
and extracts the dominant spectral content (peak frequencies and
decay/growth rates).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.analysis.flow import RGTrajectory


class QuantumFourierRG:
    """QFT spectral decomposition of an RG trajectory.

    Parameters
    ----------
    trajectory:
        An :class:`RGTrajectory` whose coupling time-series will be
        analysed.
    """

    def __init__(self, trajectory: RGTrajectory) -> None:
        self._trajectory = trajectory

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_trajectory(self, coupling_name: str) -> np.ndarray:
        """Encode trajectory values as normalised state amplitudes.

        The coupling array is padded (with zeros) to the nearest power
        of 2 and then normalised so that ``sum |a_j|^2 = 1``.

        Parameters
        ----------
        coupling_name:
            Name of the coupling to encode.

        Returns
        -------
        np.ndarray
            Normalised amplitude vector of length ``2^n``.
        """
        values = self._trajectory.coupling_values[coupling_name].copy()

        # Pad to nearest power of 2
        n_raw = len(values)
        n_qubits = int(np.ceil(np.log2(max(n_raw, 2))))
        n_padded = 1 << n_qubits
        padded = np.zeros(n_padded)
        padded[:n_raw] = values

        # Normalise
        norm = np.linalg.norm(padded)
        if norm < 1e-30:
            return padded
        return padded / norm

    # ------------------------------------------------------------------
    # QFT circuit
    # ------------------------------------------------------------------

    def build_qft_circuit(self, n_qubits: int) -> Any:
        """Build a QFT circuit on *n_qubits* qubits.

        The standard QFT uses Hadamard and controlled-phase gates:

        .. math::
            \\text{QFT} |j\\rangle = \\frac{1}{\\sqrt{N}}
                \\sum_k e^{2\\pi i j k / N} |k\\rangle

        Returns
        -------
        QuantumCircuit
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        qc = QuantumCircuit(n_qubits, name="QFT")

        for target in range(n_qubits):
            qc.h(target)
            for control in range(target + 1, n_qubits):
                k = control - target + 1
                qc.cp(2 * np.pi / (1 << k), control, target)

        # Swap qubits for standard QFT ordering
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - 1 - i)

        return qc

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def spectral_analysis(self, coupling_name: str) -> dict:
        """Apply QFT and extract peak frequencies and amplitudes.

        Uses exact statevector simulation (via
        :class:`qiskit.quantum_info.Statevector`) to obtain the
        frequency-domain amplitudes, then identifies the dominant peaks.

        Parameters
        ----------
        coupling_name:
            Name of the coupling to analyse.

        Returns
        -------
        dict
            Keys: ``frequencies`` (all), ``amplitudes`` (magnitude
            squared), ``peak_frequencies`` (indices of dominant peaks).
        """
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for quantum features. "
                "Install with: pip install asymsafety[quantum]"
            ) from exc

        amplitudes = self.encode_trajectory(coupling_name)
        n_padded = len(amplitudes)
        n_qubits = int(np.log2(n_padded))

        # Initialise statevector and apply QFT
        sv = Statevector(amplitudes)
        qft_circuit = self.build_qft_circuit(n_qubits)
        sv = sv.evolve(qft_circuit)

        freq_amplitudes = np.abs(sv.data) ** 2

        # Frequency bins
        dt = 1.0
        if len(self._trajectory.t_values) > 1:
            dt = float(
                self._trajectory.t_values[1] - self._trajectory.t_values[0]
            )
        frequencies = np.arange(n_padded) / (n_padded * dt)

        # Peak detection: simple threshold at mean + 2*std
        threshold = np.mean(freq_amplitudes) + 2.0 * np.std(freq_amplitudes)
        peak_indices = np.where(freq_amplitudes > threshold)[0]

        return {
            "frequencies": frequencies,
            "amplitudes": freq_amplitudes,
            "peak_frequencies": peak_indices,
        }
