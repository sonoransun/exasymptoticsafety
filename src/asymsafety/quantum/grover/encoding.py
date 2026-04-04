"""Binary encoding of coupling space grid into qubit states.

Maps a discretised grid of coupling constant values onto the
computational-basis states of a qubit register.  Each coupling is
encoded in *n_bits* qubits; the total register width is
``n_couplings * n_bits``.

Encoding rule
    |b_1 ... b_n>  -->  g_i = g_min + (g_max - g_min) * int(bits) / (2^n - 1)

where ``int(bits)`` is the unsigned integer represented by the binary
string for coupling *i*.
"""

from __future__ import annotations

import math


class CouplingGridEncoding:
    """Encode coupling space grid into qubit states.

    Parameters
    ----------
    coupling_ranges:
        Mapping from coupling name to ``(g_min, g_max)`` bounds.
    n_bits:
        Number of qubits per coupling dimension.
    """

    def __init__(
        self,
        coupling_ranges: dict[str, tuple[float, float]],
        n_bits: int = 6,
    ) -> None:
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self._coupling_ranges = dict(coupling_ranges)
        self._coupling_names: list[str] = list(coupling_ranges.keys())
        self._n_bits = n_bits
        self._n_couplings = len(coupling_ranges)
        self._max_int = (1 << n_bits) - 1  # 2^n_bits - 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        """Total number of qubits in the register."""
        return self._n_couplings * self._n_bits

    @property
    def n_grid_points(self) -> int:
        """Total grid points = (2^n_bits)^n_couplings."""
        return (1 << self._n_bits) ** self._n_couplings

    @property
    def coupling_names(self) -> list[str]:
        return list(self._coupling_names)

    @property
    def n_bits(self) -> int:
        return self._n_bits

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def index_to_couplings(self, index: int) -> dict[str, float]:
        """Convert a flat grid index to coupling values.

        The index is decomposed in mixed-radix form: the first coupling
        varies fastest.
        """
        if index < 0 or index >= self.n_grid_points:
            raise ValueError(
                f"index {index} out of range [0, {self.n_grid_points})"
            )
        n_per = 1 << self._n_bits
        result: dict[str, float] = {}
        remaining = index
        for name in self._coupling_names:
            sub_index = remaining % n_per
            remaining //= n_per
            g_min, g_max = self._coupling_ranges[name]
            if self._max_int == 0:
                result[name] = g_min
            else:
                result[name] = g_min + (g_max - g_min) * sub_index / self._max_int
        return result

    def couplings_to_index(self, couplings: dict[str, float]) -> int:
        """Convert coupling values to nearest grid index."""
        n_per = 1 << self._n_bits
        index = 0
        multiplier = 1
        for name in self._coupling_names:
            g_min, g_max = self._coupling_ranges[name]
            value = couplings[name]
            # Clamp to range
            value = max(g_min, min(g_max, value))
            if self._max_int == 0:
                sub_index = 0
            else:
                sub_index = round((value - g_min) / (g_max - g_min) * self._max_int)
                sub_index = max(0, min(self._max_int, sub_index))
            index += sub_index * multiplier
            multiplier *= n_per
        return index

    def bitstring_to_couplings(self, bitstring: str) -> dict[str, float]:
        """Convert measurement bitstring to coupling values.

        The bitstring is read as big-endian (leftmost bit is most
        significant).  The first ``n_bits`` bits correspond to the
        *last* coupling (qiskit convention: qubit 0 is rightmost).
        We reverse so the rightmost ``n_bits`` bits correspond to the
        first coupling.
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(
                f"Expected bitstring of length {self.n_qubits}, "
                f"got {len(bitstring)}"
            )
        # Reverse to get qubit-0 first
        reversed_bits = bitstring[::-1]
        result: dict[str, float] = {}
        for i, name in enumerate(self._coupling_names):
            start = i * self._n_bits
            end = start + self._n_bits
            segment = reversed_bits[start:end]
            # Segment is LSB-first; convert to integer
            sub_index = int(segment[::-1], 2)
            g_min, g_max = self._coupling_ranges[name]
            if self._max_int == 0:
                result[name] = g_min
            else:
                result[name] = g_min + (g_max - g_min) * sub_index / self._max_int
        return result
