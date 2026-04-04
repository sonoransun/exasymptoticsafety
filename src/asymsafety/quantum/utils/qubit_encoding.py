"""Qubit encoding utilities for mapping floating-point couplings to qubits.

All functions are pure NumPy -- no qiskit dependency.
"""

from __future__ import annotations

import numpy as np


def float_to_binary(
    value: float,
    n_bits: int,
    value_range: tuple[float, float],
) -> list[int]:
    """Encode a float as binary qubits.

    The continuous interval ``[lo, hi]`` is discretised into ``2**n_bits``
    bins.  *value* is mapped to the nearest bin index, which is then
    represented in big-endian binary.

    Args:
        value: The floating-point value to encode.
        n_bits: Number of qubits (bits) to use.
        value_range: ``(lo, hi)`` bounds of the encoding interval.

    Returns:
        List of ``n_bits`` integers, each 0 or 1 (big-endian).
    """
    lo, hi = value_range
    n_bins = 2**n_bits
    clamped = np.clip(value, lo, hi)
    index = int(round((clamped - lo) / (hi - lo) * (n_bins - 1)))
    index = np.clip(index, 0, n_bins - 1)
    bits = [(index >> (n_bits - 1 - k)) & 1 for k in range(n_bits)]
    return bits


def binary_to_float(
    bits: list[int],
    value_range: tuple[float, float],
) -> float:
    """Decode binary qubits back to a floating-point value.

    Inverse of :func:`float_to_binary`.

    Args:
        bits: List of 0/1 integers (big-endian).
        value_range: ``(lo, hi)`` bounds used during encoding.

    Returns:
        Decoded float in ``[lo, hi]``.
    """
    lo, hi = value_range
    n_bits = len(bits)
    n_bins = 2**n_bits
    index = sum(b << (n_bits - 1 - k) for k, b in enumerate(bits))
    return lo + (hi - lo) * index / (n_bins - 1)


def gray_code(n: int) -> list[int]:
    """Generate the ``n``-bit Gray code sequence.

    Returns a list of ``2**n`` integers whose binary representations
    form a Gray code (successive entries differ in exactly one bit).

    Args:
        n: Number of bits.

    Returns:
        List of ``2**n`` integer Gray code values.
    """
    if n <= 0:
        return [0]
    return [i ^ (i >> 1) for i in range(2**n)]


def _to_gray(index: int) -> int:
    """Convert a binary integer to its Gray code representation."""
    return index ^ (index >> 1)


def _from_gray(gray_val: int) -> int:
    """Convert a Gray code integer back to standard binary."""
    mask = gray_val >> 1
    result = gray_val
    while mask:
        result ^= mask
        mask >>= 1
    return result


def float_to_gray(
    value: float,
    n_bits: int,
    value_range: tuple[float, float],
) -> list[int]:
    """Encode a float using Gray code.

    Gray code encoding is preferable for continuous optimisation because
    adjacent values differ in only a single bit, reducing Hamming
    distance jumps during variational updates.

    Args:
        value: The floating-point value to encode.
        n_bits: Number of qubits (bits) to use.
        value_range: ``(lo, hi)`` bounds of the encoding interval.

    Returns:
        List of ``n_bits`` Gray-coded integers, each 0 or 1 (big-endian).
    """
    lo, hi = value_range
    n_bins = 2**n_bits
    clamped = np.clip(value, lo, hi)
    index = int(round((clamped - lo) / (hi - lo) * (n_bins - 1)))
    index = np.clip(index, 0, n_bins - 1)
    gray_val = _to_gray(index)
    bits = [(gray_val >> (n_bits - 1 - k)) & 1 for k in range(n_bits)]
    return bits


def gray_to_float(
    bits: list[int],
    value_range: tuple[float, float],
) -> float:
    """Decode Gray code to a floating-point value.

    Inverse of :func:`float_to_gray`.

    Args:
        bits: List of 0/1 Gray-coded integers (big-endian).
        value_range: ``(lo, hi)`` bounds used during encoding.

    Returns:
        Decoded float in ``[lo, hi]``.
    """
    lo, hi = value_range
    n_bits = len(bits)
    n_bins = 2**n_bits
    gray_val = sum(b << (n_bits - 1 - k) for k, b in enumerate(bits))
    index = _from_gray(gray_val)
    return lo + (hi - lo) * index / (n_bins - 1)
