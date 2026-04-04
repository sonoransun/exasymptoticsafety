"""Laplacian eigenvalues on S^d encoded as a qubit Hamiltonian.

Takes the mode spectrum from the geometry module, truncates to a
manageable number of angular-momentum modes, and encodes the resulting
diagonal Hamiltonian into a qubit register.  Pauli decomposition is
provided for use with variational and Trotterised algorithms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import sympy

if TYPE_CHECKING:
    from asymsafety.geometry.decomposition import ModeSpectrum


class LaplacianHamiltonian:
    """Encode Laplacian eigenvalues on S^d as a qubit Hamiltonian.

    The Hamiltonian is diagonal in the computational basis:

    .. math::
        H = \\sum_l \\lambda_l |l\\rangle\\langle l|

    Parameters
    ----------
    spectrum:
        Mode spectrum from the geometry module.
    field_type:
        Field type passed to the spectrum (``"scalar"``, ``"vector"``,
        ``"TT"``).
    l_max:
        Maximum angular momentum quantum number to include.
    sphere_radius_sq:
        Squared radius ``a^2`` of the background sphere.
    """

    def __init__(
        self,
        spectrum: ModeSpectrum,
        field_type: str = "scalar",
        l_max: int = 3,
        sphere_radius_sq: float = 1.0,
    ) -> None:
        self._spectrum = spectrum
        self._field_type = field_type
        self._l_max = l_max
        self._a_sq = sphere_radius_sq

        # Pre-compute eigenvalues and multiplicities
        l_min = spectrum.min_l(field_type)
        a_sq_sym = sympy.Rational(sphere_radius_sq).limit_denominator(10**9)

        self._l_values: list[int] = []
        self._eigenvalue_list: list[float] = []
        self._multiplicity_list: list[int] = []

        for l in range(l_min, l_max + 1):
            ev = spectrum.eigenvalue(field_type, l, a_sq_sym)
            mult = spectrum.multiplicity(field_type, l)
            self._l_values.append(l)
            self._eigenvalue_list.append(float(ev))
            self._multiplicity_list.append(mult)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        """Number of modes after truncation."""
        return len(self._l_values)

    @property
    def n_qubits(self) -> int:
        """Number of qubits: ``ceil(log2(n_modes))``."""
        if self.n_modes <= 1:
            return 1
        return math.ceil(math.log2(self.n_modes))

    # ------------------------------------------------------------------
    # Spectrum access
    # ------------------------------------------------------------------

    def eigenvalues(self) -> np.ndarray:
        """Sorted array of Laplacian eigenvalues."""
        return np.array(self._eigenvalue_list)

    def multiplicities(self) -> np.ndarray:
        """Multiplicities corresponding to each eigenvalue."""
        return np.array(self._multiplicity_list)

    # ------------------------------------------------------------------
    # Matrix representation
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Full Hamiltonian matrix ``(2^n x 2^n)``.

        Diagonal with eigenvalues; unused computational-basis states
        (those beyond ``n_modes``) are padded with zeros.
        """
        dim = 1 << self.n_qubits
        H = np.zeros((dim, dim))
        evals = self.eigenvalues()
        for i, ev in enumerate(evals):
            H[i, i] = ev
        return H

    # ------------------------------------------------------------------
    # Pauli decomposition
    # ------------------------------------------------------------------

    def to_pauli_decomposition(self) -> list[tuple[float, str]]:
        """Decompose H into the Pauli basis.

        .. math::
            H = \\sum_P c_P \\, P, \\qquad P \\in \\{I,X,Y,Z\\}^{\\otimes n}

        Since H is diagonal and real, only products of ``I`` and ``Z``
        appear.

        Returns
        -------
        list of (coefficient, pauli_string) pairs
            Only terms with ``|c_P| > 1e-15`` are included.
        """
        n = self.n_qubits
        dim = 1 << n
        diag = np.zeros(dim)
        evals = self.eigenvalues()
        diag[: len(evals)] = evals

        terms: list[tuple[float, str]] = []

        # Iterate over all 2^n IZ strings
        for mask in range(dim):
            # Build the Pauli string for this mask
            pauli_chars: list[str] = []
            for q in range(n):
                pauli_chars.append("Z" if (mask >> q) & 1 else "I")
            pauli_str = "".join(reversed(pauli_chars))

            # Compute coefficient: c_P = (1/2^n) * sum_j diag_j * P_jj
            # For an IZ string with mask m, the diagonal entry at |j> is
            # (-1)^{popcount(j & m)}.
            coeff = 0.0
            for j in range(dim):
                sign = (-1) ** bin(j & mask).count("1")
                coeff += diag[j] * sign
            coeff /= dim

            if abs(coeff) > 1e-15:
                terms.append((coeff, pauli_str))

        return terms
