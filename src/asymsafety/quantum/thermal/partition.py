"""Extract Seeley-DeWitt coefficients from the partition function.

The heat-kernel partition function admits a small-*s* (high-temperature)
expansion:

.. math::
    Z(\\beta) = (4\\pi\\beta)^{-d/2}
                \\bigl[b_0 + \\beta\\,b_2 + \\beta^2\\,b_4 + \\cdots\\bigr]

By evaluating ``Z(beta)`` at several small values of ``beta`` and
fitting the polynomial on the right-hand side, the Seeley-DeWitt
coefficients ``b_0, b_2, b_4`` can be extracted numerically and
compared with their classical (symbolic) values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymsafety.quantum.thermal.gibbs import GibbsStatePreparer


class PartitionFunctionEstimator:
    """Extract Seeley-DeWitt coefficients from Z(beta).

    Parameters
    ----------
    gibbs:
        A :class:`GibbsStatePreparer` that can evaluate the partition
        function.
    d:
        Spacetime dimension (default 4).
    """

    def __init__(self, gibbs: GibbsStatePreparer, d: int = 4) -> None:
        self._gibbs = gibbs
        self._d = d

    @property
    def gibbs(self) -> GibbsStatePreparer:
        """The wrapped Gibbs-state preparer (used by partition-function plots)."""
        return self._gibbs

    @property
    def d(self) -> int:
        """Spacetime dimension."""
        return self._d

    # ------------------------------------------------------------------
    # Coefficient extraction
    # ------------------------------------------------------------------

    def estimate_coefficients(
        self,
        beta_values: np.ndarray | None = None,
        n_coeffs: int = 3,
    ) -> dict[str, float]:
        """Fit ``Z(beta)`` to extract ``b_0, b_2, b_4, ...``.

        Procedure
        ---------
        1. Evaluate ``Z(beta)`` at several small ``beta`` values.
        2. Compute ``Z_reduced = Z * (4*pi*beta)^{d/2}``.
        3. Fit polynomial ``Z_reduced = b_0 + beta*b_2 + beta^2*b_4 + ...``
        4. Return the coefficients.

        Parameters
        ----------
        beta_values:
            Array of ``beta`` values at which to sample.  Defaults to
            ``np.linspace(0.01, 0.5, 20)``.
        n_coeffs:
            Number of Seeley-DeWitt coefficients to extract
            (default 3 for ``b_0, b_2, b_4``).

        Returns
        -------
        dict
            Mapping ``"b0" -> value, "b2" -> value, "b4" -> value, ...``
        """
        if beta_values is None:
            beta_values = np.linspace(0.01, 0.5, 20)

        d = self._d
        z_reduced = np.empty(len(beta_values))

        for i, beta in enumerate(beta_values):
            z_val = self._gibbs.partition_function(beta)
            # Z_reduced = Z * (4*pi*beta)^{d/2}
            prefactor = (4.0 * np.pi * beta) ** (d / 2.0)
            z_reduced[i] = z_val * prefactor

        # Fit polynomial of degree (n_coeffs - 1) in beta
        # Z_reduced ~ b_0 + beta*b_2 + beta^2*b_4 + ...
        # This is a polynomial in beta with coefficients b_{2k}
        powers = np.column_stack([beta_values ** k for k in range(n_coeffs)])
        coeffs, _, _, _ = np.linalg.lstsq(powers, z_reduced, rcond=None)

        labels = ["b0", "b2", "b4", "b6", "b8", "b10"]
        result: dict[str, float] = {}
        for k in range(n_coeffs):
            label = labels[k] if k < len(labels) else f"b{2 * k}"
            result[label] = float(coeffs[k])
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against_classical(
        self,
        sdw: Any,  # SeeleyDeWittCoefficients
        field_type: str = "scalar",
    ) -> dict[str, Any]:
        """Compare quantum-extracted coefficients with classical values.

        Parameters
        ----------
        sdw:
            A :class:`SeeleyDeWittCoefficients` instance for the same
            spacetime dimension and background.
        field_type:
            Field type to compare (``"scalar"``, ``"vector"``, etc.).

        Returns
        -------
        dict
            Keys: ``"quantum"``, ``"classical"``, ``"deviations"``.
        """
        quantum_coeffs = self.estimate_coefficients()

        import sympy
        # Evaluate classical SDW coefficients numerically
        # sdw.b0, sdw.b2, sdw.b4_on_sphere all return sympy Expr
        classical: dict[str, float] = {}
        classical["b0"] = float(sdw.b0(field_type))
        classical["b2"] = float(sdw.b2(field_type))
        try:
            classical["b4"] = float(sdw.b4_on_sphere(field_type))
        except Exception:
            classical["b4"] = float("nan")

        deviations: dict[str, float] = {}
        for key in quantum_coeffs:
            if key in classical:
                deviations[key] = abs(quantum_coeffs[key] - classical[key])
            else:
                deviations[key] = float("nan")

        return {
            "quantum": quantum_coeffs,
            "classical": classical,
            "deviations": deviations,
        }
