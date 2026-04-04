"""Stability analysis and RG comparison for hydraulic networks.

Provides the hydraulic-domain stability analysis at steady states and
tools to compare the linearised impedance spectrum against RG critical
exponents, verifying that the analogue mapping is faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from asymsafety.hydraulic.equations import network_impedance_matrix

if TYPE_CHECKING:
    from asymsafety.analysis.stability import StabilityAnalysis
    from asymsafety.hydraulic.network import HydraulicNetwork
    from asymsafety.hydraulic.simulator import HydraulicSteadyState


@dataclass
class ComparisonResult:
    """Result of comparing hydraulic and RG analyses.

    Attributes:
        hydraulic_eigenvalues: Eigenvalues of the linearised impedance
            matrix at the hydraulic steady state.
        rg_critical_exponents: Critical exponents from the RG stability
            analysis (theta_i = -eigenvalues of the stability matrix).
        max_deviation: Largest absolute difference between sorted
            eigenvalue magnitudes.
        agrees: ``True`` if *max_deviation* is below the tolerance.
    """

    hydraulic_eigenvalues: np.ndarray
    rg_critical_exponents: np.ndarray
    max_deviation: float
    agrees: bool


class HydraulicStabilityAnalysis:
    """Stability analysis in the hydraulic domain.

    Computes the linearised impedance matrix at a steady state,
    extracts eigenvalues, and optionally compares them against
    RG critical exponents.

    Args:
        network: The :class:`HydraulicNetwork` to analyse.
    """

    def __init__(self, network: HydraulicNetwork) -> None:
        self.network = network

    def analyze_steady_state(
        self,
        steady_state: HydraulicSteadyState,
    ) -> dict:
        """Compute the impedance matrix and its spectrum at a steady state.

        Args:
            steady_state: A converged :class:`HydraulicSteadyState`.

        Returns:
            Dictionary with keys:

            - ``"impedance_matrix"``: The Jacobian ``J_ij``.
            - ``"eigenvalues"``: Eigenvalues of ``J``.
            - ``"eigenvectors"``: Right eigenvectors of ``J``.
            - ``"stability"``: Classification string.
        """
        names = self.network.node_names
        P = np.array(
            [steady_state.pressures.get(name, 0.0) for name in names],
            dtype=float,
        )

        J = network_impedance_matrix(self.network, P)
        eigenvalues, eigenvectors = np.linalg.eig(J)

        # Sort by real part (most negative first = most stable)
        sort_idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        real_parts = eigenvalues.real
        if np.all(real_parts < 0):
            stability = "stable"
        elif np.all(real_parts > 0):
            stability = "unstable"
        else:
            stability = "saddle"

        return {
            "impedance_matrix": J,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "stability": stability,
        }

    def compare_with_rg(
        self,
        stability: StabilityAnalysis,
        steady_state: HydraulicSteadyState,
        tol: float = 1e-4,
    ) -> ComparisonResult:
        """Compare hydraulic eigenvalues with RG critical exponents.

        Both spectra are sorted by magnitude before comparison so that
        the pairing is consistent regardless of ordering conventions.

        Args:
            stability: RG :class:`StabilityAnalysis` at the
                corresponding fixed point.
            steady_state: Hydraulic :class:`HydraulicSteadyState` to
                compare against.
            tol: Tolerance for agreement.

        Returns:
            :class:`ComparisonResult` with the comparison outcome.
        """
        # Hydraulic eigenvalues at the steady state
        analysis = self.analyze_steady_state(steady_state)
        h_eigs = analysis["eigenvalues"]

        # RG critical exponents (theta_i = -eigenvalues of stability matrix)
        rg_exps = stability.critical_exponents

        # Sort both by magnitude for consistent comparison
        h_sorted = h_eigs[np.argsort(np.abs(h_eigs))]
        rg_sorted = rg_exps[np.argsort(np.abs(rg_exps))]

        # Trim to the shorter length (hydraulic network may include
        # reservoir nodes not present in the RG system)
        n_compare = min(len(h_sorted), len(rg_sorted))
        h_compare = h_sorted[:n_compare]
        rg_compare = rg_sorted[:n_compare]

        max_deviation = float(np.max(np.abs(np.abs(h_compare) - np.abs(rg_compare))))
        agrees = max_deviation < tol

        return ComparisonResult(
            hydraulic_eigenvalues=h_eigs,
            rg_critical_exponents=rg_exps,
            max_deviation=max_deviation,
            agrees=agrees,
        )
