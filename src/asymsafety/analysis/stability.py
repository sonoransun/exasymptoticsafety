"""Stability analysis at fixed points.

The stability matrix M_ij = ∂β_i/∂g_j evaluated at a fixed point g*
determines the local flow structure.

Critical exponents θ_i = -eigenvalues of M:
    θ_i > 0 (real part): relevant direction (UV-attractive)
    θ_i < 0 (real part): irrelevant direction (UV-repulsive)
    θ_i complex: spiral flow

The number of relevant directions equals the number of free parameters
(physical predictions minus input measurements).
"""

from dataclasses import dataclass

import numpy as np

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.beta.system import BetaFunctionSystem


@dataclass
class StabilityAnalysis:
    """Complete stability analysis at a fixed point."""

    fixed_point: FixedPoint
    stability_matrix: np.ndarray
    eigenvalues: np.ndarray
    critical_exponents: np.ndarray
    eigenvectors: np.ndarray

    @property
    def relevant_eigenvalues(self) -> np.ndarray:
        """Critical exponents with Re(θ) > 0."""
        return self.critical_exponents[self.critical_exponents.real > 0]

    @property
    def irrelevant_eigenvalues(self) -> np.ndarray:
        """Critical exponents with Re(θ) < 0."""
        return self.critical_exponents[self.critical_exponents.real < 0]

    @property
    def marginal_eigenvalues(self) -> np.ndarray:
        """Critical exponents with Re(θ) ≈ 0."""
        return self.critical_exponents[
            np.abs(self.critical_exponents.real) < 1e-6
        ]

    @property
    def predictivity(self) -> int:
        """Number of predictions = total couplings - relevant directions."""
        return len(self.critical_exponents) - len(self.relevant_eigenvalues)

    def summary(self) -> str:
        """Human-readable summary of the stability analysis."""
        lines = [
            f"Fixed point: {self.fixed_point.location}",
            f"Critical exponents θ_i:",
        ]
        for i, theta in enumerate(self.critical_exponents):
            if abs(theta.imag) < 1e-10:
                lines.append(f"  θ_{i+1} = {theta.real:.6f} "
                           f"({'relevant' if theta.real > 0 else 'irrelevant'})")
            else:
                lines.append(f"  θ_{i+1} = {theta.real:.6f} ± {abs(theta.imag):.6f}i "
                           f"({'relevant' if theta.real > 0 else 'irrelevant'})")
        lines.append(f"Relevant directions: {len(self.relevant_eigenvalues)}")
        lines.append(f"UV attractive: {self.fixed_point.is_uv_attractive}")
        return "\n".join(lines)


def analyze_stability(system: BetaFunctionSystem,
                      fp: FixedPoint) -> StabilityAnalysis:
    """Perform stability analysis at a fixed point.

    Args:
        system: The beta function system.
        fp: The fixed point to analyze.

    Returns:
        Complete StabilityAnalysis.

    See Also:
        :func:`asymsafety.gui.visualization_3d.fixed_point_stability_3d`
            3D zoom on a single fixed point with eigenvector arrows
            consuming this :class:`StabilityAnalysis`.
        :func:`asymsafety.visualization.fixed_point_plot.plot_critical_exponents`
            2D plot of ``θ_i`` along a continuation parameter.
        :func:`asymsafety.visualization.phase_portrait.annotated_eh_phase_portrait`
            EH phase portrait with eigenvector arrows annotated by
            ``theta_i`` values.
    """
    # Compute stability matrix
    M = system.jacobian_numerical(fp.location)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Critical exponents θ = -eigenvalues
    critical_exponents = -eigenvalues

    # Sort by real part (most relevant first)
    sort_idx = np.argsort(-critical_exponents.real)
    critical_exponents = critical_exponents[sort_idx]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Update the fixed point with sorted values
    fp.eigenvalues = eigenvalues
    fp.critical_exponents = critical_exponents
    fp.eigenvectors = eigenvectors

    return StabilityAnalysis(
        fixed_point=fp,
        stability_matrix=M,
        eigenvalues=eigenvalues,
        critical_exponents=critical_exponents,
        eigenvectors=eigenvectors,
    )
