"""Hydraulic impedance <-> transfer function bridge.

Connects the hydraulic domain (pipe network impedance matrix)
with control theory (transfer functions from Laplace transforms).
"""

from __future__ import annotations

import numpy as np


class ImpedanceBridge:
    """Bridge between hydraulic impedance and transfer functions.

    The hydraulic impedance matrix Z(s) in the Laplace domain
    relates pressure to flow: P(s) = Z(s) * Q(s).
    This is analogous to the RG transfer function.
    """

    def __init__(self, stability_matrix: np.ndarray) -> None:
        """Initialize from a stability matrix.

        Args:
            stability_matrix: M_ij from StabilityAnalysis.
        """
        self.M = stability_matrix
        self.n = stability_matrix.shape[0]

    def impedance_matrix(self, s: complex) -> np.ndarray:
        """Z(s) = sI - M (impedance in Laplace domain).

        This is the inverse of the resolvent: Z = R^{-1}.
        """
        return s * np.eye(self.n) - self.M

    def transfer_function_matrix(self, s: complex) -> np.ndarray:
        """H(s) = (sI - M)^{-1} (transfer function = resolvent).

        Relates input perturbation to output response.
        """
        return np.linalg.inv(self.impedance_matrix(s))

    def bode_data(
        self,
        omega_range: tuple[float, float] = (0.01, 100.0),
        n_points: int = 200,
    ) -> dict:
        """Compute Bode plot data: magnitude and phase vs frequency.

        Evaluates H(j*omega) along the imaginary axis.

        Args:
            omega_range: (omega_min, omega_max) frequency bounds.
            n_points: Number of frequency evaluation points.

        Returns:
            Dict with keys ``"omega"``, ``"magnitude"`` (dB),
            ``"phase"`` (degrees).  Magnitude and phase arrays have
            shape ``(n, n, n_points)``.
        """
        omega = np.logspace(
            np.log10(omega_range[0]),
            np.log10(omega_range[1]),
            n_points,
        )
        magnitude = np.zeros((self.n, self.n, n_points))
        phase = np.zeros((self.n, self.n, n_points))

        for k, w in enumerate(omega):
            H = self.transfer_function_matrix(1j * w)
            magnitude[:, :, k] = 20 * np.log10(np.abs(H) + 1e-30)
            phase[:, :, k] = np.degrees(np.angle(H))

        return {"omega": omega, "magnitude": magnitude, "phase": phase}

    def to_hydraulic_params(self) -> dict:
        """Convert stability matrix to equivalent hydraulic parameters.

        Maps M_ij entries to pipe resistances and node capacitances.

        Model: C * dP/dt = -G * P where G = -M, C = I (unit capacitance).
        G_ij represents conductance between nodes i and j.

        Returns:
            Dict with ``"conductance_matrix"`` (ndarray) and
            ``"resistances"`` (dict mapping ``"R_ij"`` to float).
        """
        G = -self.M.copy()
        # Diagonal: self-conductance (node to reservoir)
        # Off-diagonal: inter-node conductance
        resistances: dict[str, float] = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(G[i, j]) > 1e-10:
                    resistances[f"R_{i}{j}"] = 1.0 / abs(G[i, j])
        return {"conductance_matrix": G, "resistances": resistances}
