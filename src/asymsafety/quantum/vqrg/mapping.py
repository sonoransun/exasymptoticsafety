"""Linear map between coupling space and circuit rotation angles.

Maps each coupling ``g_i`` in ``[g_min, g_max]`` to a rotation angle
``theta_i`` in ``[0, pi]`` and vice-versa.  This bijection defines
the coordinate chart that connects the physical coupling space to
the variational circuit parameter space.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class CouplingAngleMap:
    """Linear map between coupling space and circuit rotation angles.

    ``g_i in [g_min, g_max]  <->  theta_i in [0, pi]``

    The mapping is:
        ``theta = pi * (g - g_min) / (g_max - g_min)``
        ``g     = g_min + (g_max - g_min) * theta / pi``
    """

    def __init__(self, coupling_ranges: dict[str, tuple[float, float]]) -> None:
        """Initialise the map.

        Args:
            coupling_ranges: Ordered mapping from coupling name to
                ``(g_min, g_max)`` bounds, e.g.
                ``{"g": (0.0, 2.0), "lambda": (-0.5, 0.5)}``.
        """
        self._ranges = OrderedDict(coupling_ranges)
        self._names = list(self._ranges.keys())
        self._lo = np.array([self._ranges[n][0] for n in self._names])
        self._hi = np.array([self._ranges[n][1] for n in self._names])
        self._span = self._hi - self._lo

    # ------------------------------------------------------------------
    # Forward / inverse maps
    # ------------------------------------------------------------------

    def couplings_to_angles(self, couplings: dict[str, float]) -> np.ndarray:
        """Map coupling values to rotation angles theta in ``[0, pi]``.

        Args:
            couplings: Dictionary ``{name: value}`` for every coupling.

        Returns:
            1-D array of angles with shape ``(n_couplings,)``.
        """
        g = np.array([couplings[n] for n in self._names])
        return np.pi * (g - self._lo) / self._span

    def angles_to_couplings(self, angles: np.ndarray) -> dict[str, float]:
        """Map rotation angles back to coupling values.

        Args:
            angles: 1-D array of angles in ``[0, pi]``.

        Returns:
            Dictionary ``{name: value}`` for every coupling.
        """
        g = self._lo + self._span * np.asarray(angles) / np.pi
        return {name: float(g[i]) for i, name in enumerate(self._names)}

    # ------------------------------------------------------------------
    # Jacobian of the map (used for metric transformations)
    # ------------------------------------------------------------------

    def jacobian_angles_to_couplings(self) -> np.ndarray:
        """Return d(g)/d(theta) as a diagonal matrix.

        Since the map is linear and decoupled, the Jacobian is a
        diagonal matrix with entries ``(g_max - g_min) / pi``.

        Returns:
            Array of shape ``(n_couplings, n_couplings)``.
        """
        return np.diag(self._span / np.pi)

    def jacobian_couplings_to_angles(self) -> np.ndarray:
        """Return d(theta)/d(g) as a diagonal matrix.

        Returns:
            Array of shape ``(n_couplings, n_couplings)``.
        """
        return np.diag(np.pi / self._span)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coupling_names(self) -> list[str]:
        """Ordered list of coupling names."""
        return list(self._names)

    @property
    def n_couplings(self) -> int:
        """Number of couplings in the map."""
        return len(self._names)
