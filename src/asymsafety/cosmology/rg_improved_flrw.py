"""RG-improved Friedmann-Lemaître-Robertson-Walker cosmology.

Promotes G and Λ in the Friedmann equations to scale-dependent quantities
along an RG trajectory:

    H² = (8πG(t)/3) ρ(t) - k_curv/a² + Λ(t)/3,

where the cutoff is identified via either a time matching ``k = ξ/t`` or
a Hubble matching ``k = ξ H(t)``.

The leading phenomenology is a Planck-scale modification at very early
times that may resolve the initial singularity (Bonanno-Reuter cosmology),
plus a possible late-time effect on dark-energy interpretations of Λ.

References:
    Bonanno & Reuter (2002), Phys. Rev. D 65, 043508 [hep-th/0106133]
    Reuter & Weyer (2004), Phys. Rev. D 69, 104022 [hep-th/0311196]
    Platania (2020), Eur. Phys. J. C 80, 470 (review)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.cosmology.scale_identification import (
    HubbleScale,
    ScaleIdentification,
)


@dataclass
class RGImprovedFLRW:
    """Modified Friedmann equation with running G(k(t)) and Λ(k(t)).

    Args:
        trajectory: RGTrajectory containing ``g`` and (optionally)
            ``lambda`` couplings.
        scale: Identification ``k(t)``. Default: time-inverse matching.
        k0: Reference scale (Planck scale; defaults to 1).
        curvature: Spatial curvature ``k_curv`` (-1, 0, +1) in units of
            ``H_0² a_0²``.
        equation_of_state: Constant ``w`` for ``p = w ρ``. Defaults to
            radiation (``w = 1/3``); use ``0`` for matter.
    """

    trajectory: RGTrajectory
    scale: ScaleIdentification = None  # type: ignore[assignment]
    k0: float = 1.0
    curvature: float = 0.0
    equation_of_state: float = 1.0 / 3.0

    def __post_init__(self):
        if self.scale is None:
            self.scale = HubbleScale(xi=1.0)
        if "g" not in self.trajectory.coupling_names:
            raise ValueError(
                "Trajectory must contain 'g' for the running Newton constant."
            )

    # -- Coupling lookups --------------------------------------------------

    def _t_of_k(self, k: float) -> float:
        """RG time ``t = log(k / k0)``."""
        return float(np.log(max(k, 1e-300) / self.k0))

    def G(self, k: float) -> float:
        """Dimensionful Newton constant at RG scale ``k``."""
        t = self._t_of_k(k)
        return float(self.trajectory.at_scale(t)["g"]) / k**2

    def Lambda(self, k: float) -> float:
        """Dimensionful cosmological constant at RG scale ``k``."""
        if "lambda" not in self.trajectory.coupling_names:
            return 0.0
        t = self._t_of_k(k)
        return float(self.trajectory.at_scale(t)["lambda"]) * k**2

    # -- Friedmann equation ------------------------------------------------

    def hubble_squared(
        self, t: float, a: float, rho: float, k: float
    ) -> float:
        """Right-hand side ``H² = (8πG/3)ρ - k_curv/a² + Λ/3``."""
        G_k = self.G(k)
        Lam_k = self.Lambda(k)
        return (
            (8.0 * np.pi * G_k / 3.0) * rho
            - self.curvature / a**2
            + Lam_k / 3.0
        )

    def integrate(
        self,
        a0: float = 1e-4,
        t_span: tuple[float, float] = (1e-2, 10.0),
        rho0: float = 1.0,
        n_steps: int = 200,
    ) -> dict[str, np.ndarray]:
        """Integrate the modified Friedmann equation forward in cosmic time.

        Uses Hubble-matching ``k = ξ H(t)`` when ``scale`` is a
        ``HubbleScale``; otherwise uses ``scale.k_of_t(t)``.

        Args:
            a0: Initial scale factor.
            t_span: ``(t_start, t_end)`` cosmic-time interval.
            rho0: Initial energy density at ``t_start``.
            n_steps: Number of stored time points.

        Returns:
            Dict with arrays ``t``, ``a``, ``H``, ``rho``, ``G``, ``Lambda``.
        """
        w = self.equation_of_state

        def k_of(t: float, a: float, rho: float) -> float:
            # Best-effort scale: prefer Hubble matching if available
            try:
                # Attempt without H
                return float(self.scale.k_of_t(t))
            except Exception:  # noqa: BLE001
                # HubbleScale path: estimate H from current rho, a
                G_guess = self.trajectory.at_scale(self._t_of_k(1.0 / t))["g"]
                G_dim = G_guess / (1.0 / t) ** 2
                H_est = np.sqrt(
                    max(
                        (8.0 * np.pi * G_dim / 3.0) * rho
                        - self.curvature / a**2,
                        0.0,
                    )
                )
                return float(self.scale.k_of_t(t, hubble=H_est))

        def rhs(t, y):
            a, rho = y
            k = k_of(t, a, rho)
            H_sq = self.hubble_squared(t, a, rho, k)
            H = np.sqrt(max(H_sq, 0.0))
            da = a * H
            drho = -3.0 * H * (rho + w * rho)
            return [da, drho]

        t_eval = np.linspace(t_span[0], t_span[1], n_steps)
        sol = solve_ivp(
            rhs, t_span, [a0, rho0], t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
        )

        # Reconstruct H, G, Lambda along the trajectory
        H_arr = np.zeros_like(sol.t)
        G_arr = np.zeros_like(sol.t)
        Lam_arr = np.zeros_like(sol.t)
        for i, ti in enumerate(sol.t):
            ai, rhoi = sol.y[0, i], sol.y[1, i]
            ki = k_of(ti, ai, rhoi)
            H_arr[i] = np.sqrt(
                max(self.hubble_squared(ti, ai, rhoi, ki), 0.0)
            )
            G_arr[i] = self.G(ki)
            Lam_arr[i] = self.Lambda(ki)

        return {
            "t": sol.t,
            "a": sol.y[0],
            "rho": sol.y[1],
            "H": H_arr,
            "G": G_arr,
            "Lambda": Lam_arr,
        }
