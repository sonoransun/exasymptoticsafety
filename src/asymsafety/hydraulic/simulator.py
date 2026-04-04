"""Transient and steady-state simulation of hydraulic networks.

Mirrors the RG flow integration (:mod:`asymsafety.analysis.flow`) and
fixed-point search (:mod:`asymsafety.analysis.fixed_points`) in the
hydraulic domain.  Uses :func:`scipy.integrate.solve_ivp` for transient
simulations and :func:`scipy.optimize.fsolve` for steady-state search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from asymsafety.hydraulic.equations import (
    assemble_flow_equations,
    network_impedance_matrix,
)

if TYPE_CHECKING:
    from asymsafety.hydraulic.network import HydraulicNetwork


@dataclass
class HydraulicTrajectory:
    """Time-resolved pressure history at every node.

    Mirrors :class:`~asymsafety.analysis.flow.RGTrajectory` with
    hydraulic quantities.

    Attributes:
        t_values: Time array.
        pressure_values: Mapping from node name to pressure time-series.
        node_names: Ordered list of node names.
        success: Whether the ODE integration succeeded.
    """

    t_values: np.ndarray
    pressure_values: dict[str, np.ndarray]
    node_names: list[str]
    success: bool = True


@dataclass
class HydraulicSteadyState:
    """Steady-state solution of the hydraulic network.

    Mirrors :class:`~asymsafety.analysis.fixed_points.FixedPoint` with
    hydraulic quantities.

    Attributes:
        pressures: Mapping from node name to steady-state pressure.
        eigenvalues: Eigenvalues of the linearised impedance matrix.
        stability: Classification string: ``"stable"``, ``"unstable"``,
            or ``"saddle"``.
    """

    pressures: dict[str, float]
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    stability: str = "unknown"


class HydraulicSimulator:
    """Simulate transient and steady-state behaviour of a hydraulic network.

    Args:
        network: The :class:`HydraulicNetwork` to simulate.
    """

    def __init__(self, network: HydraulicNetwork) -> None:
        self.network = network

    def simulate_transient(
        self,
        P_init: dict[str, float],
        t_span: tuple[float, float] = (0.0, 10.0),
        n_points: int = 500,
    ) -> HydraulicTrajectory:
        """Run a transient ODE simulation of network pressures.

        Integrates the nodal flow-balance equations using
        ``scipy.integrate.solve_ivp`` with the RK45 method, matching the
        integration style of
        :meth:`~asymsafety.analysis.flow.FlowIntegrator.integrate`.

        Args:
            P_init: Initial pressures keyed by node name.
            t_span: ``(t_start, t_end)`` integration interval.
            n_points: Number of output time-points.

        Returns:
            :class:`HydraulicTrajectory` with the pressure history.
        """
        names = self.network.node_names
        y0 = np.array([P_init.get(name, 0.0) for name in names], dtype=float)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        rhs = assemble_flow_equations(self.network)

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )

        pressure_values = {
            name: sol.y[i] for i, name in enumerate(names)
        }

        return HydraulicTrajectory(
            t_values=sol.t,
            pressure_values=pressure_values,
            node_names=names,
            success=sol.success,
        )

    def find_steady_state(
        self,
        P_init: dict[str, float],
        tol: float = 1e-10,
    ) -> HydraulicSteadyState | None:
        """Find a steady state of the hydraulic network.

        Uses :func:`scipy.optimize.fsolve`, mirroring the approach of
        :class:`~asymsafety.analysis.fixed_points.FixedPointFinder`.

        Args:
            P_init: Initial guess for node pressures.
            tol: Convergence tolerance on the residual.

        Returns:
            :class:`HydraulicSteadyState` if convergence succeeds,
            ``None`` otherwise.
        """
        names = self.network.node_names
        y0 = np.array([P_init.get(name, 0.0) for name in names], dtype=float)

        rhs = assemble_flow_equations(self.network)
        rhs_static = lambda y: rhs(0.0, y)

        try:
            sol, info, ier, msg = fsolve(rhs_static, y0, full_output=True)

            if ier != 1:
                return None

            residual = np.max(np.abs(info["fvec"]))
            if residual > tol:
                return None

            pressures = {name: float(sol[i]) for i, name in enumerate(names)}

            # Compute impedance eigenvalues for stability classification
            J = network_impedance_matrix(self.network, sol)
            eigenvalues = np.linalg.eig(J)[0]

            # Classify stability
            real_parts = eigenvalues.real
            if np.all(real_parts < 0):
                stability = "stable"
            elif np.all(real_parts > 0):
                stability = "unstable"
            else:
                stability = "saddle"

            return HydraulicSteadyState(
                pressures=pressures,
                eigenvalues=eigenvalues,
                stability=stability,
            )
        except Exception:
            return None
