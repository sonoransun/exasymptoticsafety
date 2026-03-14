"""RG flow integration: solve ∂_t g_i = β_i(g) as an ODE system.

Integrates the RG flow equations using scipy.integrate.solve_ivp,
storing the trajectory as RGTrajectory objects for analysis and
visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from asymsafety.beta.system import BetaFunctionSystem


@dataclass
class RGTrajectory:
    """Solution of the RG flow equations.

    Attributes:
        t_values: RG time t = log(k/k_0) values.
        coupling_values: Dictionary mapping coupling names to arrays.
        coupling_names: Ordered list of coupling names.
    """

    t_values: np.ndarray
    coupling_values: dict[str, np.ndarray]
    coupling_names: list[str]
    success: bool = True

    def at_scale(self, t: float) -> dict[str, float]:
        """Interpolate coupling values at a given RG time.

        Args:
            t: RG time t = log(k/k_0).

        Returns:
            Dictionary of coupling values at scale t.
        """
        result = {}
        for name, values in self.coupling_values.items():
            result[name] = float(np.interp(t, self.t_values, values))
        return result

    @property
    def uv_values(self) -> dict[str, float]:
        """Coupling values at the UV end of the trajectory."""
        return self.at_scale(self.t_values[-1])

    @property
    def ir_values(self) -> dict[str, float]:
        """Coupling values at the IR end of the trajectory."""
        return self.at_scale(self.t_values[0])


class FlowIntegrator:
    """Integrate RG flow equations ∂_t g_i = β_i(g)."""

    def __init__(self, beta_system: BetaFunctionSystem):
        self.system = beta_system

    def integrate(
        self,
        initial_conditions: dict[str, float],
        t_span: tuple[float, float] = (-10, 10),
        method: str = "RK45",
        max_step: float = 0.1,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        t_eval: np.ndarray | None = None,
    ) -> RGTrajectory:
        """Integrate the RG flow from initial conditions.

        Args:
            initial_conditions: Initial coupling values.
            t_span: (t_start, t_end) range in RG time.
            method: ODE solver method.
            max_step: Maximum step size.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            t_eval: Times at which to store the solution.

        Returns:
            RGTrajectory with the solution.
        """
        names = self.system.coupling_names
        y0 = [initial_conditions.get(name, 0.0) for name in names]
        rhs = self.system.rhs_vector()

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        # Add event to detect singularities (e.g., λ → 1/2)
        def singularity_event(t, y):
            # Stop if any coupling becomes very large
            return 100.0 - np.max(np.abs(y))
        singularity_event.terminal = True

        sol = solve_ivp(
            rhs, t_span, y0,
            method=method,
            t_eval=t_eval,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
            events=[singularity_event],
            dense_output=True,
        )

        coupling_values = {
            name: sol.y[i] for i, name in enumerate(names)
        }

        return RGTrajectory(
            t_values=sol.t,
            coupling_values=coupling_values,
            coupling_names=names,
            success=sol.success,
        )

    def integrate_to_fixed_point(
        self,
        initial_conditions: dict[str, float],
        direction: str = "UV",
        t_max: float = 50.0,
        tol: float = 1e-6,
    ) -> RGTrajectory:
        """Integrate toward a fixed point until convergence.

        Args:
            initial_conditions: Starting point.
            direction: "UV" (t → +∞) or "IR" (t → -∞).
            t_max: Maximum integration time.
            tol: Convergence tolerance on |β_i|.

        Returns:
            RGTrajectory terminating near the fixed point.
        """
        if direction == "UV":
            t_span = (0, t_max)
        else:
            t_span = (0, -t_max)

        def convergence_event(t, y):
            betas = self.system.rhs_vector()(t, y)
            return np.max(np.abs(betas)) - tol
        convergence_event.terminal = True

        names = self.system.coupling_names
        y0 = [initial_conditions.get(name, 0.0) for name in names]
        rhs = self.system.rhs_vector()

        sol = solve_ivp(
            rhs, t_span, y0,
            method="RK45",
            events=[convergence_event],
            max_step=0.1,
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
        )

        coupling_values = {
            name: sol.y[i] for i, name in enumerate(names)
        }

        return RGTrajectory(
            t_values=sol.t,
            coupling_values=coupling_values,
            coupling_names=names,
            success=sol.success,
        )
