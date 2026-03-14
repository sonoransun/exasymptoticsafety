"""Accelerated flow integration for multiple trajectories in parallel."""

from __future__ import annotations
import numpy as np

from asymsafety.analysis.flow import FlowIntegrator, RGTrajectory
from asymsafety.beta.system import BetaFunctionSystem


class AcceleratedFlowIntegrator:
    """Parallel RG flow integrator.

    Distributes multiple independent trajectory integrations across
    the compute backend. Single trajectory integration is sequential
    (ODE solving), but multiple trajectories are embarrassingly parallel.
    """

    def __init__(self, system: BetaFunctionSystem, backend=None):
        from asymsafety.compute.backends.base import LocalBackend
        self.system = system
        self.backend = backend or LocalBackend()
        self._base = FlowIntegrator(system)

    def integrate(self, initial_conditions: dict[str, float], **kwargs) -> RGTrajectory:
        """Single trajectory (delegates to base integrator)."""
        return self._base.integrate(initial_conditions, **kwargs)

    def integrate_multiple(
        self,
        initial_conditions_list: list[dict[str, float]],
        t_span: tuple[float, float] = (-10, 10),
        **kwargs,
    ) -> list[RGTrajectory]:
        """Integrate multiple trajectories in parallel.

        Each trajectory is independent and distributed across the backend.
        """
        def _integrate_one(ic):
            integrator = FlowIntegrator(self.system)
            return integrator.integrate(ic, t_span=t_span, **kwargs)

        return self.backend.map(_integrate_one, initial_conditions_list)

    def integrate_grid(
        self,
        coupling_ranges: dict[str, tuple[float, float]],
        n_per_dim: int = 5,
        t_span: tuple[float, float] = (-10, 10),
        **kwargs,
    ) -> list[RGTrajectory]:
        """Integrate from a grid of initial conditions.

        Creates n_per_dim^n_couplings initial conditions on a grid
        and integrates all in parallel.
        """
        names = self.system.coupling_names
        grids = [np.linspace(coupling_ranges.get(name, (-1, 1))[0],
                             coupling_ranges.get(name, (-1, 1))[1],
                             n_per_dim) for name in names]
        mesh = np.meshgrid(*grids)
        points = np.column_stack([m.ravel() for m in mesh])

        ic_list = [{names[i]: points[j, i] for i in range(len(names))} for j in range(len(points))]
        return self.integrate_multiple(ic_list, t_span=t_span, **kwargs)
