"""Mapping between RG beta function systems and hydraulic networks.

This module is the core bridge between the renormalisation-group and
hydraulic domains.  The :class:`RGToHydraulicMapper` translates every
element of a :class:`BetaFunctionSystem` into an equivalent hydraulic
network element so that steady states of the pipe network correspond
to fixed points of the RG flow, and the linearised impedance spectrum
reproduces the critical exponents.

Physical correspondence:

=========================  ====================================
RG concept                 Hydraulic analogue
=========================  ====================================
Coupling g_i               Pressure P_i at node i
Beta function beta_i(g)    Net flow equation at node i
Fixed point beta(g*)=0     Hydraulic steady state
Stability matrix M_ij      Linearised pipe-network impedance
Critical exponent theta_i  Eigenvalue of hydraulic impedance
Litim regulator             On/off valve (LitimValve)
Exponential regulator       Proportional valve (ExponentialValve)
=========================  ====================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from asymsafety.hydraulic.network import (
    HydraulicNetwork,
    HydraulicNode,
    HydraulicPipe,
)
from asymsafety.hydraulic.valves import ExponentialValve, LitimValve, Valve

if TYPE_CHECKING:
    from asymsafety.beta.system import BetaFunctionSystem


class RGToHydraulicMapper:
    """Map a :class:`BetaFunctionSystem` to a :class:`HydraulicNetwork`.

    The mapping proceeds as follows:

    1. Each coupling becomes a :class:`HydraulicNode`.
    2. The Jacobian (stability matrix) is evaluated at a reference point.
    3. Each non-zero off-diagonal entry ``M_ij`` produces a
       :class:`HydraulicPipe` from node *i* to node *j* whose friction
       factor encodes ``|M_ij|``.
    4. Each diagonal entry produces a pipe connecting the coupling node
       to a reservoir (fixed-pressure boundary) to capture self-coupling.
    5. Valves are added according to the chosen regulator type.

    Args:
        beta_system: The beta function system to map.
    """

    def __init__(self, beta_system: BetaFunctionSystem) -> None:
        self.beta_system = beta_system
        self._names = beta_system.coupling_names

    def build_network(
        self,
        regulator_type: str = "litim",
        pipe_params: dict | None = None,
        reference_point: dict[str, float] | None = None,
    ) -> HydraulicNetwork:
        """Build the hydraulic network from the beta function system.

        Args:
            regulator_type: ``"litim"`` for :class:`LitimValve` or
                ``"exponential"`` for :class:`ExponentialValve`.
            pipe_params: Optional overrides for default pipe geometry
                (keys: ``"length"``, ``"diameter"``, ``"base_friction"``).
            reference_point: Point in coupling space at which to
                evaluate the Jacobian.  Defaults to the origin.

        Returns:
            Fully constructed :class:`HydraulicNetwork`.
        """
        params = {
            "length": 1.0,
            "diameter": 0.1,
            "base_friction": 0.02,
        }
        if pipe_params is not None:
            params.update(pipe_params)

        names = self._names

        # 1. Create one HydraulicNode per coupling
        nodes: dict[str, HydraulicNode] = {}
        for name in names:
            nodes[name] = HydraulicNode(name=name, pressure=0.0)

        # Reference point for Jacobian evaluation
        if reference_point is None:
            reference_point = {name: 0.0 for name in names}

        # 2. Evaluate the Jacobian at the reference point
        J = self.beta_system.jacobian_numerical(reference_point)

        # 3. Build pipes from Jacobian entries
        pipes: list[HydraulicPipe] = []
        n = len(names)

        # Reservoir nodes for diagonal (self-coupling) entries
        for i in range(n):
            diag = abs(J[i, i])
            reservoir_name = f"_reservoir_{names[i]}"
            nodes[reservoir_name] = HydraulicNode(
                name=reservoir_name,
                pressure=0.0,
                is_reservoir=True,
            )
            friction = params["base_friction"] / max(diag, 1e-10)
            pipes.append(HydraulicPipe(
                node_from=names[i],
                node_to=reservoir_name,
                length=params["length"],
                diameter=params["diameter"],
                friction_factor=friction,
            ))

        # Off-diagonal entries -> inter-node pipes
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                entry = abs(J[i, j])
                if entry < 1e-14:
                    continue
                friction = params["base_friction"] / max(entry, 1e-10)
                pipes.append(HydraulicPipe(
                    node_from=names[i],
                    node_to=names[j],
                    length=params["length"],
                    diameter=params["diameter"],
                    friction_factor=friction,
                ))

        # 4. Create valves based on regulator type
        valves: list[Valve] = []
        if regulator_type == "litim":
            valves.append(LitimValve(cv=1.0, threshold=0.5))
        elif regulator_type == "exponential":
            valves.append(ExponentialValve(cv=1.0, p_scale=1.0))
        else:
            raise ValueError(
                f"Unknown regulator type: {regulator_type!r}. "
                "Choose 'litim' or 'exponential'."
            )

        return HydraulicNetwork(
            nodes=nodes,
            pipes=pipes,
            valves=valves,
        )

    def pressures_to_couplings(
        self, pressures: np.ndarray,
    ) -> dict[str, float]:
        """Convert a pressure vector to a coupling dictionary.

        Args:
            pressures: Array of pressures, one per coupling (in the
                same order as :attr:`BetaFunctionSystem.coupling_names`).

        Returns:
            Mapping from coupling name to value.
        """
        return {
            name: float(pressures[i])
            for i, name in enumerate(self._names)
        }

    def couplings_to_pressures(
        self, couplings: dict[str, float],
    ) -> np.ndarray:
        """Convert a coupling dictionary to a pressure vector.

        Args:
            couplings: Mapping from coupling name to value.

        Returns:
            Pressure vector with entries ordered by
            :attr:`BetaFunctionSystem.coupling_names`.
        """
        return np.array(
            [couplings.get(name, 0.0) for name in self._names],
            dtype=float,
        )
