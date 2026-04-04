"""Flow equations for the hydraulic network.

Provides the Darcy-Weisbach pipe flow computation, assembly of the
nodal flow-balance ODE system, and the linearised impedance matrix
(analogous to the RG stability matrix / Jacobian).
"""

from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.hydraulic.network import HydraulicNetwork, HydraulicPipe


def compute_pipe_flow(
    pipe: HydraulicPipe,
    P_from: float,
    P_to: float,
    density: float = 1000.0,
    viscosity: float = 1e-3,
) -> float:
    """Darcy-Weisbach flow rate through a single pipe.

    In the turbulent regime the pressure drop is related to the
    volumetric flow rate *Q* by::

        DeltaP = (8 f L) / (pi^2 D^5) * Q |Q|

    Inverting gives::

        Q = sign(DeltaP) * sqrt(|DeltaP| * pi^2 * D^5 / (8 f L))

    Args:
        pipe: Pipe geometry and friction parameters.
        P_from: Pressure at the upstream node (Pa).
        P_to: Pressure at the downstream node (Pa).
        density: Fluid density (kg/m^3).
        viscosity: Dynamic viscosity (Pa.s).

    Returns:
        Volumetric flow rate through the pipe (m^3/s).  Positive when
        flow runs from *node_from* to *node_to*.
    """
    delta_p = P_from - P_to
    f = pipe.friction_factor
    L = pipe.length
    D = pipe.diameter

    # Resistance coefficient: DeltaP = R * Q|Q|  =>  R = 8fL / (pi^2 D^5)
    resistance = 8.0 * f * L / (math.pi ** 2 * D ** 5)
    if resistance < 1e-30:
        return 0.0

    sign = 1.0 if delta_p >= 0.0 else -1.0
    return sign * math.sqrt(abs(delta_p) / resistance)


def assemble_flow_equations(network: HydraulicNetwork) -> Callable:
    """Assemble the nodal flow-balance ODE right-hand side.

    For each non-reservoir node *i* the pressure evolves as::

        dP_i/dt = (sum of flows into node i) - (sum of flows out)

    Reservoir nodes have ``dP/dt = 0`` (fixed boundary conditions).

    The returned callable has signature ``f(t, P_vector) -> dP/dt``
    matching :meth:`BetaFunctionSystem.rhs_vector` output, so it can
    be integrated directly with :func:`scipy.integrate.solve_ivp`.

    Args:
        network: The hydraulic network to assemble equations for.

    Returns:
        A callable ``f(t, P)`` returning the time-derivative vector.
    """
    names = network.node_names
    name_to_idx = {name: i for i, name in enumerate(names)}
    n = network.n_nodes
    reservoir_flags = [network.nodes[name].is_reservoir for name in names]

    # Pre-collect pipe connectivity
    pipe_info: list[tuple[int, int, HydraulicPipe]] = []
    for pipe in network.pipes:
        i_from = name_to_idx.get(pipe.node_from)
        i_to = name_to_idx.get(pipe.node_to)
        if i_from is not None and i_to is not None:
            pipe_info.append((i_from, i_to, pipe))

    # Pre-collect valve info (valves connect sequential pairs of nodes)
    valve_info: list[tuple[int, int, object]] = []
    for valve in network.valves:
        # Valves are attached to the last pipes added (reservoir pipes)
        # They modify flow on those connections.
        pass  # Valves are handled via pipe friction modification

    def rhs(t: float, P: np.ndarray) -> np.ndarray:
        dP_dt = np.zeros(n)
        for i_from, i_to, pipe in pipe_info:
            Q = compute_pipe_flow(
                pipe, P[i_from], P[i_to],
                density=network.density,
                viscosity=network.viscosity,
            )
            # Flow Q goes from i_from to i_to:
            #   node i_from loses flow, node i_to gains flow
            dP_dt[i_from] -= Q
            dP_dt[i_to] += Q

        # Reservoir nodes: pressure is fixed
        for i, is_res in enumerate(reservoir_flags):
            if is_res:
                dP_dt[i] = 0.0

        return dP_dt

    return rhs


def network_impedance_matrix(
    network: HydraulicNetwork,
    P_steady: np.ndarray,
) -> np.ndarray:
    """Linearised Jacobian of the flow equations at a steady state.

    Computes ``J_ij = d(flow_eq_i) / dP_j`` evaluated at *P_steady*
    using central finite differences.  This is the hydraulic analogue
    of :meth:`BetaFunctionSystem.jacobian_numerical`.

    Args:
        network: The hydraulic network.
        P_steady: Pressure vector at the steady-state operating point.

    Returns:
        Jacobian matrix of shape ``(n_nodes, n_nodes)``.
    """
    rhs = assemble_flow_equations(network)
    n = network.n_nodes
    eps = 1e-7

    f0 = rhs(0.0, P_steady)
    J = np.zeros((n, n))

    for j in range(n):
        P_plus = P_steady.copy()
        P_minus = P_steady.copy()
        P_plus[j] += eps
        P_minus[j] -= eps
        f_plus = rhs(0.0, P_plus)
        f_minus = rhs(0.0, P_minus)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return J
