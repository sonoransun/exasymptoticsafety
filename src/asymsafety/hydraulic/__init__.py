"""Hydraulic flow network analogue for asymptotic safety.

Maps RG flow to a pipe network where steady states correspond to
fixed points and linearized impedance eigenvalues match critical exponents.
"""

from __future__ import annotations

from asymsafety.hydraulic.network import (
    HydraulicNetwork,
    HydraulicNode,
    HydraulicPipe,
)
from asymsafety.hydraulic.valves import ExponentialValve, LitimValve, Valve
from asymsafety.hydraulic.equations import (
    assemble_flow_equations,
    compute_pipe_flow,
    network_impedance_matrix,
)
from asymsafety.hydraulic.mapping import RGToHydraulicMapper
from asymsafety.hydraulic.simulator import (
    HydraulicSimulator,
    HydraulicSteadyState,
    HydraulicTrajectory,
)
from asymsafety.hydraulic.analysis import (
    ComparisonResult,
    HydraulicStabilityAnalysis,
)
from asymsafety.hydraulic.visualization import (
    plot_hydraulic_network,
    plot_transient_response,
)

__all__ = [
    "HydraulicNode",
    "HydraulicPipe",
    "HydraulicNetwork",
    "Valve",
    "LitimValve",
    "ExponentialValve",
    "compute_pipe_flow",
    "assemble_flow_equations",
    "network_impedance_matrix",
    "RGToHydraulicMapper",
    "HydraulicTrajectory",
    "HydraulicSteadyState",
    "HydraulicSimulator",
    "ComparisonResult",
    "HydraulicStabilityAnalysis",
    "plot_hydraulic_network",
    "plot_transient_response",
]
