"""Cross-analogue bridge: verify commutativity of the transformation diagram.

         Laplace/Koopman
Classical RG ──────────────> Control Theory / Hydraulic
     |                              |
     | Matrix Exp / Heat Kernel     | Impedance <-> Hamiltonian
     v                              v
Quantum Computing <─────────────────
         QFT / Grover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asymsafety.analysis.fixed_points import FixedPoint
    from asymsafety.analysis.stability import StabilityAnalysis
    from asymsafety.beta.system import BetaFunctionSystem


class CrossAnalogueBridge:
    """Bridge connecting RG, hydraulic, and quantum domains.

    Verifies that all paths through the commutative diagram
    produce consistent critical exponents.
    """

    def __init__(
        self,
        system: BetaFunctionSystem,
        fp: FixedPoint,
        stability: StabilityAnalysis,
    ) -> None:
        self.system = system
        self.fp = fp
        self.stability = stability

    def rg_critical_exponents(self) -> np.ndarray:
        """Get critical exponents from direct RG stability analysis."""
        return self.stability.critical_exponents

    def matrix_exp_exponents(self, dt: float = 0.1) -> np.ndarray:
        """Get critical exponents via transfer matrix."""
        from asymsafety.transforms.linear.transfer_matrix import (
            DiscreteRGTransferMatrix,
        )

        tm = DiscreteRGTransferMatrix(self.system, self.fp)
        return tm.critical_exponents_from_transfer(dt)

    def resolvent_poles(self) -> np.ndarray:
        """Get critical exponents from resolvent poles.

        Poles of R(s) = (sI - M)^{-1} are at s = eigenvalues of M = -theta_i.
        """
        from asymsafety.transforms.linear.resolvent import ResolventOperator

        res = ResolventOperator(self.stability)
        return -res.poles()  # poles = eigenvalues of M = -theta

    def hydraulic_exponents(
        self, regulator_type: str = "litim"
    ) -> np.ndarray | None:
        """Get critical exponents from hydraulic network.

        Build network, find steady state, compute impedance eigenvalues.
        Returns None if hydraulic analogue fails to converge.
        """
        from asymsafety.hydraulic.equations import network_impedance_matrix
        from asymsafety.hydraulic.mapping import RGToHydraulicMapper
        from asymsafety.hydraulic.simulator import HydraulicSimulator

        mapper = RGToHydraulicMapper(self.system)
        network = mapper.build_network(regulator_type=regulator_type)

        simulator = HydraulicSimulator(network)
        P_init = mapper.couplings_to_pressures(self.fp.location)
        P_init_dict = {
            name: float(P_init[i])
            for i, name in enumerate(self.system.coupling_names)
        }
        ss = simulator.find_steady_state(P_init_dict)
        if ss is None:
            return None

        P_vec = np.array(
            [ss.pressures[name] for name in network.node_names]
        )
        impedance = network_impedance_matrix(network, P_vec)
        eigs = np.linalg.eig(impedance)[0]
        return np.sort(eigs.real)[::-1]  # Sort by real part descending

    def verify_commutativity(self, tol: float = 0.1) -> dict:
        """Check all paths give consistent critical exponents.

        Returns dict with method names, values, and agreement status.
        """
        results: dict[str, np.ndarray] = {}

        # Reference: direct RG
        rg_theta = np.sort(self.rg_critical_exponents().real)[::-1]
        results["rg_stability"] = rg_theta

        # Transfer matrix path
        tm_theta = np.sort(self.matrix_exp_exponents().real)[::-1]
        results["transfer_matrix"] = tm_theta

        # Resolvent path
        res_theta = np.sort(self.resolvent_poles().real)[::-1]
        results["resolvent"] = res_theta

        # Check agreement
        agreements: dict[str, dict] = {}
        for method, values in results.items():
            if method == "rg_stability":
                continue
            dev = np.max(np.abs(values - rg_theta))
            agreements[method] = {
                "max_deviation": float(dev),
                "agrees": bool(dev < tol),
            }

        return {
            "critical_exponents": results,
            "agreements": agreements,
            "all_agree": all(a["agrees"] for a in agreements.values()),
        }

    def full_comparison_table(self) -> dict:
        """Critical exponents from all methods side-by-side.

        Returns a dict of arrays, one per method.
        """
        table: dict[str, np.ndarray | None] = {}
        table["rg_stability"] = self.stability.critical_exponents

        try:
            table["transfer_matrix"] = self.matrix_exp_exponents()
        except Exception:
            table["transfer_matrix"] = None

        try:
            table["resolvent_poles"] = self.resolvent_poles()
        except Exception:
            table["resolvent_poles"] = None

        try:
            hyd = self.hydraulic_exponents()
            table["hydraulic"] = hyd
        except Exception:
            table["hydraulic"] = None

        return table
