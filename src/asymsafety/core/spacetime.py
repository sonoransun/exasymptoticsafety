"""Spacetime configuration for FRG computations."""

from dataclasses import dataclass
from typing import Literal

import sympy


@dataclass(frozen=True)
class SpacetimeConfig:
    """Immutable spacetime configuration.

    Attributes:
        dimension: Spacetime dimension (integer or symbolic for dim-reg).
        signature: Metric signature convention.
        topology: Background topology for trace evaluation.
        is_foliated: Whether to use ADM/foliated formulation.
    """

    dimension: int | sympy.Symbol = 4
    signature: Literal["euclidean", "lorentzian"] = "euclidean"
    topology: str = "S4"
    is_foliated: bool = False

    def __post_init__(self):
        if self.is_foliated and self.topology == "S4":
            # S4 does not admit a global foliation
            object.__setattr__(self, "topology", "S1xS3")

    @property
    def spatial_dimension(self) -> int | sympy.Symbol:
        """Dimension of spatial slices: d - 1."""
        return self.dimension - 1

    @property
    def epsilon(self) -> int:
        """Signature factor: +1 for Euclidean, -1 for Lorentzian."""
        return 1 if self.signature == "euclidean" else -1
