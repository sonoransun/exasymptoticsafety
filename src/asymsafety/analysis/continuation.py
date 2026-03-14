"""Parameter continuation: track fixed points as parameters vary.

Useful for studying how the Reuter fixed point changes with:
    - Number of matter fields (N_s, N_D, N_v)
    - Gauge parameters (α, β)
    - Spacetime dimension d (near d=4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from asymsafety.analysis.fixed_points import FixedPoint, FixedPointFinder
from asymsafety.beta.system import BetaFunctionSystem


@dataclass
class ContinuationResult:
    """Result of parameter continuation.

    Tracks fixed point location and critical exponents as a function
    of an external parameter.
    """

    parameter_name: str
    parameter_values: np.ndarray
    fixed_points: list[FixedPoint | None]

    @property
    def locations(self) -> dict[str, np.ndarray]:
        """Fixed point locations as arrays (NaN where FP doesn't exist)."""
        if not self.fixed_points or self.fixed_points[0] is None:
            return {}
        names = list(self.fixed_points[0].location.keys())
        result = {}
        for name in names:
            vals = []
            for fp in self.fixed_points:
                if fp is not None:
                    vals.append(fp.location[name])
                else:
                    vals.append(np.nan)
            result[name] = np.array(vals)
        return result

    @property
    def critical_exponents_array(self) -> np.ndarray:
        """Critical exponents as 2D array (parameter × exponent index)."""
        max_dim = 0
        for fp in self.fixed_points:
            if fp is not None and len(fp.critical_exponents) > max_dim:
                max_dim = len(fp.critical_exponents)

        result = np.full((len(self.parameter_values), max_dim), np.nan + 0j)
        for i, fp in enumerate(self.fixed_points):
            if fp is not None and len(fp.critical_exponents) > 0:
                n = len(fp.critical_exponents)
                result[i, :n] = fp.critical_exponents
        return result


def continuation(
    system_builder,
    parameter_name: str,
    parameter_values: np.ndarray,
    initial_guess: dict[str, float],
) -> ContinuationResult:
    """Track a fixed point as an external parameter varies.

    Args:
        system_builder: Callable(param_value) -> BetaFunctionSystem.
        parameter_name: Name of the external parameter.
        parameter_values: Array of parameter values to sweep.
        initial_guess: Initial guess for the fixed point at the first
                      parameter value.

    Returns:
        ContinuationResult tracking the FP across parameter values.
    """
    fixed_points: list[FixedPoint | None] = []
    current_guess = initial_guess.copy()

    for param_val in parameter_values:
        system = system_builder(param_val)
        finder = FixedPointFinder(system)
        fp = finder.find_fixed_point(current_guess)

        fixed_points.append(fp)

        # Update guess for next iteration (continuation)
        if fp is not None:
            current_guess = fp.location.copy()

    return ContinuationResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        fixed_points=fixed_points,
    )
