"""Accelerated parameter continuation with parallel cold-start option."""

from __future__ import annotations
import numpy as np

from asymsafety.analysis.fixed_points import FixedPoint, FixedPointFinder
from asymsafety.analysis.continuation import ContinuationResult
from asymsafety.beta.system import BetaFunctionSystem


class AcceleratedContinuation:
    """Parameter continuation with optional parallel cold-start.

    Standard continuation is sequential (warm-start from previous FP).
    Cold-start mode runs all parameter values independently in parallel.
    """

    def __init__(self, backend=None):
        from asymsafety.compute.backends.base import LocalBackend
        self.backend = backend or LocalBackend()

    def continuation(
        self,
        system_builder,
        parameter_name: str,
        parameter_values: np.ndarray,
        initial_guess: dict[str, float],
        mode: str = "warm",
    ) -> ContinuationResult:
        """Track a fixed point as a parameter varies.

        Args:
            system_builder: Callable(param_value) -> BetaFunctionSystem
            parameter_name: Name of the swept parameter
            parameter_values: Array of parameter values
            initial_guess: Initial FP guess
            mode: "warm" (sequential, warm-start) or "cold" (parallel, independent)
        """
        if mode == "cold":
            return self._cold_start(system_builder, parameter_name, parameter_values, initial_guess)
        else:
            return self._warm_start(system_builder, parameter_name, parameter_values, initial_guess)

    def _warm_start(self, system_builder, parameter_name, parameter_values, initial_guess):
        """Sequential continuation with warm start (standard approach)."""
        from asymsafety.analysis.continuation import continuation
        return continuation(system_builder, parameter_name, parameter_values, initial_guess)

    def _cold_start(self, system_builder, parameter_name, parameter_values, initial_guess):
        """Parallel continuation with cold start (independent searches)."""

        def _find_for_param(param_val):
            system = system_builder(param_val)
            finder = FixedPointFinder(system)
            return finder.find_fixed_point(initial_guess)

        fps = self.backend.map(_find_for_param, list(parameter_values))

        return ContinuationResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            fixed_points=fps,
        )
