"""Accelerated fixed point finder with batch pre-filtering and parallel search."""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve

from asymsafety.analysis.fixed_points import FixedPoint, FixedPointFinder
from asymsafety.analysis.stability import analyze_stability
from asymsafety.beta.system import BetaFunctionSystem


def _select_batch_backend(system: BetaFunctionSystem, request: str):
    """Build a batch evaluator, honoring the "auto" sentinel.

    ``"auto"`` picks ``"jax"`` when importable, else ``"numpy"``.
    """
    if request == "auto":
        try:
            import jax  # noqa: F401
            request = "jax"
        except ImportError:
            request = "numpy"
    return system.batch_evaluator(request)


class AcceleratedFixedPointFinder:
    """Fixed point finder using batch pre-filtering and parallel root-finding.

    Two-phase approach:
    1. Pre-filter: batch-evaluate |β(x)| at all grid points, keep only
       points where the norm is below a threshold (eliminates 90%+ of candidates)
    2. Refine: run fsolve on surviving candidates in parallel via the backend

    The batch evaluator is selectable via ``batch_backend``:
        - ``"auto"`` (default): JAX if installed, otherwise NumPy.
        - ``"numpy"``: CPU NumPy broadcasting.
        - ``"jax"``: JIT + vmap via JAX (requires ``asymsafety[gpu]`` extra).
    """

    def __init__(
        self,
        system: BetaFunctionSystem,
        backend=None,
        batch_backend: str = "auto",
    ):
        from asymsafety.compute.backends.base import LocalBackend
        self.system = system
        self.backend = backend or LocalBackend()
        self._evaluator = _select_batch_backend(system, batch_backend)
        self.batch_backend = batch_backend
        self._base_finder = FixedPointFinder(system)

    def find_all_fixed_points(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        n_grid: int = 10,
        n_random: int = 50,
        tol: float = 1e-10,
        merge_tol: float = 1e-4,
        prefilter_threshold: float = 1.0,
    ) -> list[FixedPoint]:
        """Find all fixed points with acceleration.

        Phase 1 (batch): evaluate |β| on entire grid, filter candidates.
        Phase 2 (parallel): run fsolve on candidates via backend.map().
        """
        names = self.system.coupling_names
        n_dim = len(names)

        if bounds is None:
            bounds = {name: (-2.0, 2.0) for name in names}

        # Build grid
        grids = [np.linspace(bounds[name][0], bounds[name][1], n_grid) for name in names]
        mesh = np.meshgrid(*grids)
        all_points = np.column_stack([m.ravel() for m in mesh])

        # Add random points
        rng = np.random.default_rng(42)
        random_points = np.column_stack([
            rng.uniform(bounds[name][0], bounds[name][1], n_random) for name in names
        ])
        all_points = np.vstack([all_points, random_points])

        # Phase 1: batch pre-filter
        norms = self._evaluator.evaluate_norms(all_points)
        candidate_mask = norms < prefilter_threshold
        # Also keep nan (might be near singularity = near a FP)
        candidate_mask |= np.isnan(norms)
        candidates = all_points[candidate_mask]

        if len(candidates) == 0:
            # No candidates found, fall back to base finder
            return self._base_finder.find_all_fixed_points(bounds, n_grid, n_random, tol, merge_tol)

        # Phase 2: parallel fsolve on candidates
        guesses = [{names[i]: candidates[j, i] for i in range(n_dim)} for j in range(len(candidates))]

        def _find_one(guess):
            return self._base_finder.find_fixed_point(guess, tol=tol)

        results = self.backend.map(_find_one, guesses)

        # Collect unique FPs
        found = [self._base_finder.gaussian_fixed_point()]
        for fp in results:
            if fp is not None:
                self._base_finder._add_if_new(found, fp, merge_tol, bounds)

        return found

    def find_fixed_point(self, initial_guess: dict[str, float], tol: float = 1e-10) -> FixedPoint | None:
        """Single fixed point search (delegates to base finder)."""
        return self._base_finder.find_fixed_point(initial_guess, tol)

    def gaussian_fixed_point(self) -> FixedPoint:
        return self._base_finder.gaussian_fixed_point()
