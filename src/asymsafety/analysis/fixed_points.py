"""Fixed point finder for RG beta function systems.

A fixed point g* satisfies β_i(g*) = 0 for all couplings i.

The Gaussian fixed point (GFP) at g_i = 0 always exists.
Non-Gaussian fixed points (NGFPs) are found numerically using
multiple strategies to ensure completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve, root

from asymsafety.beta.system import BetaFunctionSystem


@dataclass
class FixedPoint:
    """A fixed point of the RG flow.

    Attributes:
        location: Coupling values at the fixed point.
        eigenvalues: Eigenvalues of the stability matrix.
        critical_exponents: θ_i = -eigenvalues (positive = relevant).
        eigenvectors: Right eigenvectors of the stability matrix.
    """

    location: dict[str, float]
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    critical_exponents: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_uv_attractive(self) -> bool:
        """True if all critical exponents have positive real part."""
        return bool(np.all(self.critical_exponents.real > 0))

    @property
    def relevant_directions(self) -> int:
        """Number of UV-relevant directions (Re(θ_i) > 0)."""
        return int(np.sum(self.critical_exponents.real > 0))

    @property
    def irrelevant_directions(self) -> int:
        """Number of UV-irrelevant directions (Re(θ_i) < 0)."""
        return int(np.sum(self.critical_exponents.real < 0))

    @property
    def is_gaussian(self) -> bool:
        """True if all couplings are (approximately) zero."""
        return all(abs(v) < 1e-10 for v in self.location.values())

    def __repr__(self) -> str:
        loc = ", ".join(f"{k}={v:.6f}" for k, v in self.location.items())
        n_rel = self.relevant_directions
        return f"FixedPoint({loc}, relevant={n_rel})"


class FixedPointFinder:
    """Find fixed points of a BetaFunctionSystem.

    Strategies:
        1. Grid scan + Newton refinement
        2. Multiple random initial conditions with fsolve
        3. Known solutions (Gaussian FP)
    """

    def __init__(self, beta_system: BetaFunctionSystem):
        self.system = beta_system
        self._rhs = None

    def _get_rhs(self):
        """Get the RHS function for root finding."""
        if self._rhs is None:
            rhs = self.system.rhs_vector()
            self._rhs = lambda y: rhs(0, y)
        return self._rhs

    def gaussian_fixed_point(self) -> FixedPoint:
        """Return the Gaussian (free) fixed point at g_i = 0."""
        names = self.system.coupling_names
        location = {name: 0.0 for name in names}

        # Compute stability at the GFP
        try:
            J = self.system.jacobian_numerical(location)
            eigenvalues = np.linalg.eig(J)[0]
            eigenvectors = np.linalg.eig(J)[1]
            critical_exponents = -eigenvalues
        except Exception:
            eigenvalues = np.array([])
            eigenvectors = np.array([])
            critical_exponents = np.array([])

        return FixedPoint(
            location=location,
            eigenvalues=eigenvalues,
            critical_exponents=critical_exponents,
            eigenvectors=eigenvectors,
        )

    def find_fixed_point(self, initial_guess: dict[str, float],
                          tol: float = 1e-10) -> FixedPoint | None:
        """Find a fixed point starting from an initial guess.

        Args:
            initial_guess: Starting point for the search.
            tol: Tolerance for convergence.

        Returns:
            FixedPoint if found, None if search failed.
        """
        names = self.system.coupling_names
        syms = self.system.coupling_symbols
        rhs = self._get_rhs()

        y0 = [initial_guess.get(name, 0.0) for name in names]

        try:
            sol, info, ier, msg = fsolve(rhs, y0, full_output=True)
            if ier != 1:
                return None

            # Check convergence
            residual = np.max(np.abs(info["fvec"]))
            if residual > tol:
                return None

            location = {name: float(sol[i]) for i, name in enumerate(names)}

            # Compute stability
            J = self.system.jacobian_numerical(location)
            evals, evecs = np.linalg.eig(J)
            critical_exponents = -evals

            return FixedPoint(
                location=location,
                eigenvalues=evals,
                critical_exponents=critical_exponents,
                eigenvectors=evecs,
            )
        except Exception:
            return None

    def find_all_fixed_points(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        n_grid: int = 10,
        n_random: int = 50,
        tol: float = 1e-10,
        merge_tol: float = 1e-4,
        prefilter_threshold: float | None = None,
        batch_backend: str = "numpy",
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> list[FixedPoint]:
        """Find all fixed points using multiple strategies.

        Args:
            bounds: Search region for each coupling.
            n_grid: Points per dimension for grid scan.
            n_random: Number of random initial conditions.
            tol: Convergence tolerance.
            merge_tol: Distance below which FPs are considered identical.
            prefilter_threshold: If set, batch-evaluate ``|β(x)|`` at every
                candidate first and only refine those below the threshold.
                Eliminates wasted fsolve calls on points far from any FP.
                A value of ~1.0 is a sensible default; ``None`` disables.
            batch_backend: Backend for the prefilter — ``"numpy"`` (default,
                always available), ``"jax"`` (requires ``asymsafety[gpu]``),
                or ``"auto"`` (use JAX if installed, else NumPy).
            parallel: If True, run the candidate fsolve refinements in a
                ProcessPoolExecutor. Best for systems with > ~50 candidates.
            max_workers: Worker count for the process pool; ``None`` uses
                the OS default.

        Returns:
            List of unique fixed points found.
        """
        names = self.system.coupling_names
        n_dim = len(names)

        if bounds is None:
            bounds = {name: (-2.0, 2.0) for name in names}

        # Build the full candidate set: grid ∪ random
        grids = [
            np.linspace(bounds[name][0], bounds[name][1], n_grid)
            for name in names
        ]
        mesh = np.meshgrid(*grids)
        grid_points = np.column_stack([m.ravel() for m in mesh])

        rng = np.random.default_rng(42)
        random_points = np.column_stack([
            rng.uniform(bounds[name][0], bounds[name][1], n_random)
            for name in names
        ])
        candidates = np.vstack([grid_points, random_points])

        # Optional batch prefilter
        if prefilter_threshold is not None:
            from asymsafety.compute.accelerated.fixed_points import (
                _select_batch_backend,
            )
            evaluator = _select_batch_backend(self.system, batch_backend)
            with np.errstate(divide="ignore", invalid="ignore"):
                norms = evaluator.evaluate_norms(candidates)
            mask = (norms < prefilter_threshold) | np.isnan(norms)
            candidates = candidates[mask]

        # Always include the Gaussian FP
        found: list[FixedPoint] = [self.gaussian_fixed_point()]

        if len(candidates) == 0:
            return found

        guesses = [
            {names[i]: float(candidates[j, i]) for i in range(n_dim)}
            for j in range(len(candidates))
        ]

        # Refinement: serial or parallel
        if parallel and len(guesses) > 1:
            results = self._refine_parallel(guesses, tol, max_workers)
        else:
            results = [self.find_fixed_point(g, tol=tol) for g in guesses]

        for fp in results:
            if fp is not None:
                self._add_if_new(found, fp, merge_tol, bounds)

        return found

    def _refine_parallel(
        self,
        guesses: list[dict[str, float]],
        tol: float,
        max_workers: int | None,
    ) -> list[FixedPoint | None]:
        """Refine candidate guesses with fsolve in a ProcessPoolExecutor.

        Guesses are bundled into per-worker batches (~`len/(4·workers)` each)
        so the per-pickle overhead amortizes over many sub-millisecond
        fsolve calls. ``BetaFunctionSystem.__getstate__`` strips the
        lambdified cache, so each worker rebuilds it lazily on first call
        within the chunk.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        n_workers = max_workers or os.cpu_count() or 1
        chunk_size = max(1, len(guesses) // (4 * n_workers))
        chunks = [
            guesses[i : i + chunk_size]
            for i in range(0, len(guesses), chunk_size)
        ]

        worker = partial(_refine_chunk, self.system, tol)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            chunked_results = list(pool.map(worker, chunks))

        # Flatten back to a flat list aligned with the original guesses
        return [fp for chunk_results in chunked_results for fp in chunk_results]

    @staticmethod
    def _add_if_new(found: list[FixedPoint], fp: FixedPoint,
                    merge_tol: float,
                    bounds: dict[str, tuple[float, float]]) -> None:
        """Add FP to list if it's not a duplicate and within bounds."""
        # Check bounds
        for name, val in fp.location.items():
            if name in bounds:
                lo, hi = bounds[name]
                if val < lo - 1 or val > hi + 1:
                    return

        # Check for duplicates
        for existing in found:
            dist = sum(
                (fp.location[k] - existing.location[k])**2
                for k in fp.location
            )**0.5
            if dist < merge_tol:
                return

        found.append(fp)


def _refine_chunk(
    system: BetaFunctionSystem,
    tol: float,
    guesses: list[dict[str, float]],
) -> list[FixedPoint | None]:
    """Worker that refines a batch of guesses inside a single subprocess.

    The subprocess builds one FixedPointFinder, which lazily lambdifies the
    system once and reuses it across all guesses in this chunk.
    """
    finder = FixedPointFinder(system)
    return [finder.find_fixed_point(g, tol=tol) for g in guesses]
