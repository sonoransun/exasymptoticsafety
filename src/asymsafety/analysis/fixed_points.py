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
    ) -> list[FixedPoint]:
        """Find all fixed points using multiple strategies.

        Args:
            bounds: Search region for each coupling.
            n_grid: Points per dimension for grid scan.
            n_random: Number of random initial conditions.
            tol: Convergence tolerance.
            merge_tol: Distance below which FPs are considered identical.

        Returns:
            List of unique fixed points found.
        """
        names = self.system.coupling_names
        n_dim = len(names)

        if bounds is None:
            bounds = {name: (-2.0, 2.0) for name in names}

        found: list[FixedPoint] = []

        # Always include the Gaussian FP
        gfp = self.gaussian_fixed_point()
        found.append(gfp)

        # Strategy 1: Grid scan
        grids = [
            np.linspace(bounds[name][0], bounds[name][1], n_grid)
            for name in names
        ]
        mesh = np.meshgrid(*grids)
        points = np.column_stack([m.ravel() for m in mesh])

        for point in points:
            guess = {names[i]: point[i] for i in range(n_dim)}
            fp = self.find_fixed_point(guess, tol=tol)
            if fp is not None:
                self._add_if_new(found, fp, merge_tol, bounds)

        # Strategy 2: Random initial conditions
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            guess = {}
            for name in names:
                lo, hi = bounds[name]
                guess[name] = rng.uniform(lo, hi)
            fp = self.find_fixed_point(guess, tol=tol)
            if fp is not None:
                self._add_if_new(found, fp, merge_tol, bounds)

        return found

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
