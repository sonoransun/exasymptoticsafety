"""Koopman operator analysis of RG flows.

The Koopman operator K acts on observables phi : coupling_space -> R
and advances them by one RG time step:

    (K phi)(g) = phi(F(g))

where F is the (discrete) RG flow map.  Although the original dynamics
is nonlinear in coupling space, K is a *linear* operator on the
(infinite-dimensional) space of observables.

Extended Dynamic Mode Decomposition (EDMD) approximates K in a
finite dictionary of observables (monomials in the deviations
delta_g = g - g*).  Near the fixed point the leading Koopman
eigenvalues should reproduce the eigenvalues of the stability matrix,
while higher-order dictionary elements capture nonlinear corrections.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from asymsafety.analysis.fixed_points import FixedPoint
from asymsafety.analysis.flow import FlowIntegrator, RGTrajectory
from asymsafety.analysis.stability import StabilityAnalysis
from asymsafety.beta.system import BetaFunctionSystem
from asymsafety.transforms._types import ComparisonResult, KoopmanResult


class ClassicalKoopmanOperator:
    """EDMD-based Koopman analysis of the RG flow.

    Args:
        system: The beta function system.
        fp: The fixed point about which to expand.
    """

    def __init__(
        self, system: BetaFunctionSystem, fp: FixedPoint
    ) -> None:
        self._system = system
        self._fp = fp
        self._coupling_names = system.coupling_names
        self._n = system.dimension
        self._fp_array = np.array([
            fp.location[name] for name in self._coupling_names
        ])

    def compute_edmd(
        self,
        trajectories: list[RGTrajectory],
        dictionary_degree: int = 4,
    ) -> KoopmanResult:
        """Compute the Koopman operator via Extended Dynamic Mode Decomposition.

        Steps:
            1. Build a monomial dictionary {delta_g^alpha : |alpha| <= D}
               centred on the fixed point.
            2. Lift trajectory snapshots into dictionary space to form
               matrices Psi_X (current snapshots) and Psi_Y (next
               snapshots).
            3. Solve the least-squares problem
               K = (Psi_X^T Psi_X)^{-1} Psi_X^T Psi_Y.
            4. Eigendecompose K to obtain Koopman eigenvalues, modes,
               and eigenfunctions.

        Args:
            trajectories: List of RG trajectories used as training data.
            dictionary_degree: Maximum total degree D of the monomial
                dictionary.

        Returns:
            KoopmanResult with eigenvalues, eigenfunctions, modes, and
            dictionary size.
        """
        exponents = self._build_dictionary(dictionary_degree)
        dict_size = len(exponents)

        # Collect snapshot pairs from all trajectories
        psi_x_list: list[np.ndarray] = []
        psi_y_list: list[np.ndarray] = []

        for traj in trajectories:
            # Extract the trajectory as a (n_times, n_couplings) array
            values = np.column_stack([
                traj.coupling_values[name]
                for name in self._coupling_names
            ])
            n_times = values.shape[0]

            for k in range(n_times - 1):
                delta_x = values[k] - self._fp_array
                delta_y = values[k + 1] - self._fp_array

                psi_x_list.append(
                    self._evaluate_dictionary(delta_x, exponents)
                )
                psi_y_list.append(
                    self._evaluate_dictionary(delta_y, exponents)
                )

        Psi_X = np.array(psi_x_list)  # (n_snapshots, dict_size)
        Psi_Y = np.array(psi_y_list)  # (n_snapshots, dict_size)

        # Least-squares solution: K = pinv(Psi_X) @ Psi_Y
        K, _, _, _ = np.linalg.lstsq(Psi_X, Psi_Y, rcond=None)

        # Eigendecomposition of the Koopman matrix
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Sort by eigenvalue magnitude (largest first)
        sort_idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Koopman modes: project eigenfunctions back to observable space.
        # Using the first n_couplings dictionary elements (the linear
        # ones) as the canonical observables.
        modes = eigenvectors[:self._n, :].T  # (n_modes, n_couplings)

        return KoopmanResult(
            eigenvalues=eigenvalues,
            eigenfunctions=eigenvectors,
            modes=modes,
            dictionary_size=dict_size,
        )

    def _build_dictionary(
        self, degree: int
    ) -> list[tuple[int, ...]]:
        """Generate multi-index tuples for the monomial dictionary.

        Returns all tuples (a_1, ..., a_n) with a_1 + ... + a_n <= degree
        and a_i >= 0, ordered by total degree.

        Args:
            degree: Maximum total degree D.

        Returns:
            List of exponent tuples.
        """
        n = self._n
        exponents: list[tuple[int, ...]] = []

        # Generate all multi-indices with total degree <= degree
        for total_deg in range(degree + 1):
            exponents.extend(
                self._partitions(n, total_deg)
            )

        return exponents

    @staticmethod
    def _partitions(
        n_vars: int, total: int
    ) -> list[tuple[int, ...]]:
        """Generate all n_vars-tuples of non-negative integers summing to total."""
        if n_vars == 1:
            return [(total,)]

        result: list[tuple[int, ...]] = []
        for first in range(total + 1):
            for rest in ClassicalKoopmanOperator._partitions(
                n_vars - 1, total - first
            ):
                result.append((first, *rest))

        return result

    @staticmethod
    def _evaluate_dictionary(
        point: np.ndarray, exponents: list[tuple[int, ...]]
    ) -> np.ndarray:
        """Evaluate all dictionary monomials at a point.

        For exponent tuple alpha = (a_1, ..., a_n), the monomial is

            psi_alpha(delta_g) = prod_i delta_g_i^{a_i}

        Args:
            point: Displacement from the fixed point, shape
                (n_couplings,).
            exponents: List of exponent tuples from ``_build_dictionary``.

        Returns:
            Array of monomial values, shape (dictionary_size,).
        """
        result = np.empty(len(exponents))
        for k, alpha in enumerate(exponents):
            result[k] = np.prod(point ** np.array(alpha))
        return result

    def compare_with_stability(
        self,
        stability: StabilityAnalysis,
        trajectories: list[RGTrajectory] | None = None,
        dt: float = 0.01,
        n_trajectories: int = 40,
        displacement: float = 1e-5,
        dictionary_degree: int = 1,
        tol: float = 1e-3,
        seed: int = 0,
    ) -> ComparisonResult:
        """Compare EDMD Koopman eigenvalues with stability eigenvalues.

        Near the fixed point, the Koopman operator restricted to the
        linear dictionary is exp(M dt) (spectral-mapping theorem), so
        the continuous-time exponents recovered from the EDMD
        eigenvalues,

            mu_K = log(eig K) / dt,

        must reproduce the eigenvalues of the stability matrix M.
        This method actually runs EDMD: it integrates ``n_trajectories``
        short two-snapshot trajectories of the beta system from random
        initial displacements of size ``displacement`` around the fixed
        point (unless ``trajectories`` is supplied), calls
        :meth:`compute_edmd`, and matches each stability eigenvalue to
        the nearest ``log(eig K)/dt``. The comparison is genuine: a
        wrong Jacobian, fixed point, or EDMD implementation makes it
        fail.

        Args:
            stability: A completed stability analysis at the same
                fixed point.
            trajectories: Optional pre-computed training trajectories.
                Snapshots must be uniformly spaced by ``dt``; when None,
                synthetic near-linear trajectories are generated.
            dt: Snapshot spacing in RG time (the Koopman step).
            n_trajectories: Number of synthetic trajectories.
            displacement: Gaussian scale of the initial displacements
                from the fixed point (small, to stay in the linear
                regime).
            dictionary_degree: Monomial dictionary degree for EDMD.
                Degree 1 (constant + linear) isolates the linearised
                Koopman block.
            tol: Agreement tolerance on max |mu_K - mu(M)|; the EDMD
                reconstruction error on near-linear data is ~1e-5.
            seed: Seed for the initial-displacement RNG.

        Returns:
            ComparisonResult with ``values_a`` the matched
            ``log(eig K)/dt`` and ``values_b`` the stability
            eigenvalues, aligned entry by entry.
        """
        # Stability eigenvalues (of M)
        stab_evals = np.asarray(stability.eigenvalues, dtype=complex)

        if trajectories is None:
            integrator = FlowIntegrator(self._system)
            rng = np.random.default_rng(seed)
            trajectories = []
            for _ in range(n_trajectories):
                delta = rng.normal(scale=displacement, size=self._n)
                ic = {
                    name: self._fp.location[name] + delta[i]
                    for i, name in enumerate(self._coupling_names)
                }
                trajectories.append(integrator.integrate(
                    ic, t_span=(0.0, dt), t_eval=np.array([0.0, dt]),
                ))

        edmd = self.compute_edmd(
            trajectories, dictionary_degree=dictionary_degree
        )

        # Continuous-time exponents from the discrete Koopman spectrum.
        lam = edmd.eigenvalues[np.abs(edmd.eigenvalues) > 1e-12]
        mu_K = np.log(lam) / dt

        # Match each stability eigenvalue to the nearest EDMD exponent.
        koopman_evals = np.array([
            mu_K[np.argmin(np.abs(mu_K - mu))] for mu in stab_evals
        ])

        max_deviation = float(
            np.max(np.abs(koopman_evals - stab_evals))
        )

        return ComparisonResult(
            method_a="koopman_edmd",
            method_b="stability_matrix",
            values_a=koopman_evals,
            values_b=stab_evals,
            max_deviation=max_deviation,
            agrees=max_deviation < tol,
        )
