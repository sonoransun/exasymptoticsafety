"""RG-improved Schwarzschild geometry (Bonanno-Reuter 2000).

The Schwarzschild solution in asymptotic safety is improved by promoting
Newton's constant to a position-dependent quantity ``G(r)`` obtained by
inserting the running coupling ``g(k)`` along an RG trajectory at the
scale ``k(r)``:

    G(r) = g(k(r)) / k(r)²,    Λ(r) = λ(k(r)) · k(r)².

The improved lapse function is ``f(r) = 1 - 2 G(r) M / r``.

Key physical predictions (compared to classical Schwarzschild):
    - For ``M`` larger than a critical mass ``M_crit``, two horizons
      exist (inner Cauchy horizon + outer event horizon).
    - At ``M = M_crit`` the two horizons merge — extremal black hole.
    - For ``M < M_crit`` no horizon exists at all — naked-mass remnant.
    - ``f(r) → 1`` as ``r → 0``, but only *linearly*: the default
      ``k = ξ/r`` identification gives ``G(r) ≃ g* r²/ξ²`` near the
      origin, hence ``f(r) ≃ 1 - 2 g* M r/ξ²`` with ``f'(0) ≠ 0``, and
      the Ricci scalar ``R ≃ 12 g* M/(ξ² r)`` still diverges — much
      milder than the classical ``∼ M/r³`` curvature growth, but the
      singularity is softened, not removed. Bonanno & Reuter's regular
      de Sitter core (``1 - f ∝ r²``, finite curvature) requires the
      softened proper-distance cutoff ``k ∝ 1/d(r)`` with
      ``d(r) ∝ r^{3/2}`` near the origin [hep-th/0002196], which none
      of the shipped ``ScaleIdentification`` classes implement.

References:
    Bonanno & Reuter (2000), Phys. Rev. D 62, 043008 [hep-th/0002196]
    Reuter & Saueressig (2012), New J. Phys. 14, 055022 [1202.2274]
    Platania (2019), Eur. Phys. J. C 79, 470
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from asymsafety.analysis.flow import RGTrajectory
from asymsafety.cosmology.scale_identification import (
    InverseDistanceScale,
    ScaleIdentification,
)


@dataclass
class RGImprovedSchwarzschild:
    """RG-improved Schwarzschild geometry for an ADM mass ``M``.

    Args:
        trajectory: An RGTrajectory containing 'g' (and optionally
            'lambda') coupling values vs. RG time t = log(k/k0).
        scale: Identification ``k(r)`` (default: ``k = 1/r``).
        M: ADM mass in units where ``G_N = 1``.
        k0: Reference scale at which the dimensionless ``g`` is normalized
            (usually the Planck scale; defaults to 1).
        include_lambda: If True, also include a position-dependent Λ(r)
            in the lapse: ``f = 1 - 2GM/r - Λ r²/3``.
    """

    trajectory: RGTrajectory
    scale: ScaleIdentification = None  # type: ignore[assignment]
    M: float = 1.0
    k0: float = 1.0
    include_lambda: bool = False

    def __post_init__(self):
        if self.scale is None:
            self.scale = InverseDistanceScale(xi=1.0)
        if "g" not in self.trajectory.coupling_names:
            raise ValueError(
                "Trajectory must contain a 'g' coupling for the RG-improved "
                "Newton constant."
            )

    def _t_of_r(self, r: np.ndarray) -> np.ndarray:
        """RG time ``t = log(k(r) / k0)``."""
        k = self.scale.k_of_r(r)
        return np.log(np.maximum(k, 1e-300) / self.k0)

    def G(self, r: float | np.ndarray) -> np.ndarray:
        """Running Newton constant ``G(r) = g(k(r)) / k(r)²`` (dim units)."""
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        k = self.scale.k_of_r(r_arr)
        t = np.log(np.maximum(k, 1e-300) / self.k0)
        g_vals = np.array([
            self.trajectory.at_scale(float(ti))["g"] for ti in t
        ])
        result = g_vals / k**2
        return float(result[0]) if result.size == 1 else result

    def Lambda(self, r: float | np.ndarray) -> np.ndarray:
        """Running cosmological constant ``Λ(r) = λ(k(r)) · k(r)²``."""
        if "lambda" not in self.trajectory.coupling_names:
            return np.zeros_like(np.atleast_1d(np.asarray(r, dtype=float)))
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        k = self.scale.k_of_r(r_arr)
        t = np.log(np.maximum(k, 1e-300) / self.k0)
        lam_vals = np.array([
            self.trajectory.at_scale(float(ti))["lambda"] for ti in t
        ])
        result = lam_vals * k**2
        return float(result[0]) if result.size == 1 else result

    def lapse(self, r: float | np.ndarray) -> np.ndarray:
        """Improved lapse ``f(r) = 1 - 2G(r)M/r [- Λ(r) r²/3]``."""
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        G_r = np.atleast_1d(self.G(r_arr))
        f = 1.0 - 2.0 * G_r * self.M / r_arr
        if self.include_lambda:
            Lam_r = np.atleast_1d(self.Lambda(r_arr))
            f -= Lam_r * r_arr**2 / 3.0
        return float(f[0]) if f.size == 1 else f

    def horizons(
        self,
        r_min: float = 1e-3,
        r_max: float = 100.0,
        n_samples: int = 10000,
    ) -> list[float]:
        """Locate roots of ``f(r) = 0`` in ``[r_min, r_max]``.

        Returns a list of horizon radii. Empty when no horizon exists
        (sub-critical mass).
        """
        rs = np.geomspace(r_min, r_max, n_samples)
        fs = self.lapse(rs)
        # Detect sign changes
        sign_changes = np.where(np.diff(np.sign(fs)) != 0)[0]
        roots: list[float] = []
        for idx in sign_changes:
            try:
                root = brentq(self.lapse, rs[idx], rs[idx + 1])
                roots.append(float(root))
            except (ValueError, RuntimeError):
                continue
        return roots

    def critical_mass(
        self,
        r_search: tuple[float, float] = (1e-3, 100.0),
        n_samples: int = 200,
        M_search: tuple[float, float] = (1e-3, 10.0),
    ) -> float | None:
        """Find ``M_crit`` below which no horizon exists.

        Bisection on M, returning the largest ``M`` for which
        ``horizons()`` returns the empty list, or the smallest mass at
        which exactly one (extremal) horizon appears.
        """
        original_M = self.M

        def has_horizon(M: float) -> bool:
            self.M = M
            return len(self.horizons(r_search[0], r_search[1], n_samples)) > 0

        try:
            M_lo, M_hi = M_search
            if not has_horizon(M_hi):
                # Even the upper search bound has no horizon — increase range
                return None
            if has_horizon(M_lo):
                # Even the lower bound has a horizon — return lo as upper estimate
                return float(M_lo)
            for _ in range(50):
                mid = 0.5 * (M_lo + M_hi)
                if has_horizon(mid):
                    M_hi = mid
                else:
                    M_lo = mid
                if (M_hi - M_lo) < 1e-6:
                    break
            return 0.5 * (M_lo + M_hi)
        finally:
            self.M = original_M

    def metric_components(
        self, r: float | np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return ``g_tt``, ``g_rr`` for the RG-improved metric."""
        f = np.atleast_1d(self.lapse(r))
        return {
            "g_tt": -f,
            "g_rr": 1.0 / f,
        }
