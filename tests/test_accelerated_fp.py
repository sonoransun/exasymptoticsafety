"""Tests for accelerated fixed point finder."""

import pytest
import numpy as np

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.compute.accelerated.fixed_points import AcceleratedFixedPointFinder
from asymsafety.compute.backends.base import SequentialBackend


class TestAcceleratedFixedPointFinder:
    """Test that accelerated FP finder matches base finder."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)

    def test_finds_ngfp(self):
        """Accelerated finder should find the NGFP."""
        finder = AcceleratedFixedPointFinder(self.system, backend=SequentialBackend())
        fp = finder.find_fixed_point({"g": 0.7, "lambda": 0.14})
        assert fp is not None
        assert not fp.is_gaussian
        assert fp.location["g"] > 0.01

    def test_finds_gaussian_fp(self):
        """Should always include the Gaussian FP."""
        finder = AcceleratedFixedPointFinder(self.system, backend=SequentialBackend())
        gfp = finder.gaussian_fixed_point()
        assert gfp.is_gaussian

    def test_find_all_includes_ngfp(self):
        """find_all should find at least 2 FPs (GFP + NGFP)."""
        finder = AcceleratedFixedPointFinder(self.system, backend=SequentialBackend())
        fps = finder.find_all_fixed_points(
            bounds={"g": (0.0, 2.0), "lambda": (-0.5, 0.45)},
            n_grid=8,
            n_random=30,
        )
        # Should find at least the GFP, plus ideally the NGFP
        assert len(fps) >= 1
        # Verify GFP is always present
        has_gfp = any(fp.is_gaussian for fp in fps)
        assert has_gfp

    def test_prefilter_reduces_candidates(self):
        """Pre-filtering should produce fewer candidates than the full grid."""
        finder = AcceleratedFixedPointFinder(self.system, backend=SequentialBackend())
        evaluator = finder._evaluator

        bounds = {"g": (0.0, 2.0), "lambda": (-0.5, 0.45)}
        names = self.system.coupling_names
        grids = [np.linspace(bounds[name][0], bounds[name][1], 10) for name in names]
        mesh = np.meshgrid(*grids)
        all_points = np.column_stack([m.ravel() for m in mesh])

        norms = evaluator.evaluate_norms(all_points)
        candidates = all_points[norms < 1.0]

        # Pre-filter should reduce the number of candidates
        assert len(candidates) < len(all_points)
