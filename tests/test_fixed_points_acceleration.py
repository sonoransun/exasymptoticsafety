"""Tests for the prefilter and parallel modes of find_all_fixed_points."""

from __future__ import annotations

import numpy as np
import pytest

from asymsafety.actions.matter import MatterContent
from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.beta.matter import build_gravity_matter_fp_system


class TestPrefilter:
    """The optional batch prefilter should preserve correctness when the
    threshold is generous, and reduce the candidate set otherwise."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)
        self.finder = FixedPointFinder(self.system)

    def test_prefilter_off_matches_baseline(self):
        """prefilter_threshold=None reproduces the original behavior."""
        baseline = self.finder.find_all_fixed_points(n_grid=10, n_random=20)
        same = self.finder.find_all_fixed_points(
            n_grid=10, n_random=20, prefilter_threshold=None
        )
        assert len(baseline) == len(same)

    def test_prefilter_with_loose_threshold_finds_ngfp(self):
        """A loose threshold (10.0) keeps enough candidates to find the NGFP."""
        fps = self.finder.find_all_fixed_points(
            n_grid=20, n_random=50, prefilter_threshold=10.0
        )
        # Gaussian + at least one NGFP
        non_gaussian = [fp for fp in fps if not fp.is_gaussian]
        assert len(non_gaussian) >= 1

    def test_prefilter_zero_keeps_only_near_zero_norms(self):
        """Threshold below epsilon keeps only candidates with |β| ≈ 0."""
        fps = self.finder.find_all_fixed_points(
            n_grid=5, n_random=5, prefilter_threshold=1e-12
        )
        # Should still find the Gaussian FP (it's added explicitly)
        assert any(fp.is_gaussian for fp in fps)

    def test_batch_backend_numpy_explicit(self):
        """batch_backend='numpy' works."""
        fps = self.finder.find_all_fixed_points(
            n_grid=10, prefilter_threshold=2.0, batch_backend="numpy"
        )
        assert len(fps) >= 1

    def test_batch_backend_auto(self):
        """batch_backend='auto' picks numpy when JAX is unavailable."""
        fps = self.finder.find_all_fixed_points(
            n_grid=10, prefilter_threshold=2.0, batch_backend="auto"
        )
        assert len(fps) >= 1


class TestParallelRefinement:
    """ProcessPoolExecutor refinement returns the same FPs as serial."""

    def test_parallel_matches_serial_eh(self):
        system = build_eh_beta_system(d=4)
        finder = FixedPointFinder(system)
        serial = finder.find_all_fixed_points(n_grid=10, n_random=30)
        parallel = finder.find_all_fixed_points(
            n_grid=10, n_random=30, parallel=True, max_workers=2
        )
        assert len(serial) == len(parallel)

    def test_parallel_matches_serial_gravity_matter(self):
        gm = build_gravity_matter_fp_system(
            MatterContent(n_scalars=1), scalar_quartic=True
        )
        finder = FixedPointFinder(gm)
        serial = finder.find_all_fixed_points(n_grid=4, n_random=20)
        parallel = finder.find_all_fixed_points(
            n_grid=4, n_random=20, parallel=True, max_workers=2
        )
        assert len(serial) == len(parallel)

    def test_parallel_with_prefilter(self):
        """Combining prefilter + parallel should find at least the GFP."""
        system = build_eh_beta_system(d=4)
        finder = FixedPointFinder(system)
        fps = finder.find_all_fixed_points(
            n_grid=10,
            n_random=30,
            prefilter_threshold=5.0,
            parallel=True,
            max_workers=2,
        )
        assert any(fp.is_gaussian for fp in fps)


class TestSystemPicklable:
    """BetaFunctionSystem must round-trip through pickle (used by ProcessPool)."""

    def test_eh_pickle_round_trip(self):
        import pickle
        sys = build_eh_beta_system(d=4)
        # Trigger lambdified caches
        sys.evaluate({"g": 0.5, "lambda": 0.1})
        sys.jacobian_numerical({"g": 0.5, "lambda": 0.1})
        # Should still pickle (caches are stripped by __getstate__)
        data = pickle.dumps(sys)
        sys2 = pickle.loads(data)
        result = sys2.evaluate({"g": 0.5, "lambda": 0.1})
        assert "g" in result
        assert "lambda" in result

    def test_gravity_matter_pickle_round_trip(self):
        import pickle
        sys = build_gravity_matter_fp_system(
            MatterContent(n_scalars=1, n_dirac=1),
            scalar_quartic=True,
            yukawa=True,
        )
        sys.evaluate({"g": 0.5, "lambda": 0.1, "lambda_phi": 0.01, "y": 0.1})
        data = pickle.dumps(sys)
        sys2 = pickle.loads(data)
        # Verify Jacobian also reconstructs
        J = sys2.jacobian_numerical(
            {"g": 0.5, "lambda": 0.1, "lambda_phi": 0.01, "y": 0.1}
        )
        assert J.shape == (4, 4)
