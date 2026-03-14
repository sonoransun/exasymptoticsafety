"""Tests for batch beta function evaluation."""

import pytest
import numpy as np

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.compute.batch.evaluator import NumpyBatchEvaluator


class TestNumpyBatchEvaluator:
    """Test batch evaluation matches sequential evaluation."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)
        self.evaluator = NumpyBatchEvaluator(self.system)

    def test_single_point_matches_rhs(self):
        """Batch eval of 1 point should match rhs_vector()."""
        rhs = self.system.rhs_vector()
        point = np.array([[0.7, 0.14]])
        batch_result = self.evaluator.evaluate_batch(point)
        seq_result = rhs(0, [0.7, 0.14])

        np.testing.assert_allclose(batch_result[0], seq_result, rtol=1e-10)

    def test_multiple_points(self):
        """Batch eval of N points should match N sequential evals."""
        rhs = self.system.rhs_vector()
        points = np.array([
            [0.3, 0.1],
            [0.7, 0.14],
            [1.0, 0.2],
            [0.5, 0.0],
        ])
        batch_result = self.evaluator.evaluate_batch(points)

        for i in range(len(points)):
            seq_result = rhs(0, list(points[i]))
            np.testing.assert_allclose(batch_result[i], seq_result, rtol=1e-10)

    def test_shape(self):
        """Output shape should be (N, n_couplings)."""
        points = np.random.rand(50, 2) * 0.5
        result = self.evaluator.evaluate_batch(points)
        assert result.shape == (50, 2)

    def test_norms(self):
        """evaluate_norms should return (N,) array of |β(x)|."""
        points = np.array([[0.0, 0.0], [0.7, 0.14]])
        norms = self.evaluator.evaluate_norms(points)
        assert norms.shape == (2,)
        # At GFP (0,0), norm should be 0
        assert abs(norms[0]) < 1e-10

    def test_handles_singularity(self):
        """Should handle λ = 0.5 singularity without crashing."""
        points = np.array([[0.5, 0.5], [0.5, 0.499]])
        result = self.evaluator.evaluate_batch(points)
        # Should not crash; result may contain NaN at singularity
        assert result.shape == (2, 2)

    def test_batch_evaluator_via_system(self):
        """BetaFunctionSystem.batch_evaluator() should work."""
        evaluator = self.system.batch_evaluator("numpy")
        assert isinstance(evaluator, NumpyBatchEvaluator)
        points = np.array([[0.3, 0.1]])
        result = evaluator.evaluate_batch(points)
        assert result.shape == (1, 2)
