"""Tests for compute backends."""

import pytest
import numpy as np

from asymsafety.compute.backends.base import LocalBackend, SequentialBackend
from asymsafety.compute.config import available_backends, get_backend


class TestLocalBackend:
    """Test the local CPU backend."""

    def test_map_simple(self):
        """map() should apply function to all items."""
        backend = LocalBackend(n_workers=2)
        result = backend.map(lambda x: x * 2, [1, 2, 3, 4])
        assert result == [2, 4, 6, 8]

    def test_map_single_item(self):
        """map() with one item should not use multiprocessing."""
        backend = LocalBackend()
        result = backend.map(lambda x: x + 1, [5])
        assert result == [6]

    def test_map_batch(self):
        """map_batch() should process array."""
        backend = LocalBackend()
        points = np.array([[1, 2], [3, 4], [5, 6]])
        result = backend.map_batch(lambda p: p * 2, points)
        np.testing.assert_array_equal(result, points * 2)

    def test_name(self):
        assert LocalBackend().name == "local"

    def test_capabilities(self):
        caps = LocalBackend().capabilities
        assert "cpu" in caps
        assert "multicore" in caps


class TestSequentialBackend:
    """Test the sequential (debugging) backend."""

    def test_map(self):
        backend = SequentialBackend()
        result = backend.map(lambda x: x ** 2, [1, 2, 3])
        assert result == [1, 4, 9]

    def test_name(self):
        assert SequentialBackend().name == "sequential"


class TestConfig:
    """Test backend configuration and detection."""

    def test_available_backends_includes_local(self):
        backends = available_backends()
        assert "local" in backends
        assert "sequential" in backends

    def test_get_backend_local(self):
        backend = get_backend("local")
        assert backend.name == "local"

    def test_get_backend_sequential(self):
        backend = get_backend("sequential")
        assert backend.name == "sequential"

    def test_get_backend_auto(self):
        """Auto should return a valid backend."""
        backend = get_backend("auto")
        assert hasattr(backend, "map")
        assert hasattr(backend, "map_batch")
