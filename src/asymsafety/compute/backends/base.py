"""Core compute-backend protocol and built-in local backends."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol that every compute backend must satisfy."""

    def map(self, fn: Callable, items: list) -> list:
        """Map *fn* over *items* in parallel.

        Each element of *items* is passed as a single positional
        argument to *fn*.
        """
        ...

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Map a batch function over array chunks.

        Args:
            fn: Callable that accepts an ``(N, D)`` array and returns an
                ``(N, D)`` (or ``(N,)``) array.
            points: Full array of input points.
            chunk_size: Maximum number of rows per chunk.

        Returns:
            Concatenated results.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    def capabilities(self) -> set[str]:
        """Set of capability tags (e.g. ``"gpu"``, ``"distributed"``)."""
        ...


class LocalBackend:
    """Local CPU backend using :class:`~concurrent.futures.ProcessPoolExecutor`."""

    def __init__(self, n_workers: int | None = None):
        self._n_workers = n_workers or os.cpu_count()

    @property
    def name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> set[str]:
        return {"cpu", "multicore"}

    def map(self, fn: Callable, items: list) -> list:
        """Map *fn* over *items*, falling back to sequential for single items.

        Uses ProcessPoolExecutor when possible. Falls back to sequential
        execution if the function cannot be pickled (e.g. lambdas, closures).
        """
        if len(items) <= 1:
            return [fn(item) for item in items]
        try:
            with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                return list(pool.map(fn, items))
        except (AttributeError, TypeError, Exception) as e:
            if "pickle" in str(e).lower() or "Pickling" in str(e):
                # Fall back to sequential for unpicklable functions
                return [fn(item) for item in items]
            raise

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Evaluate *fn* on *points*, chunking if the array is large."""
        if len(points) <= chunk_size:
            return fn(points)
        chunks = [
            points[i : i + chunk_size]
            for i in range(0, len(points), chunk_size)
        ]
        results = [fn(chunk) for chunk in chunks]
        return np.concatenate(results, axis=0)


class SequentialBackend:
    """Sequential backend for debugging.  No parallelism whatsoever."""

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def capabilities(self) -> set[str]:
        return {"cpu"}

    def map(self, fn: Callable, items: list) -> list:
        """Evaluate *fn* on each item one at a time."""
        return [fn(item) for item in items]

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Evaluate *fn* on the full *points* array (no chunking)."""
        return fn(points)
