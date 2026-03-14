"""Dask distributed compute backend."""

from __future__ import annotations

from typing import Callable

import numpy as np


class DaskBackend:
    """Distributed compute backend using Dask.

    Uses ``dask.delayed`` for lazy task graphs and supports multiple
    schedulers (``"processes"``, ``"threads"``, ``"synchronous"``).
    """

    def __init__(
        self,
        scheduler: str = "processes",
        n_workers: int | None = None,
    ) -> None:
        try:
            import dask  # noqa: F401, WPS433
        except ImportError as exc:
            raise ImportError(
                "Dask required. Install with: pip install asymsafety[dask]"
            ) from exc

        self._scheduler = scheduler
        self._n_workers = n_workers

    @property
    def name(self) -> str:
        return f"dask ({self._scheduler})"

    @property
    def capabilities(self) -> set[str]:
        return {"cpu", "distributed", "multicore"}

    def map(self, fn: Callable, items: list) -> list:
        """Build a Dask task graph and compute all items."""
        import dask  # noqa: WPS433

        delayed_results = [dask.delayed(fn)(item) for item in items]
        return list(
            dask.compute(
                *delayed_results,
                scheduler=self._scheduler,
                num_workers=self._n_workers,
            )
        )

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Chunk *points* and evaluate via Dask delayed tasks."""
        import dask  # noqa: WPS433

        chunks = [
            points[i : i + chunk_size]
            for i in range(0, len(points), chunk_size)
        ]
        delayed_results = [dask.delayed(fn)(chunk) for chunk in chunks]
        results = list(
            dask.compute(
                *delayed_results,
                scheduler=self._scheduler,
                num_workers=self._n_workers,
            )
        )
        return np.concatenate(results, axis=0)
