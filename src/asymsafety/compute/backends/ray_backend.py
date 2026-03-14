"""Ray distributed compute backend."""

from __future__ import annotations

from typing import Callable

import numpy as np


class RayBackend:
    """Distributed compute backend using Ray.

    Initialises a Ray runtime (local or cluster) on first use and
    distributes work via ``ray.remote``.
    """

    def __init__(
        self,
        address: str | None = None,
        num_cpus: int | None = None,
    ) -> None:
        try:
            import ray  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "Ray required. Install with: pip install asymsafety[distributed]"
            ) from exc

        if not ray.is_initialized():
            ray.init(
                address=address,
                num_cpus=num_cpus,
                ignore_reinit_error=True,
            )
        self._ray = ray

    @property
    def name(self) -> str:
        nodes = len(self._ray.nodes())
        return f"ray ({nodes} node{'s' if nodes != 1 else ''})"

    @property
    def capabilities(self) -> set[str]:
        return {"cpu", "distributed", "multicore"}

    def map(self, fn: Callable, items: list) -> list:
        """Distribute *fn* calls across Ray workers."""
        ray = self._ray
        remote_fn = ray.remote(fn)
        futures = [remote_fn.remote(item) for item in items]
        return ray.get(futures)

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Chunk *points* and evaluate in parallel on Ray workers."""
        ray = self._ray
        chunks = [
            points[i : i + chunk_size]
            for i in range(0, len(points), chunk_size)
        ]
        remote_fn = ray.remote(fn)
        futures = [remote_fn.remote(chunk) for chunk in chunks]
        results = ray.get(futures)
        return np.concatenate(results, axis=0)
