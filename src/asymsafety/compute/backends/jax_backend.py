"""JAX compute backend for GPU-accelerated batch operations."""

from __future__ import annotations

from typing import Callable

import numpy as np


class JaxBackend:
    """GPU compute backend using JAX.

    JAX is designed for batch (SIMD) workloads rather than task-level
    parallelism.  The :meth:`map_batch` method therefore delegates
    directly to the caller's function (which is expected to use
    ``jax.vmap`` / ``jax.jit`` internally), while :meth:`map` falls
    back to sequential execution.
    """

    def __init__(self) -> None:
        try:
            import jax  # noqa: WPS433

            self._jax = jax
            self._devices = jax.devices()
        except ImportError as exc:
            raise ImportError(
                "JAX required. Install with: pip install asymsafety[gpu]"
            ) from exc

    @property
    def name(self) -> str:
        device = self._devices[0].platform if self._devices else "cpu"
        return f"jax ({device})"

    @property
    def capabilities(self) -> set[str]:
        caps: set[str] = {"cpu", "batch"}
        if any(d.platform != "cpu" for d in self._devices):
            caps.add("gpu")
        return caps

    def map(self, fn: Callable, items: list) -> list:
        """Sequential map -- JAX is for batch ops, not task-parallel."""
        return [fn(item) for item in items]

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        """Delegate to *fn* directly; JAX handles large batches natively."""
        return fn(points)
