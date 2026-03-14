"""BOINC/crowd computing backend via REST task server."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable

import numpy as np


class BOINCBackend:
    """Crowd computing backend that distributes work via HTTP task server.

    Work units are submitted to a task server, which distributes them
    to volunteer computing clients (BOINC wrappers or direct HTTP clients).
    Results are polled until all units complete.

    Requires a running task server
    (see ``asymsafety.compute.distributed.task_server``).
    """

    def __init__(self, server_url: str = "http://localhost:8765") -> None:
        self._server_url = server_url.rstrip("/")

    @property
    def name(self) -> str:
        return f"boinc ({self._server_url})"

    @property
    def capabilities(self) -> set[str]:
        return {"distributed", "crowd"}

    def map(self, fn: Callable, items: list) -> list:
        """Submit each item as a work unit and poll for results."""
        # Submit work units
        work_ids: list[str] = []
        for item in items:
            payload = json.dumps(
                {"task": "map", "data": _serialize_item(item)}
            ).encode()
            req = urllib.request.Request(
                f"{self._server_url}/submit",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req)  # noqa: S310
            result = json.loads(resp.read())
            work_ids.append(result["id"])

        # Poll for results
        results: list[Any] = [None] * len(work_ids)
        remaining = set(range(len(work_ids)))
        while remaining:
            time.sleep(1.0)
            for i in list(remaining):
                req = urllib.request.Request(
                    f"{self._server_url}/result/{work_ids[i]}"
                )
                resp = urllib.request.urlopen(req)  # noqa: S310
                data = json.loads(resp.read())
                if data["status"] == "complete":
                    results[i] = _deserialize_item(data["result"])
                    remaining.discard(i)

        return results

    def map_batch(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Chunk *points* and submit as individual work units."""
        chunks = [
            points[i : i + chunk_size]
            for i in range(0, len(points), chunk_size)
        ]
        results = self.map(fn, chunks)
        return np.concatenate(results, axis=0)


# ------------------------------------------------------------------
# Serialisation helpers for JSON transport
# ------------------------------------------------------------------


def _serialize_item(item: Any) -> dict:
    """Serialize an item for JSON transport."""
    if isinstance(item, np.ndarray):
        return {
            "type": "ndarray",
            "data": item.tolist(),
            "dtype": str(item.dtype),
        }
    if isinstance(item, dict):
        return {
            "type": "dict",
            "data": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in item.items()
            },
        }
    return {"type": "raw", "data": item}


def _deserialize_item(data: Any) -> Any:
    """Deserialize an item from JSON transport."""
    if isinstance(data, dict) and "type" in data:
        if data["type"] == "ndarray":
            return np.array(data["data"], dtype=data.get("dtype", "float64"))
        if data["type"] == "dict":
            return data["data"]
        return data["data"]
    return data
