"""Backend detection and configuration for compute acceleration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asymsafety.compute.backends.base import ComputeBackend


@dataclass
class ComputeConfig:
    """Configuration for compute acceleration."""

    backend: str = "local"  # "local", "jax", "ray", "dask", "boinc"
    n_workers: int | None = None  # None = auto-detect
    gpu: bool = False
    ray_address: str | None = None  # None = local, "auto" = existing cluster


def available_backends() -> list[str]:
    """Return list of available backends based on installed packages.

    Always includes ``"local"``.  Other backends are listed only when
    their required third-party package can be imported.
    """
    backends = ["local", "sequential"]

    if _can_import("jax"):
        backends.append("jax")
    if _can_import("ray"):
        backends.append("ray")
    if _can_import("dask"):
        backends.append("dask")

    # BOINC backend only requires urllib (stdlib), always available
    backends.append("boinc")

    return backends


def get_backend(name: str = "auto", **kwargs) -> "ComputeBackend":
    """Get a compute backend by name.

    Args:
        name: Backend identifier.  ``"auto"`` selects the best available
              (JAX if a GPU is detected, otherwise local with multiprocessing).
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A :class:`ComputeBackend` instance.

    Raises:
        ValueError: If *name* is not recognised.
        ImportError: If the requested backend's dependencies are missing.
    """
    if name == "auto":
        name = _auto_select()

    if name == "local":
        from asymsafety.compute.backends.base import LocalBackend

        return LocalBackend(**kwargs)

    if name == "sequential":
        from asymsafety.compute.backends.base import SequentialBackend

        return SequentialBackend()

    if name == "jax":
        from asymsafety.compute.backends.jax_backend import JaxBackend

        return JaxBackend(**kwargs)

    if name == "ray":
        from asymsafety.compute.backends.ray_backend import RayBackend

        return RayBackend(**kwargs)

    if name == "dask":
        from asymsafety.compute.backends.dask_backend import DaskBackend

        return DaskBackend(**kwargs)

    if name == "boinc":
        from asymsafety.compute.backends.boinc_backend import BOINCBackend

        return BOINCBackend(**kwargs)

    raise ValueError(
        f"Unknown backend: {name!r}.  Available: {available_backends()}"
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    """Return *True* if *module_name* can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _auto_select() -> str:
    """Heuristic: pick the best backend that is currently available."""
    if _can_import("jax"):
        try:
            import jax  # noqa: WPS433

            devices = jax.devices()
            if any(d.platform != "cpu" for d in devices):
                return "jax"
        except Exception:  # noqa: BLE001
            pass
    return "local"
