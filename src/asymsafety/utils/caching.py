"""Caching utilities for expensive symbolic computations.

Two layers:
    - ``symbolic_cache`` (in-memory): zero-config memoization for repeated
      symbolic work within a single interpreter session.
    - ``disk_cache`` (opt-in, persistent): survives interpreter restarts.
      Requires ``joblib`` (``pip install joblib``); falls back to
      in-memory caching when joblib is unavailable.

Cache directory selection (in priority order):
    1. ``ASYMSAFETY_CACHE_DIR`` environment variable
    2. Platform-default user cache dir (``~/.cache/asymsafety/`` on Linux,
       ``~/Library/Caches/asymsafety/`` on macOS)

Cache keys include ``sympy.__version__`` so version upgrades automatically
invalidate stale entries.
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import sympy

_SYMPY_VERSION = sympy.__version__
_DEFAULT_SUBDIR = "asymsafety"


# ---------------------------------------------------------------------------
# In-memory cache (back-compat: existing API)
# ---------------------------------------------------------------------------

def symbolic_cache(func: Callable) -> Callable:
    """In-memory memoization decorator for symbolic computations.

    Caches results based on all arguments. Hashing uses each argument's
    SymPy ``srepr`` when available (stable across sessions and Python
    object identity), falling back to ``str()`` otherwise.
    """
    cache: dict[str, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = _make_key(args, kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def _make_key(args: tuple, kwargs: dict) -> str:
    """Stable hashable key from function arguments.

    Uses ``sympy.srepr`` for SymPy objects, ``repr`` for the rest.
    Includes the SymPy version so upgrades invalidate stale entries.
    """
    parts: list[str] = [_SYMPY_VERSION]
    for a in args:
        parts.append(_serialize(a))
    for k in sorted(kwargs):
        parts.append(f"{k}={_serialize(kwargs[k])}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def _serialize(obj: Any) -> str:
    """Best-effort stable string for hashing. Prefers sympy.srepr."""
    if hasattr(obj, "free_symbols") or isinstance(obj, sympy.Basic):
        try:
            return sympy.srepr(obj)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_serialize(x) for x in obj) + "]"
    return repr(obj)


# ---------------------------------------------------------------------------
# Disk cache (opt-in via joblib)
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the on-disk cache directory, creating it if needed."""
    env = os.environ.get("ASYMSAFETY_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Caches" / _DEFAULT_SUBDIR
    else:
        # Linux/Unix XDG default
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        path = base / _DEFAULT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _try_import_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except ImportError:
        return None


def disk_cache(name: str | None = None) -> Callable:
    """Persistent memoization decorator backed by joblib.Memory.

    Falls back transparently to in-memory ``symbolic_cache`` when joblib
    is not installed, so the decorator is always safe to apply.

    Args:
        name: Optional subdirectory name for grouping related caches.
            Defaults to the function's ``__qualname__``.

    Cache keys are derived from arguments via the same stable serializer
    used by ``symbolic_cache``. The SymPy version prefixes every key, so
    SymPy upgrades automatically force recomputation.
    """
    joblib = _try_import_joblib()

    if joblib is None:
        # No joblib → behave exactly like in-memory symbolic_cache.
        def passthrough(func: Callable) -> Callable:
            return symbolic_cache(func)
        return passthrough

    def decorator(func: Callable) -> Callable:
        subdir = name or func.__qualname__.replace(".", "_")
        memory = joblib.Memory(
            location=str(cache_dir() / subdir),
            verbose=0,
        )

        # joblib hashes its own args; we wrap with a string-key shim so
        # SymPy expressions hash identically across runs.
        @memory.cache
        def _impl(key: str, payload: Any) -> Any:  # noqa: ANN401
            args, kwargs = payload
            return func(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(args, kwargs)
            return _impl(key, (args, kwargs))

        wrapper.cache_clear = lambda: memory.clear(warn=False)  # type: ignore[attr-defined]
        wrapper.cache_dir = memory.location  # type: ignore[attr-defined]
        return wrapper

    return decorator


def joblib_available() -> bool:
    """Whether the disk cache backend is installed."""
    return _try_import_joblib() is not None
