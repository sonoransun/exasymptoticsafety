"""Caching utilities for expensive symbolic computations."""

import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable


def symbolic_cache(func: Callable) -> Callable:
    """In-memory memoization decorator for symbolic computations.

    Caches results based on all arguments (converted to strings for hashing).
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
    """Create a hashable key from function arguments."""
    parts = [str(a) for a in args]
    parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()
