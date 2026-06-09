"""Tests for the in-memory and disk-backed caching utilities."""

from __future__ import annotations

import os
from pathlib import Path

import sympy
from sympy import Symbol

from asymsafety.utils.caching import (
    cache_dir,
    disk_cache,
    joblib_available,
    symbolic_cache,
)


class TestSymbolicCache:
    """In-memory symbolic_cache: hits, misses, sympy keys."""

    def test_cache_hit_avoids_recomputation(self):
        calls = []

        @symbolic_cache
        def f(x):
            calls.append(x)
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls == [3]

    def test_distinct_sympy_args_distinct_keys(self):
        calls = []

        @symbolic_cache
        def f(expr):
            calls.append(expr)
            return expr * 2

        x, y = Symbol("x"), Symbol("y")
        f(x + 1)
        f(y + 1)
        assert len(calls) == 2  # different exprs, two calls

    def test_equal_sympy_args_same_key(self):
        calls = []

        @symbolic_cache
        def f(expr):
            calls.append(expr)
            return expr * 2

        x = Symbol("x")
        # Two structurally identical expressions, built independently
        f(x + Symbol("x") - Symbol("x"))
        f(x + Symbol("x") - Symbol("x"))
        assert len(calls) == 1

    def test_cache_clear(self):
        calls = []

        @symbolic_cache
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(1)
        f.cache_clear()
        f(1)
        assert calls == [1, 1]


class TestCacheDir:
    """The disk cache resolves to a sensible directory."""

    def test_cache_dir_exists(self):
        d = cache_dir()
        assert d.exists()
        assert d.is_dir()

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASYMSAFETY_CACHE_DIR", str(tmp_path / "custom"))
        d = cache_dir()
        assert d == tmp_path / "custom"
        assert d.exists()


class TestDiskCacheFallback:
    """When joblib is unavailable, disk_cache transparently falls back."""

    def test_decorator_works_without_joblib(self, tmp_path, monkeypatch):
        # Whether or not joblib is present, the decorator must be a no-op
        # in terms of API surface. The cache dir must be fresh: a persistent
        # cache from a previous test run would satisfy f(5) without calling f.
        monkeypatch.setenv("ASYMSAFETY_CACHE_DIR", str(tmp_path / "cache"))
        calls = []

        @disk_cache()
        def f(x):
            calls.append(x)
            return x + 1

        assert f(5) == 6
        assert f(5) == 6
        # Either disk or memory: both must hit on the second call
        assert calls == [5]

    def test_joblib_available_is_bool(self):
        assert isinstance(joblib_available(), bool)


class TestJacobianDiskCache:
    """The free function _compute_jacobian_symbolic uses the disk cache."""

    def test_identical_inputs_identical_output(self):
        from asymsafety.beta.system import _compute_jacobian_symbolic

        x, y = Symbol("x"), Symbol("y")
        exprs = (x**2 + y, x * y)
        syms = (x, y)
        m1 = _compute_jacobian_symbolic(exprs, syms)
        m2 = _compute_jacobian_symbolic(exprs, syms)
        assert m1 == m2

    def test_different_inputs_different_output(self):
        from asymsafety.beta.system import _compute_jacobian_symbolic

        x, y = Symbol("x"), Symbol("y")
        m1 = _compute_jacobian_symbolic((x**2,), (x,))
        m2 = _compute_jacobian_symbolic((y**3,), (y,))
        assert m1 != m2
