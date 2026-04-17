"""Tests for the runtime config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymsafety.config import (
    CacheConfig,
    ComputeRuntimeConfig,
    Config,
    PlotConfig,
    SolverConfig,
    load_config,
)


class TestDefaults:
    def test_defaults_load_without_files(self, tmp_path):
        cfg = load_config(
            user_path=tmp_path / "missing-user.toml",
            project_path=tmp_path / "missing-project.toml",
            env={},
        )
        assert isinstance(cfg, Config)
        assert isinstance(cfg.solver, SolverConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.compute, ComputeRuntimeConfig)
        assert isinstance(cfg.plot, PlotConfig)
        # default values
        assert cfg.solver.fixed_point_tol == 1.0e-10
        assert cfg.compute.batch_backend == "auto"
        assert cfg.cache.disk_enabled is True
        assert cfg.compute.parallel_default is False

    def test_to_dict_round_trip(self, tmp_path):
        cfg = load_config(
            user_path=tmp_path / "x.toml",
            project_path=tmp_path / "y.toml",
            env={},
        )
        d = cfg.to_dict()
        assert "solver" in d
        assert "compute" in d
        assert d["solver"]["fixed_point_tol"] == 1.0e-10


class TestFileOverride:
    def test_user_file_overrides_defaults(self, tmp_path):
        user = tmp_path / "user.toml"
        user.write_text(
            "[solver]\nfixed_point_tol = 1.0e-6\n"
            "[compute]\nbatch_backend = 'numpy'\n"
        )
        cfg = load_config(
            user_path=user,
            project_path=tmp_path / "missing.toml",
            env={},
        )
        assert cfg.solver.fixed_point_tol == 1.0e-6
        assert cfg.compute.batch_backend == "numpy"
        # Untouched fields keep defaults
        assert cfg.solver.flow_max_step == 0.1

    def test_project_file_overrides_user_file(self, tmp_path):
        user = tmp_path / "user.toml"
        project = tmp_path / "project.toml"
        user.write_text("[solver]\nfixed_point_tol = 1.0e-6\n")
        project.write_text("[solver]\nfixed_point_tol = 1.0e-3\n")
        cfg = load_config(user_path=user, project_path=project, env={})
        assert cfg.solver.fixed_point_tol == 1.0e-3

    def test_invalid_toml_silently_ignored(self, tmp_path):
        user = tmp_path / "broken.toml"
        user.write_text("this is not = valid toml [[[")
        # Should fall back to defaults rather than raise
        cfg = load_config(user_path=user, env={})
        assert cfg.solver.fixed_point_tol == 1.0e-10

    def test_unknown_keys_ignored(self, tmp_path):
        user = tmp_path / "user.toml"
        user.write_text(
            "[solver]\nfixed_point_tol = 1.0e-7\nbogus_key = 42\n"
        )
        cfg = load_config(user_path=user, env={})
        assert cfg.solver.fixed_point_tol == 1.0e-7
        assert not hasattr(cfg.solver, "bogus_key")


class TestEnvOverride:
    def test_env_overrides_file(self, tmp_path):
        user = tmp_path / "user.toml"
        user.write_text("[solver]\nfixed_point_tol = 1.0e-6\n")
        cfg = load_config(
            user_path=user,
            project_path=tmp_path / "missing.toml",
            env={"ASYMSAFETY_SOLVER_FIXED_POINT_TOL": "1e-3"},
        )
        assert cfg.solver.fixed_point_tol == 1.0e-3

    def test_env_bool_coercion(self, tmp_path):
        cases = {
            "true": True, "True": True, "1": True, "yes": True, "on": True,
            "false": False, "False": False, "0": False, "no": False, "off": False,
        }
        for raw, expected in cases.items():
            cfg = load_config(
                user_path=tmp_path / "x.toml",
                project_path=tmp_path / "y.toml",
                env={"ASYMSAFETY_COMPUTE_PARALLEL_DEFAULT": raw},
            )
            assert cfg.compute.parallel_default is expected, raw

    def test_env_int_coercion(self, tmp_path):
        cfg = load_config(
            user_path=tmp_path / "x.toml",
            env={"ASYMSAFETY_COMPUTE_MAX_WORKERS": "8"},
        )
        assert cfg.compute.max_workers == 8

    def test_env_unknown_section_ignored(self, tmp_path):
        # Should not raise even when prefix matches but section/key don't exist
        cfg = load_config(
            user_path=tmp_path / "x.toml",
            env={"ASYMSAFETY_FOO_BAR": "baz"},
        )
        assert cfg.solver.fixed_point_tol == 1.0e-10


class TestCacheDirSideEffect:
    def test_cache_dir_propagates_to_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ASYMSAFETY_CACHE_DIR", raising=False)
        user = tmp_path / "user.toml"
        user.write_text(f"[cache]\ndir = '{tmp_path / 'mycache'}'\n")
        cfg = load_config(user_path=user, env={})
        # The loader should have set ASYMSAFETY_CACHE_DIR for utils.caching
        import os
        assert os.environ["ASYMSAFETY_CACHE_DIR"] == str(tmp_path / "mycache")

    def test_existing_env_cache_dir_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASYMSAFETY_CACHE_DIR", "/already/set")
        user = tmp_path / "user.toml"
        user.write_text(f"[cache]\ndir = '{tmp_path / 'ignored'}'\n")
        load_config(user_path=user, env={"ASYMSAFETY_CACHE_DIR": "/already/set"})
        import os
        assert os.environ["ASYMSAFETY_CACHE_DIR"] == "/already/set"
