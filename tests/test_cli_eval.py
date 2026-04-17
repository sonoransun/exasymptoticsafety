"""End-to-end tests for ``asymsafety eval``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asymsafety.cli.io import read_npz
from asymsafety.cli.main import main


class TestEvalSubcommand:
    def test_eh_grid_npz(self, tmp_path):
        out = tmp_path / "eh.npz"
        rc = main([
            "eval",
            "--truncation", "eh",
            "--grid", "g:0:1:5,lambda:-0.4:0.4:4",
            "--output", str(out),
        ])
        assert rc == 0
        assert out.exists()
        arrays, meta = read_npz(out)
        # 5 × 4 = 20 grid points, 2 couplings
        assert arrays["grid"].shape == (20, 2)
        assert arrays["betas"].shape == (20, 2)
        assert meta["truncation"] == "eh"
        assert meta["n_points"] == 20

    def test_eh_grid_json(self, tmp_path):
        from asymsafety.cli.io import read_json
        out = tmp_path / "eh.json"
        rc = main([
            "eval",
            "--truncation", "eh",
            "--grid", "g:0:1:3",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, meta = read_json(out)
        # Only g grid → 3 points; lambda axis defaults to a single point at 0
        assert arrays["grid"].shape == (3, 2)
        assert arrays["betas"].shape == (3, 2)

    def test_quadratic_axis_arrays_present(self, tmp_path):
        out = tmp_path / "q.npz"
        rc = main([
            "eval",
            "--truncation", "quadratic",
            "--grid", "g:0:1:4",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, _ = read_npz(out)
        # axis_<name> arrays are emitted for each coupling
        assert "axis_g" in arrays
        assert "axis_lambda" in arrays
        assert "axis_alpha" in arrays
        assert "axis_beta" in arrays
        assert arrays["axis_g"].shape == (4,)

    def test_unknown_coupling_in_grid_raises(self, tmp_path):
        out = tmp_path / "x.npz"
        with pytest.raises(ValueError, match="unknown coupling"):
            main([
                "eval",
                "--truncation", "eh",
                "--grid", "nonsense:0:1:3",
                "--output", str(out),
            ])

    def test_malformed_grid_token_raises(self, tmp_path):
        out = tmp_path / "x.npz"
        with pytest.raises(ValueError, match="expected NAME:LO:HI:N"):
            main([
                "eval",
                "--truncation", "eh",
                "--grid", "g:0:1",  # missing N
                "--output", str(out),
            ])

    def test_param_passed_through(self, tmp_path):
        out = tmp_path / "ehm.npz"
        rc = main([
            "eval",
            "--truncation", "eh_matter",
            "--param", "n_scalars=3",
            "--grid", "g:0:0.5:3",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, meta = read_npz(out)
        assert meta["params"]["n_scalars"] == "3"
        assert arrays["betas"].shape == (3, 2)


class TestEvalBackends:
    def test_backend_numpy_explicit(self, tmp_path):
        out = tmp_path / "b.npz"
        rc = main([
            "eval", "--truncation", "eh",
            "--grid", "g:0:1:3",
            "--backend", "numpy",
            "--output", str(out),
        ])
        assert rc == 0

    def test_backend_auto(self, tmp_path):
        out = tmp_path / "b.npz"
        rc = main([
            "eval", "--truncation", "eh",
            "--grid", "g:0:1:3",
            "--backend", "auto",
            "--output", str(out),
        ])
        assert rc == 0
