"""End-to-end tests for ``asymsafety scan``."""

from __future__ import annotations

import numpy as np
import pytest

from asymsafety.cli.io import read_json, read_npz
from asymsafety.cli.main import main


class TestScanSubcommand:
    def test_gravity_matter_scalar_scan_json(self, tmp_path):
        out = tmp_path / "gm.json"
        rc = main([
            "scan",
            "--truncation", "gravity_matter",
            "--param", "scalar_quartic=true",
            "--param-range", "n_scalars=1:3:3",
            "--guess", "g=0.65,lambda=0.14,lambda_phi=0.01",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, meta = read_json(out)
        assert arrays["n_scalars"].shape == (3,)
        assert "g" in arrays
        assert "lambda" in arrays
        assert "lambda_phi" in arrays
        assert "fp_exists" in arrays
        assert meta["param_swept"] == "n_scalars"
        # At least the first FP should be found
        assert bool(arrays["fp_exists"][0]) is True

    def test_eh_dimension_scan_npz(self, tmp_path):
        out = tmp_path / "eh_d.npz"
        # Note: 'd' is an integer parameter; sweep d=4 only (single-point sanity)
        rc = main([
            "scan",
            "--truncation", "eh",
            "--param-range", "d=4:4:1",
            "--guess", "g=0.7,lambda=0.14",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, meta = read_npz(out)
        assert arrays["d"].shape == (1,)
        assert meta["truncation"] == "eh"

    def test_param_range_malformed_raises(self, tmp_path):
        out = tmp_path / "x.npz"
        with pytest.raises(ValueError, match="LO:HI:N"):
            main([
                "scan",
                "--truncation", "eh",
                "--param-range", "n_scalars=1:5",  # missing third value
                "--output", str(out),
            ])


class TestScanArrays:
    def test_theta_arrays_present(self, tmp_path):
        out = tmp_path / "gm.json"
        rc = main([
            "scan",
            "--truncation", "gravity_matter",
            "--param", "scalar_quartic=true",
            "--param-range", "n_scalars=1:2:2",
            "--guess", "g=0.65,lambda=0.14,lambda_phi=0.01",
            "--output", str(out),
        ])
        assert rc == 0
        arrays, _ = read_json(out)
        # theta_real / theta_imag are (n_values, n_couplings) padded with NaN
        assert "theta_real" in arrays
        assert "theta_imag" in arrays
        assert arrays["theta_real"].shape[0] == 2
