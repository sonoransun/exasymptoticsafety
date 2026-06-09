"""Tests for the ``asymsafety amplitude`` CLI subcommand."""

import numpy as np
import pytest

from asymsafety.cli.io import read_json, read_npz
from asymsafety.cli.main import main


def test_amplitude_npz(tmp_path):
    out = tmp_path / "amp.npz"
    rc = main([
        "amplitude", "--truncation", "eh", "--guess", "g=0.7,lambda=0.14",
        "--s-range", "1e-2:1e6:40", "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()
    arrays, meta = read_npz(out)
    assert set(arrays) >= {"s", "abs_M_as", "abs_M_gr", "form_factor"}
    assert arrays["s"].shape == (40,)
    # AS amplitude bounded relative to GR at the highest energy.
    assert arrays["abs_M_as"][-1] < arrays["abs_M_gr"][-1]
    assert meta["truncation"] == "eh"
    assert "fixed_point" in meta


def test_amplitude_with_checks_and_bridge_json(tmp_path):
    out = tmp_path / "amp.json"
    rc = main([
        "amplitude", "--truncation", "eh", "--guess", "g=0.7,lambda=0.14",
        "--s-range", "1e-2:1e6:30", "--checks", "--compare-string",
        "--output", str(out),
    ])
    assert rc == 0
    _arrays, meta = read_json(out)
    assert meta["consistency"]["uv_finite"]
    # The IR-upward trajectory construction reaches the Newtonian regime,
    # so the (honest) roll-up that includes IR recovery passes for real.
    assert meta["consistency"]["ir_newtonian_recovery"]
    assert meta["consistency"]["all_passed"]
    assert abs(meta["ir_ratio_to_gr"] - 1.0) < 0.1
    assert meta["bridge"]["as_physically_consistent"]
    assert meta["bridge"]["distinct_from_strings"]


def test_amplitude_fixed_scale_is_unsafe(tmp_path):
    # A frozen scale should NOT soften -> AS curve grows like GR.
    out = tmp_path / "amp.npz"
    rc = main([
        "amplitude", "--truncation", "eh", "--guess", "g=0.7,lambda=0.14",
        "--scale", "fixed", "--xi-scale", "1.0",
        "--s-range", "1e-2:1e6:30", "--output", str(out),
    ])
    assert rc == 0
    arrays, _ = read_npz(out)
    # With no running, the dressed amplitude grows (form factor ~ const).
    assert arrays["abs_M_as"][-1] > 100.0 * arrays["abs_M_as"][0]


def test_amplitude_gravity_matter_with_xi(tmp_path):
    out = tmp_path / "amp.npz"
    rc = main([
        "amplitude", "--truncation", "gravity_matter",
        "--param", "n_scalars=1,scalar_quartic=true,running_xi=true",
        "--guess", "g=0.69,lambda=0.14,lambda_phi=0.011,xi=0.0",
        "--s-range", "1e-2:1e5:25", "--output", str(out),
    ])
    assert rc == 0
    arrays, meta = read_npz(out)
    assert "xi" in meta["couplings"]
    assert np.isfinite(arrays["abs_M_as"]).all()
