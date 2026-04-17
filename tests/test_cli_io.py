"""Output dispatch tests for asymsafety.cli.io."""

from __future__ import annotations

import json

import numpy as np
import pytest

from asymsafety.cli.io import read_json, read_npz, write_results
from asymsafety.cli.build_truncation import (
    SUPPORTED_TRUNCATIONS,
    build_truncation,
    parse_kv_pairs,
)


# ---------------------------------------------------------------------------
# build_truncation
# ---------------------------------------------------------------------------

class TestBuildTruncation:
    def test_eh(self):
        sys = build_truncation("eh", {})
        assert sys.coupling_names == ["g", "lambda"]

    def test_quadratic(self):
        sys = build_truncation("quadratic", {})
        assert sys.coupling_names == ["g", "lambda", "alpha", "beta"]

    def test_foliated(self):
        sys = build_truncation("foliated", {})
        assert "lambda_ADM" in sys.coupling_names

    def test_eh_matter(self):
        sys = build_truncation("eh_matter", {"n_scalars": "3"})
        assert sys.coupling_names == ["g", "lambda"]

    def test_gravity_matter_quartic(self):
        sys = build_truncation(
            "gravity_matter",
            {"n_scalars": "1", "scalar_quartic": "true"},
        )
        assert "lambda_phi" in sys.coupling_names

    def test_gravity_matter_yukawa_requires_dirac(self):
        with pytest.raises(ValueError, match="n_dirac"):
            build_truncation(
                "gravity_matter",
                {"n_scalars": "1", "yukawa": "true"},
            )

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown truncation"):
            build_truncation("doesnotexist", {})

    def test_supported_set_advertised(self):
        for name in SUPPORTED_TRUNCATIONS:
            assert isinstance(name, str)


class TestParseKvPairs:
    def test_empty(self):
        assert parse_kv_pairs("") == {}
        assert parse_kv_pairs(None) == {}

    def test_simple(self):
        assert parse_kv_pairs("a=1,b=2") == {"a": "1", "b": "2"}

    def test_whitespace(self):
        assert parse_kv_pairs("  a = 1 , b=2") == {"a": "1", "b": "2"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="missing"):
            parse_kv_pairs("foo")


# ---------------------------------------------------------------------------
# IO round trips
# ---------------------------------------------------------------------------

class TestNpzRoundTrip:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "out.npz"
        arrays = {
            "betas": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "grid": np.array([[0.1, 0.2], [0.3, 0.4]]),
        }
        meta = {"truncation": "eh", "n_points": 2}
        write_results(path, arrays, meta)
        arrs, m = read_npz(path)
        assert set(arrs.keys()) == {"betas", "grid"}
        np.testing.assert_array_equal(arrs["betas"], arrays["betas"])
        assert m["truncation"] == "eh"
        assert m["n_points"] == 2


class TestJsonRoundTrip:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "out.json"
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        meta = {"label": "test"}
        write_results(path, arrays, meta)
        arrs, m = read_json(path)
        np.testing.assert_array_equal(arrs["x"], arrays["x"])
        assert m == meta


class TestExtensionDispatch:
    def test_unknown_extension_falls_back_to_npz(self, tmp_path):
        path = tmp_path / "out.weird"
        write_results(path, {"x": np.array([1.0])}, {})
        # numpy.savez_compressed appends .npz when the extension is
        # unfamiliar — verify the file lands as out.weird.npz
        actual = path.with_suffix(path.suffix + ".npz")
        assert actual.exists()
        arrs, _ = read_npz(actual)
        assert "x" in arrs


class TestHdf5:
    def test_h5_requires_h5py_or_raises(self, tmp_path):
        try:
            import h5py  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="h5py"):
                write_results(tmp_path / "x.h5", {"a": np.array([1.0])}, {})
        else:
            # If h5py available, verify round-trip via h5py directly
            path = tmp_path / "x.h5"
            write_results(path, {"a": np.array([1.0, 2.0])}, {"k": 5})
            with h5py.File(path, "r") as fh:
                assert list(fh["a"][:]) == [1.0, 2.0]
                assert fh.attrs["k"] == 5
