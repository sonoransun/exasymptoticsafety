"""Tests for the AS-vs-string scattering bridge."""

import pytest

from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
from asymsafety.scattering.bootstrap import StringAmplitude
from asymsafety.scattering.bridge import ScatteringBridge
from asymsafety.scattering.form_factor import GravitonFormFactor


@pytest.fixture
def bridge(as_trajectory):
    amp = GravitonMediatedAmplitude(GravitonFormFactor(as_trajectory))
    string = StringAmplitude(kind="virasoro_shapiro", alphap=0.25)
    return ScatteringBridge(amp, string)


class TestHighEnergy:
    def test_as_uv_constant_string_ultrasoft(self, bridge):
        he = bridge.compare_high_energy()
        assert he["as_uv_constant"]          # AS amplitude UV-constant
        assert not he["as_ultrasoft"]        # ... not ultrasoft
        assert he["string_ultrasoft"]        # strings are ultrasoft
        assert he["distinct_uv"]


class TestConsistencyTable:
    def test_both_crossing_symmetric(self, bridge):
        table = bridge.consistency_table()
        assert table["crossing"]["as"]
        assert table["crossing"]["string"]

    def test_regge_tower_only_strings(self, bridge):
        table = bridge.consistency_table()
        assert table["infinite_regge_tower"]["as"] is False
        assert table["infinite_regge_tower"]["string"] is True

    def test_ultrasoft_only_strings(self, bridge):
        table = bridge.consistency_table()
        assert table["ultrasoft_falloff"]["as"] is False
        assert table["ultrasoft_falloff"]["string"] is True

    def test_as_foundational_pass(self, bridge):
        table = bridge.consistency_table()
        assert table["uv_finite_or_bounded"]["as"]
        assert table["partial_wave_bounded"]["as"]
        assert table["no_ghost_pole"]["as"]

    def test_full_comparison_table_shape(self, bridge):
        rows = bridge.full_comparison_table()
        assert len(rows) == 6
        assert {r["criterion"] for r in rows} >= {"crossing", "ultrasoft_falloff"}


class TestVerdict:
    def test_as_physically_consistent(self, bridge):
        v = bridge.verify()
        assert v["as_physically_consistent"]
        assert all(v["foundational_checks"].values())

    def test_distinct_from_strings(self, bridge):
        v = bridge.verify()
        assert v["distinct_from_strings"]
        assert v["distinguishing_features"]["as_uv_constant_not_ultrasoft"]
        assert "softening" in v["summary"]
