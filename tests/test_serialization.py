"""Tests for BetaFunctionSystem serialization."""

import pytest
import numpy as np

from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.compute.distributed.serialization import (
    serialize_beta_system,
    deserialize_beta_system,
    serialize_to_json,
    deserialize_from_json,
    generate_standalone_code,
)


class TestSerialization:
    """Test round-trip serialization of BetaFunctionSystem."""

    def setup_method(self):
        self.system = build_eh_beta_system(d=4)

    def test_round_trip_dict(self):
        """serialize → deserialize should produce identical beta functions."""
        data = serialize_beta_system(self.system)
        restored = deserialize_beta_system(data)

        # Same coupling names
        assert restored.coupling_names == self.system.coupling_names

        # Same beta function values at a test point
        point = {"g": 0.7, "lambda": 0.14}
        orig = self.system.evaluate(point)
        rest = restored.evaluate(point)
        for name in self.system.coupling_names:
            np.testing.assert_allclose(rest[name], orig[name], rtol=1e-10)

    def test_round_trip_json(self):
        """JSON round-trip should preserve beta functions."""
        json_str = serialize_to_json(self.system)
        restored = deserialize_from_json(json_str)

        point = {"g": 0.5, "lambda": 0.1}
        orig = self.system.evaluate(point)
        rest = restored.evaluate(point)
        for name in self.system.coupling_names:
            np.testing.assert_allclose(rest[name], orig[name], rtol=1e-10)

    def test_json_is_string(self):
        """serialize_to_json should return a valid JSON string."""
        json_str = serialize_to_json(self.system)
        assert isinstance(json_str, str)
        import json
        data = json.loads(json_str)  # Should not raise
        assert "coupling_names" in data
        assert "expressions" in data

    def test_standalone_code_generation(self):
        """generate_standalone_code should produce valid Python."""
        code = generate_standalone_code(self.system)
        assert "def beta_g" in code
        assert "def beta_lambda" in code
        assert "def evaluate_all" in code
        assert "import numpy" in code

        # The generated code should be syntactically valid
        compile(code, "<generated>", "exec")

    def test_serialized_data_structure(self):
        """Serialized dict should have expected keys."""
        data = serialize_beta_system(self.system)
        assert "coupling_names" in data
        assert "couplings" in data
        assert "expressions" in data
        assert len(data["coupling_names"]) == 2
        assert len(data["couplings"]) == 2
        assert len(data["expressions"]) == 2
