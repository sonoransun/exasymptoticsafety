"""Image-hash regression tests for deterministic figures.

For figures whose output is byte-deterministic — conceptual diagrams,
closed-form heat-kernel coefficient bars, scale-identification curves
— this module compares the sha256 of the freshly generated PNG to a
baseline stored in ``tests/baselines/<name>.png.sha256``. A mismatch
indicates an unintended rendering change and fails the test.

Skip strategies
---------------
- Set ``ASYMSAFETY_ALLOW_RERENDER=1`` to bypass the hash check
  (e.g., when intentionally regenerating images during a matplotlib
  upgrade — pair with ``scripts/generate_figures.py --update-baselines``
  to refresh the stored hashes).
- A missing baseline file skips its test (so adding new entries to
  ``_HASH_TESTED`` doesn't break CI before the baseline is generated).

Determinism notes
-----------------
The hash-tested set deliberately excludes:
- Anything depending on ODE integration with absolute tolerances
  whose floating-point output drifts across SciPy versions.
- Monte-Carlo / random-init figures (``flow_basin_3d``).
- 3D figures whose camera-orientation defaults flicker between
  matplotlib versions.
- Continuation plots that re-run a fixed-point search.

The deterministic survivors are mostly conceptual schematics, plus a
handful of numerical figures whose data path is closed-form
(`heat_kernel_coefficients`, `scale_identification`).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # noqa: E402

# Import the generators module from the script directory; we treat
# the script file as a regular module by inserting its parent path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
_gen = importlib.import_module("generate_figures")

# The single source of truth for which figures get hash-tested lives in
# the generator module. We re-export it here so a quick `pytest -v`
# trace shows exactly which figures CI is gating on.
HASHED_FIGURES = list(_gen._HASH_TESTED)  # noqa: SLF001
_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines"


def _generator_for(name: str):
    fn = getattr(_gen, f"gen_{name}", None)
    if fn is None:
        raise LookupError(
            f"no generator named gen_{name} in scripts/generate_figures.py"
        )
    return fn


@pytest.mark.parametrize("name", HASHED_FIGURES)
def test_figure_hash(name: str, tmp_path: Path) -> None:
    """Render *name* and compare its sha256 against the baseline."""
    if os.environ.get("ASYMSAFETY_ALLOW_RERENDER"):
        pytest.skip("ASYMSAFETY_ALLOW_RERENDER set — skipping hash check.")

    baseline_file = _BASELINE_DIR / f"{name}.png.sha256"
    if not baseline_file.exists():
        pytest.skip(
            f"no baseline at {baseline_file.relative_to(_REPO_ROOT)} — "
            "regenerate with: python scripts/generate_figures.py "
            f"--update-baselines --only {name}"
        )

    from asymsafety.visualization.style import apply_style
    apply_style()

    gen = _generator_for(name)
    gen(tmp_path, "png")

    image_path = tmp_path / f"{name}.png"
    assert image_path.exists(), f"generator did not produce {image_path}"

    expected = baseline_file.read_text().strip()
    actual = hashlib.sha256(image_path.read_bytes()).hexdigest()

    assert actual == expected, (
        f"image hash mismatch for {name}:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"  baseline: {baseline_file.relative_to(_REPO_ROOT)}\n"
        "If the rendering change is intentional, refresh the baseline:\n"
        "  python scripts/generate_figures.py "
        f"--update-baselines --only {name}\n"
        "Otherwise inspect the diff between the new PNG and the previous one."
    )
