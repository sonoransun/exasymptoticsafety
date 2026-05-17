"""Smoke tests for the animation infrastructure.

Animation outputs are deliberately not pixel-hashed (ffmpeg and pillow
produce byte-different files for identical frames). These tests assert
only that a representative animation:

* generates without raising;
* emits a file with a writer-appropriate extension;
* has a non-trivial but bounded size (10 KiB < size < 10 MiB).

We force ``writer="pillow"`` so the test does not depend on the
system having ``ffmpeg`` on ``PATH``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None
pytestmark = pytest.mark.skipif(
    not PILLOW_AVAILABLE,
    reason="Pillow is required for the gif animation smoke test",
)


def test_parameter_sweep_animation_smoke(tmp_path: Path):
    """A short pillow gif is produced and is within reasonable size bounds."""
    import matplotlib
    matplotlib.use("Agg")

    from asymsafety.visualization.animation import (
        parameter_sweep_animation,
        save_animation,
    )

    x = np.linspace(0.0, 2 * np.pi, 12)
    y = np.sin(x)
    fig, update, n_frames = parameter_sweep_animation(
        x, {"sin": y}, y,
        x_label="x", y_label="sin(x)", title="smoke",
        y_bounds=(-1.2, 1.2),
    )
    out = save_animation(
        fig, update, n_frames, tmp_path / "smoke.gif",
        fps=12, writer="pillow",
    )

    assert out.exists()
    assert out.suffix == ".gif"
    size = out.stat().st_size
    assert 10 * 1024 < size < 10 * 1024 * 1024, (
        f"Animation file size {size} outside 10 KiB .. 10 MiB sanity range"
    )


def test_rg_trajectory_animation_smoke(tmp_path: Path):
    """The RG-trajectory helper produces a sensible gif from a toy RHS."""
    import matplotlib
    matplotlib.use("Agg")

    from asymsafety.visualization.animation import (
        rg_trajectory_animation,
        save_animation,
    )

    def decay_rhs(t: float, y: np.ndarray) -> np.ndarray:
        return -np.asarray(y)

    initial = np.array([[1.0, 0.5], [-0.7, 0.8], [0.3, -0.6]])
    fp = np.array([0.0, 0.0])
    fig, update, n_frames = rg_trajectory_animation(
        decay_rhs, initial, fp,
        n_frames=8, dt=0.2, coupling_labels=("g_1", "g_2"),
    )
    out = save_animation(
        fig, update, n_frames, tmp_path / "rg_smoke.gif",
        fps=10, writer="pillow",
    )

    assert out.exists()
    assert out.suffix == ".gif"
    size = out.stat().st_size
    assert 10 * 1024 < size < 10 * 1024 * 1024


def test_save_animation_rejects_unknown_writer(tmp_path: Path):
    """Unknown writer name raises ValueError before any work is done."""
    import matplotlib
    matplotlib.use("Agg")

    from asymsafety.visualization.animation import (
        parameter_sweep_animation,
        save_animation,
    )

    x = np.linspace(0, 1, 5)
    y = x ** 2
    fig, update, n_frames = parameter_sweep_animation(
        x, None, y, x_label="x", y_label="y", title="reject",
    )
    with pytest.raises(ValueError):
        save_animation(
            fig, update, n_frames, tmp_path / "bad.gif",
            writer="nonexistent-writer",
        )
