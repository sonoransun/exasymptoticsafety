"""Animation utilities for RG flow and parameter sweeps.

Thin wrapper around :class:`matplotlib.animation.FuncAnimation`. This
is the first animation infrastructure in the repository — kept
deliberately minimal so that other modules can drive their own
``update`` callbacks without dragging in a heavy framework.

Output writer selection
-----------------------
The writer is chosen at runtime by :func:`save_animation`:

* ``writer="auto"`` (default) — use ``ffmpeg`` if it is on ``PATH``
  (outputs ``.mp4``), otherwise fall back to :class:`PillowWriter`
  (outputs ``.gif``). The chosen extension is appended to
  ``out_path`` if missing.
* ``writer="ffmpeg"`` / ``writer="pillow"`` — force a specific
  backend; raises ``RuntimeError`` if unavailable.

We deliberately do **not** hash-baseline animation outputs: ``ffmpeg``
and ``pillow`` produce byte-different files for identical frames, so
pixel-level regression checks would be unstable. Tests in
``tests/test_animations_smoke.py`` perform size-sanity checks only.

The companion script :mod:`asymsafety.scripts.generate_animations`
(invoked as ``python scripts/generate_animations.py``) emits the
shipped animations into ``docs/animations/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.figure import Figure
except ImportError as e:  # pragma: no cover - matplotlib is a core dep
    raise ImportError(
        "matplotlib is required for asymsafety.visualization.animation"
    ) from e


_GIF_EXT = ".gif"
_MP4_EXT = ".mp4"


def _select_writer(writer: str) -> tuple[str, str]:
    """Resolve ``writer="auto"`` to a concrete (writer_name, extension)."""
    if writer == "auto":
        if shutil.which("ffmpeg") is not None:
            return "ffmpeg", _MP4_EXT
        return "pillow", _GIF_EXT
    if writer == "ffmpeg":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "writer='ffmpeg' requested but ffmpeg is not on PATH"
            )
        return "ffmpeg", _MP4_EXT
    if writer == "pillow":
        return "pillow", _GIF_EXT
    raise ValueError(f"Unknown writer: {writer!r}")


def save_animation(
    fig: Figure,
    update_fn: Callable[[int], Iterable],
    frames: int,
    out_path: str | Path,
    *,
    fps: int = 24,
    writer: str = "auto",
    init_fn: Callable[[], Iterable] | None = None,
    blit: bool = False,
    dpi: int = 100,
) -> Path:
    """Build and save a :class:`FuncAnimation` from ``fig`` + ``update_fn``.

    Parameters
    ----------
    fig
        Pre-built figure. The ``update_fn`` should mutate artists on
        this figure and return the modified artists (for ``blit=True``)
        or any iterable (for ``blit=False``).
    update_fn
        Callable ``(frame_idx) -> iterable_of_artists``.
    frames
        Number of frames to render.
    out_path
        Destination path; the extension is overridden to match the
        chosen writer if it does not already match.
    fps, dpi
        Standard FuncAnimation knobs.
    writer
        ``"auto"`` (default), ``"ffmpeg"`` (``.mp4``), or
        ``"pillow"`` (``.gif``).
    init_fn, blit
        Passed through to :class:`FuncAnimation`.

    Returns
    -------
    pathlib.Path
        The final path written.
    """
    writer_name, default_ext = _select_writer(writer)

    out_path = Path(out_path)
    if out_path.suffix.lower() not in (_GIF_EXT, _MP4_EXT):
        out_path = out_path.with_suffix(default_ext)
    elif out_path.suffix.lower() != default_ext:
        # Honor the caller's extension only if it is consistent with the
        # chosen writer — otherwise force the writer's native extension.
        out_path = out_path.with_suffix(default_ext)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    anim = FuncAnimation(
        fig, update_fn,
        frames=frames,
        init_func=init_fn,
        blit=blit,
        interval=1000 / fps,
    )

    if writer_name == "ffmpeg":
        writer_obj = FFMpegWriter(fps=fps)
    else:
        writer_obj = PillowWriter(fps=fps)

    anim.save(str(out_path), writer=writer_obj, dpi=dpi)
    plt.close(fig)
    return out_path


# ── ready-made animation builders ────────────────────────────────────


def rg_trajectory_animation(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_points: np.ndarray,
    fixed_point: np.ndarray,
    *,
    n_frames: int = 90,
    dt: float = 0.05,
    bounds: tuple[float, float, float, float] | None = None,
    title: str = "RG flow",
    coupling_labels: tuple[str, str] = ("g_1", "g_2"),
) -> tuple[Figure, Callable[[int], Iterable], int]:
    """Build a 2D RG-flow trajectory animation.

    Integrates each row of ``initial_points`` forward under ``rhs`` for
    ``n_frames`` Euler steps of size ``dt`` and reveals the trajectory
    point-by-point, with the fixed point marked as a star. Returns
    ``(fig, update_fn, frames)`` — pass through :func:`save_animation`.

    Parameters
    ----------
    rhs
        RHS in the form ``f(t, y) -> dy/dt`` for a 2D state vector.
        Use :meth:`BetaFunctionSystem.rhs_vector` for the natural
        signature.
    initial_points
        Array of shape ``(n_traj, 2)``.
    fixed_point
        Length-2 array marking the target FP.
    n_frames, dt
        Number of forward Euler steps and step size.
    bounds
        ``(x_min, x_max, y_min, y_max)``. If ``None``, inferred from
        the initial-point cloud plus a 30% margin.
    title, coupling_labels
        Plot decorations.
    """
    n_traj = initial_points.shape[0]
    trajectories = np.zeros((n_traj, n_frames + 1, 2))
    trajectories[:, 0, :] = initial_points
    for i in range(n_frames):
        for j in range(n_traj):
            dy = rhs(0.0, trajectories[j, i, :])
            trajectories[j, i + 1, :] = trajectories[j, i, :] + dt * dy

    if bounds is None:
        all_pts = np.vstack([initial_points, fixed_point[None, :]])
        x_min, y_min = all_pts.min(axis=0) - 0.3 * np.ptp(all_pts, axis=0)
        x_max, y_max = all_pts.max(axis=0) + 0.3 * np.ptp(all_pts, axis=0)
        bounds = (x_min, x_max, y_min, y_max)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel(f"${coupling_labels[0]}$", fontsize=12)
    ax.set_ylabel(f"${coupling_labels[1]}$", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.plot(fixed_point[0], fixed_point[1], "*",
            ms=20, mfc="gold", mec="black", mew=1.0, zorder=5,
            label="charged FP")
    ax.legend(loc="upper right", fontsize=9)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_traj))
    lines = [
        ax.plot([], [], color=c, lw=1.7, alpha=0.85)[0] for c in colors
    ]
    heads = [
        ax.plot([], [], "o", ms=5, color=c)[0] for c in colors
    ]
    frame_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
        va="top", color="0.3",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7",
                  alpha=0.85),
    )

    def update(i: int):
        for j, (line, head) in enumerate(zip(lines, heads)):
            seg = trajectories[j, : i + 1, :]
            line.set_data(seg[:, 0], seg[:, 1])
            head.set_data([seg[-1, 0]], [seg[-1, 1]])
        frame_text.set_text(f"t = {i * dt:.2f}")
        return [*lines, *heads, frame_text]

    return fig, update, n_frames + 1


def parameter_sweep_animation(
    x_values: np.ndarray,
    y_static_curves: dict[str, np.ndarray] | None,
    y_sweep_values: np.ndarray,
    *,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Parameter sweep",
    y_bounds: tuple[float, float] | None = None,
    marker_color: str = "tab:red",
) -> tuple[Figure, Callable[[int], Iterable], int]:
    """Build an animation that reveals a curve point-by-point.

    Useful for pedagogical figures: shows a reference curve (or
    several) in the background, then reveals a moving marker tracing
    out ``y_sweep_values`` against ``x_values`` one point per frame.

    Returns
    -------
    fig, update_fn, frames
        Pass through :func:`save_animation`.
    """
    if x_values.shape != y_sweep_values.shape:
        raise ValueError(
            "x_values and y_sweep_values must have the same shape"
        )

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_values.min(), x_values.max())
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)

    if y_static_curves:
        for name, ys in y_static_curves.items():
            ax.plot(x_values, ys, lw=1.6, alpha=0.55, label=name)
        ax.legend(loc="best", fontsize=9)

    swept_line, = ax.plot([], [], "-", color=marker_color, lw=2.2,
                          alpha=0.9, zorder=5)
    marker, = ax.plot([], [], "o", color=marker_color, ms=9,
                      mec="black", mew=0.8, zorder=6)

    def update(i: int):
        swept_line.set_data(x_values[: i + 1], y_sweep_values[: i + 1])
        marker.set_data([x_values[i]], [y_sweep_values[i]])
        return [swept_line, marker]

    return fig, update, len(x_values)
