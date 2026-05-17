#!/usr/bin/env python3
"""Generate the shipped animations under ``docs/animations/``.

This is the animation companion to :mod:`scripts.generate_figures` and
mirrors its parallel-ProcessPoolExecutor harness. Two animations ship
by default:

* ``rg_flow_ahm`` — RG trajectories flowing into the charged fixed
  point of the 3D Abelian-Higgs model, for ``N_f ∈ {30, 60}``.
* ``nu_vs_nf_sweep`` — the static
  :func:`asymsafety.visualization.fixed_point_plot.nu_vs_Nf` curve
  built point-by-point as ``N_f`` sweeps from 12 to 120.

Writer selection: ``--writer auto`` (default) prefers ``ffmpeg``
(``.mp4``) when on ``PATH``, else falls back to :class:`PillowWriter`
(``.gif``). Pass ``--writer pillow`` to force gif output (no system
ffmpeg dependency, suitable for CI).

Animation outputs are deliberately *not* hash-baselined: ffmpeg and
pillow produce byte-different files for identical frames. See
``tests/test_animations_smoke.py`` for the size-sanity check that
runs in CI.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Headless backend — must be set before any pyplot import.
import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Animation generators
# ---------------------------------------------------------------------------


def gen_anim_rg_flow_ahm(out_dir: Path, writer: str, fps: int) -> Path:
    """RG trajectories flowing into the AHM charged FP (attractive (u, r) plane).

    The charged FP is UV-attractive only on the critical surface (the
    `(u, r)` plane at fixed `α = α*`); the gauge direction `α` is
    UV-repulsive. Animating the full 3D flow with generic initial
    conditions therefore produces runaways. Instead we hold `α = α*`
    and integrate the projected `(u, r)` flow forward in RG time —
    every trajectory then spirals into the FP, illustrating the
    physical content of "the charged FP attracts the critical surface".
    """
    from asymsafety.transforms.bridge.gauge_higgs import (
        build_ahm_system,
        charged_fp_guess,
    )
    from asymsafety.visualization.animation import (
        rg_trajectory_animation,
        save_animation,
    )

    N = 60
    system = build_ahm_system(N=N, Nc=1, epsilon=1.0)
    rhs = system.rhs_vector()  # f(t, y) over (alpha, u, r)

    fp_guess = charged_fp_guess(N, epsilon=1.0)
    alpha_star = fp_guess["alpha"]
    u_star = fp_guess["u"]

    # Project to (u, r) at fixed alpha = alpha*; this is the critical
    # surface and the charged FP attracts under forward RG time.
    fp_2d = np.array([u_star, 0.0])

    def rhs_2d(t: float, y: np.ndarray) -> np.ndarray:
        full = np.array([alpha_star, y[0], y[1]])
        full_dot = np.array(rhs(t, full))
        return full_dot[1:]

    n_traj = 8
    theta = np.linspace(0, 2 * np.pi, n_traj, endpoint=False)
    radius_u = 0.6 * u_star
    radius_r = 0.05
    init = fp_2d[None, :] + np.stack(
        [radius_u * np.cos(theta), radius_r * np.sin(theta)], axis=-1
    )

    bounds = (
        -0.3 * u_star, 2.2 * u_star,
        -0.08, 0.08,
    )

    fig, update, n_frames = rg_trajectory_animation(
        rhs_2d, init, fp_2d,
        n_frames=80, dt=0.05, bounds=bounds,
        title=(rf"AHM critical surface: flow into charged FP  "
               rf"($N_f = {N}$, $N_c = 1$, $\alpha = \alpha^*$)"),
        coupling_labels=("u", "r"),
    )
    out_path = out_dir / "rg_flow_ahm"
    return save_animation(
        fig, update, n_frames, out_path, fps=fps, writer=writer,
    )


def gen_anim_nu_vs_nf_sweep(out_dir: Path, writer: str, fps: int) -> Path:
    """Sweep `N_f` from 12 to 120 and reveal `ν(N_f)` point by point."""
    from asymsafety.transforms.bridge.gauge_higgs import (
        correlation_length_exponent,
    )
    from asymsafety.validation.bonati_2025 import BONATI_LARGE_NF
    from asymsafety.visualization.animation import (
        parameter_sweep_animation,
        save_animation,
    )

    Nfs = np.arange(12, 121)
    nu_toolkit = np.array(
        [correlation_length_exponent(int(n), epsilon=1.0) for n in Nfs]
    )
    nu_large_nf = 1.0 - BONATI_LARGE_NF["nu_coeff"] / Nfs.astype(float)

    fig, update, n_frames = parameter_sweep_animation(
        Nfs.astype(float),
        {
            r"toolkit one-loop 4-$\varepsilon$": nu_toolkit,
            r"large-$N_f$:  $1 - 9.727/N_f$": nu_large_nf,
        },
        nu_toolkit,
        x_label=r"$N_f$",
        y_label=r"$\nu$",
        title=r"$\nu(N_f)$ at the charged FP: sweep",
        y_bounds=(0.4, 1.1),
        marker_color="tab:red",
    )
    out_path = out_dir / "nu_vs_nf_sweep"
    return save_animation(
        fig, update, n_frames, out_path, fps=fps, writer=writer,
    )


# Order matters: matches CLI --only matching by suffix after "gen_anim_".
ALL_ANIMATIONS = [
    gen_anim_rg_flow_ahm,
    gen_anim_nu_vs_nf_sweep,
]


# ---------------------------------------------------------------------------
# Worker harness (mirrors generate_figures.py)
# ---------------------------------------------------------------------------


def _worker(
    gen_fn_name: str, out_dir: str, writer: str, fps: int, timeout_s: float,
) -> tuple[str, str, float]:
    """Run one animation generator in this subprocess."""
    import matplotlib
    matplotlib.use("Agg")  # noqa: F401 — must hold in worker too

    t0 = time.time()
    gen_fn = globals().get(gen_fn_name)
    name = gen_fn_name[len("gen_anim_"):]
    if gen_fn is None:
        return name, f"FAIL: generator {gen_fn_name!r} not found", 0.0
    try:
        out_path = gen_fn(Path(out_dir), writer, fps)
        elapsed = time.time() - t0
        size_kb = out_path.stat().st_size / 1024
        return name, f"OK ({out_path.name}, {size_kb:.0f} KiB)", elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        return name, f"FAIL: {type(exc).__name__}: {exc}", elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="docs/animations",
        help="Directory for generated animations (default: docs/animations)",
    )
    parser.add_argument(
        "--writer", default="auto",
        choices=["auto", "ffmpeg", "pillow"],
        help="Animation writer (default: auto — prefers ffmpeg, "
             "falls back to pillow gif).",
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Frames per second (default: 24).",
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(2, (os.cpu_count() or 2))),
        help="Parallel worker processes (default: min(2, ncpu)).",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-animation timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Generate only the named animation(s). May be repeated. "
             "Example: --only rg_flow_ahm",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = ALL_ANIMATIONS
    if args.only:
        wanted = {f"gen_anim_{name}" for name in args.only}
        selected = [g for g in ALL_ANIMATIONS if g.__name__ in wanted]
        if not selected:
            print(f"No animations match --only {args.only}", file=sys.stderr)
            sys.exit(2)

    print(
        f"Generating {len(selected)} animation(s) → {out_dir}/ "
        f"(writer={args.writer}, fps={args.fps}, "
        f"workers={args.workers}, timeout={args.timeout:.0f}s)\n"
    )
    t_start = time.time()

    n_ok = 0
    failures: list[tuple[str, str]] = []

    if args.workers <= 1:
        for gen_fn in selected:
            name = gen_fn.__name__[len("gen_anim_"):]
            t0 = time.time()
            try:
                out_path = gen_fn(out_dir, args.writer, args.fps)
                elapsed = time.time() - t0
                size_kb = out_path.stat().st_size / 1024
                print(
                    f"  OK   {out_path.name}  ({size_kb:.0f} KiB, "
                    f"{elapsed:.1f}s)",
                    flush=True,
                )
                n_ok += 1
            except Exception as exc:
                elapsed = time.time() - t0
                print(
                    f"  FAIL {name}: {type(exc).__name__}: {exc}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
                failures.append((name, str(exc)))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx,
        ) as ex:
            futures = {
                ex.submit(
                    _worker,
                    gen_fn.__name__, str(out_dir),
                    args.writer, args.fps, args.timeout,
                ): gen_fn.__name__
                for gen_fn in selected
            }
            for fut in as_completed(futures):
                gen_name = futures[fut]
                try:
                    name, status, elapsed = fut.result()
                except Exception as exc:
                    name = gen_name[len("gen_anim_"):]
                    status = f"FAIL: {exc}"
                    elapsed = 0.0
                if status.startswith("OK"):
                    n_ok += 1
                    print(f"  OK   {name}  {status[3:]}  ({elapsed:.1f}s)",
                          flush=True)
                else:
                    failures.append((name, status))
                    print(f"  {status:<60}  {name}  ({elapsed:.1f}s)",
                          flush=True)

    elapsed = time.time() - t_start
    print(
        f"\nDone in {elapsed:.1f}s: "
        f"{n_ok} succeeded, {len(failures)} failed."
    )
    if failures:
        print("\nFailed animations:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
