"""``asymsafety eval`` — batch-evaluate β over a coupling grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from asymsafety.cli.build_truncation import (
    SUPPORTED_TRUNCATIONS,
    build_truncation,
    parse_kv_pairs,
)
from asymsafety.cli.io import write_results


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``eval`` subcommand."""
    p = subparsers.add_parser(
        "eval",
        help="Batch-evaluate β over a coupling grid.",
        description=(
            "Build a beta-function system, evaluate β at every point of "
            "an N-dimensional grid (Cartesian product of per-coupling "
            "linspaces), and save the result."
        ),
    )
    p.add_argument(
        "--truncation", required=True, choices=SUPPORTED_TRUNCATIONS,
        help="Beta-function truncation to evaluate.",
    )
    p.add_argument(
        "--param", default="", metavar="K=V,K=V",
        help=(
            "Truncation parameters (e.g. d=4, n_scalars=2, "
            "scalar_quartic=true). Comma-separated."
        ),
    )
    p.add_argument(
        "--grid", required=True, metavar="COUPLING:LO:HI:N[,...]",
        help=(
            "Per-coupling 1-D grid spec, e.g. 'g:0:1:50,lambda:-0.5:0.5:50'. "
            "The total candidate set is the Cartesian product."
        ),
    )
    p.add_argument(
        "--backend", default="numpy", choices=("numpy", "jax", "auto"),
        help="Batch evaluator backend (default: numpy).",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="FILE",
        help="Output file (.npz / .h5 / .json).",
    )
    p.set_defaults(func=run_eval)


def run_eval(args: argparse.Namespace) -> int:
    params = parse_kv_pairs(args.param)
    system = build_truncation(args.truncation, params)
    grid_spec = _parse_grid(args.grid)
    _validate_grid_against_system(grid_spec, system)
    grid_points, axes = _build_grid(grid_spec, system.coupling_names)

    evaluator = _select_backend(system, args.backend)
    betas = evaluator.evaluate_batch(grid_points)

    arrays = {
        "grid": grid_points,
        "betas": np.asarray(betas),
    }
    for name, axis in axes.items():
        arrays[f"axis_{name}"] = axis

    metadata = {
        "truncation": args.truncation,
        "params": params,
        "couplings": system.coupling_names,
        "n_points": len(grid_points),
        "backend": args.backend,
    }

    out = write_results(args.output, arrays, metadata)
    print(
        f"Evaluated {len(grid_points)} points × {len(system.coupling_names)} "
        f"couplings → {out}"
    )
    return 0


def _parse_grid(spec: str) -> list[tuple[str, float, float, int]]:
    """Parse 'g:0:1:50,lambda:-0.5:0.5:50' → list of (name, lo, hi, n)."""
    out: list[tuple[str, float, float, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Bad --grid token {token!r}; expected NAME:LO:HI:N"
            )
        name, lo, hi, n = parts
        out.append((name.strip(), float(lo), float(hi), int(n)))
    if not out:
        raise ValueError("--grid is empty")
    return out


def _validate_grid_against_system(
    spec: list[tuple[str, float, float, int]], system
) -> None:
    grid_names = {name for name, *_ in spec}
    sys_names = set(system.coupling_names)
    unknown = grid_names - sys_names
    if unknown:
        raise ValueError(
            f"--grid mentions unknown coupling(s) {sorted(unknown)}; "
            f"system has {sorted(sys_names)}"
        )


def _build_grid(
    spec: list[tuple[str, float, float, int]],
    coupling_names: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Construct the Cartesian product points.

    Couplings not mentioned in ``spec`` get a single-point axis at 0.0.
    """
    spec_map = {name: (lo, hi, n) for name, lo, hi, n in spec}
    axes: dict[str, np.ndarray] = {}
    for name in coupling_names:
        if name in spec_map:
            lo, hi, n = spec_map[name]
            axes[name] = np.linspace(lo, hi, n)
        else:
            axes[name] = np.array([0.0])

    mesh = np.meshgrid(*[axes[n] for n in coupling_names], indexing="ij")
    flat = np.column_stack([m.ravel() for m in mesh])
    return flat, axes


def _select_backend(system, request: str):
    if request == "auto":
        try:
            import jax  # noqa: F401
            request = "jax"
        except ImportError:
            request = "numpy"
    return system.batch_evaluator(request)
