"""``asymsafety scan`` — find an NGFP for each value of an external parameter.

Sweeps an integer parameter (e.g. ``n_scalars``) over a range and finds
the shifted NGFP at each point. Records location and critical exponents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from asymsafety.analysis.fixed_points import FixedPointFinder
from asymsafety.cli.build_truncation import (
    SUPPORTED_TRUNCATIONS,
    build_truncation,
    parse_kv_pairs,
)
from asymsafety.cli.io import write_results


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "scan",
        help="Sweep one external parameter, find the NGFP at each value.",
        description=(
            "For each integer value in the requested range, build the "
            "truncation with the override applied, and search for an NGFP "
            "from the supplied --guess. Records (location, exponents)."
        ),
    )
    p.add_argument(
        "--truncation", required=True, choices=SUPPORTED_TRUNCATIONS,
        help="Beta-function truncation to sweep.",
    )
    p.add_argument(
        "--param", default="", metavar="K=V,K=V",
        help="Constant truncation parameters (forwarded to every build).",
    )
    p.add_argument(
        "--param-range", required=True, metavar="NAME=LO:HI:N",
        help="Integer parameter range, e.g. 'n_scalars=1:8:8'.",
    )
    p.add_argument(
        "--guess", default="", metavar="K=V,K=V",
        help="Initial coupling guess for the NGFP search.",
    )
    p.add_argument(
        "--tol", type=float, default=1.0e-10,
        help="Convergence tolerance for fsolve (default 1e-10).",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="FILE",
        help="Output file (.npz / .json recommended).",
    )
    p.set_defaults(func=run_scan)


def run_scan(args: argparse.Namespace) -> int:
    constant_params = parse_kv_pairs(args.param)
    name, values = _parse_param_range(args.param_range)
    raw_guess = parse_kv_pairs(args.guess)
    guess = {k: float(v) for k, v in raw_guess.items()}

    rows: list[dict] = []
    couplings: list[str] = []

    for value in values:
        params = dict(constant_params)
        params[name] = str(value)
        system = build_truncation(args.truncation, params)
        if not couplings:
            couplings = list(system.coupling_names)
        finder = FixedPointFinder(system)
        # Fall through unknown couplings to 0.0 via the finder's own logic
        fp = finder.find_fixed_point(guess, tol=args.tol)
        row: dict = {name: int(value), "fp_exists": fp is not None}
        if fp is not None and not fp.is_gaussian:
            row.update({c: float(fp.location.get(c, 0.0)) for c in couplings})
            row["theta_real"] = [float(t.real) for t in fp.critical_exponents]
            row["theta_imag"] = [float(t.imag) for t in fp.critical_exponents]
            row["n_relevant"] = int(np.sum(fp.critical_exponents.real > 0))
            # Warm-start the next iteration from the just-found FP
            guess = {c: float(fp.location.get(c, 0.0)) for c in couplings}
        else:
            row["fp_exists"] = False
        rows.append(row)

    arrays = _rows_to_arrays(rows, name, couplings)
    metadata = {
        "truncation": args.truncation,
        "params": constant_params,
        "param_swept": name,
        "n_values": len(values),
        "couplings": couplings,
    }
    out = write_results(args.output, arrays, metadata)
    n_found = sum(1 for r in rows if r["fp_exists"])
    print(
        f"Scanned {len(values)} values of {name}: "
        f"{n_found} NGFPs found → {out}"
    )
    return 0


def _parse_param_range(spec: str) -> tuple[str, list[int]]:
    if "=" not in spec:
        raise ValueError(f"--param-range {spec!r} missing '='")
    name, range_str = spec.split("=", 1)
    parts = range_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--param-range value {range_str!r} must be LO:HI:N"
        )
    lo, hi, n = int(parts[0]), int(parts[1]), int(parts[2])
    values = np.linspace(lo, hi, n).round().astype(int).tolist()
    return name.strip(), values


def _rows_to_arrays(
    rows: list[dict],
    swept: str,
    couplings: list[str],
) -> dict[str, np.ndarray]:
    """Convert per-row dicts into parallel arrays for HDF5/NPZ storage."""
    n = len(rows)
    arrays: dict[str, np.ndarray] = {
        swept: np.array([r[swept] for r in rows], dtype=int),
        "fp_exists": np.array([r["fp_exists"] for r in rows], dtype=bool),
        "n_relevant": np.array(
            [r.get("n_relevant", -1) for r in rows], dtype=int
        ),
    }
    for c in couplings:
        arrays[c] = np.array(
            [r.get(c, np.nan) for r in rows], dtype=float
        )
    # Critical exponents are per-row vectors; pad to ragged-uniform if mixed
    n_couplings = len(couplings)
    theta_re = np.full((n, n_couplings), np.nan)
    theta_im = np.full((n, n_couplings), np.nan)
    for i, r in enumerate(rows):
        if r.get("theta_real") is not None:
            vals = r["theta_real"]
            theta_re[i, : len(vals)] = vals
        if r.get("theta_imag") is not None:
            vals = r["theta_imag"]
            theta_im[i, : len(vals)] = vals
    arrays["theta_real"] = theta_re
    arrays["theta_imag"] = theta_im
    return arrays
