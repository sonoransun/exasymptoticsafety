"""``asymsafety amplitude`` — graviton-mediated scattering amplitudes.

Builds a beta-function truncation, locates its non-Gaussian fixed point,
integrates an RG trajectory, promotes the running couplings to a
momentum-dependent graviton form factor, and samples the graviton-mediated
2 → 2 scalar amplitude ``|M(√s)|`` for both the asymptotically-safe
(dressed) and the classical-GR (undressed) cases.  Optionally runs the
physical-scattering consistency battery and the AS-vs-string bridge.

See :mod:`asymsafety.scattering` for the physics and the RG-improvement
caveat.
"""

from __future__ import annotations

import argparse

import numpy as np

from asymsafety.cli.build_truncation import (
    SUPPORTED_TRUNCATIONS,
    build_truncation,
    parse_kv_pairs,
)
from asymsafety.cli.io import write_results


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``amplitude`` subcommand."""
    p = subparsers.add_parser(
        "amplitude",
        help="Graviton-mediated 2->2 scalar scattering amplitudes.",
        description=(
            "Promote a fixed-point RG trajectory to a momentum-dependent "
            "graviton form factor and sample the scattering amplitude vs "
            "energy, with optional physical-consistency and string-bootstrap "
            "comparison."
        ),
    )
    p.add_argument(
        "--truncation", default="eh", choices=SUPPORTED_TRUNCATIONS,
        help="Beta-function truncation (must expose a 'g' coupling).",
    )
    p.add_argument(
        "--param", default="", metavar="K=V,K=V",
        help="Truncation parameters (e.g. n_scalars=4,running_xi=true).",
    )
    p.add_argument(
        "--guess", default="g=0.7,lambda=0.2", metavar="K=V,K=V",
        help="Fixed-point initial guess (default g=0.7,lambda=0.2).",
    )
    p.add_argument(
        "--scale", default="energy", choices=("energy", "fixed"),
        help="Momentum-scale identification: energy (k=xi*sqrt(s)) or fixed.",
    )
    p.add_argument(
        "--xi-scale", type=float, default=1.0,
        help="Matching coefficient xi in k(p^2)=xi*sqrt(|p^2|).",
    )
    p.add_argument(
        "--s-range", default="1e-2:1e8:200", metavar="LO:HI:N",
        help="Geometric s grid (Mandelstam, in k0^2 units).",
    )
    p.add_argument(
        "--angle", type=float, default=0.3, metavar="COS_THETA",
        help="Fixed scattering-angle cosine (default 0.3).",
    )
    p.add_argument(
        "--checks", action="store_true",
        help="Run the physical-scattering consistency battery.",
    )
    p.add_argument(
        "--compare-string", action="store_true",
        help="Run the AS-vs-string bootstrap bridge.",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="FILE",
        help="Output file (.npz / .h5 / .json).",
    )
    p.set_defaults(func=run_amplitude)


def _parse_srange(spec: str) -> tuple[float, float, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad --s-range {spec!r}; expected LO:HI:N")
    return float(parts[0]), float(parts[1]), int(parts[2])


def _build_trajectory(system, guess: dict[str, float]):
    """Locate the fixed point and integrate a UV→IR trajectory."""
    from asymsafety.analysis.fixed_points import FixedPointFinder
    from asymsafety.analysis.flow import FlowIntegrator

    names = system.coupling_names
    full_guess = {n: float(guess.get(n, 0.01)) for n in names}
    fp = FixedPointFinder(system).find_fixed_point(full_guess)
    if fp is None:
        raise ValueError(
            f"No fixed point found from guess {full_guess}; try --guess."
        )
    # The NGFP is UV-attractive (all gravitational directions relevant),
    # so integrating *upward* from a deep-IR point with classical scaling
    # g = G_N k² converges onto the fixed point toward the UV. The result
    # is the Type-IIIa-like trajectory the form factor needs: G(p²) ≈ G_N
    # constant in the IR, g → g* (UV softening) above the transition.
    # Perturbing off the NGFP and flowing down instead runs into the
    # λ = 1/2 pole or the λ → -∞ runaway after a few e-folds.
    import math

    g_newton_ir = 0.02
    t_ir = -8.0
    ic_ir = {n: 0.0 for n in names}
    ic_ir["g"] = g_newton_ir * math.exp(2.0 * t_ir)
    traj = FlowIntegrator(system).integrate(
        ic_ir, t_span=(t_ir, 15.0), max_step=0.05
    )
    return fp, traj


def run_amplitude(args: argparse.Namespace) -> int:
    from asymsafety.scattering.amplitude import GravitonMediatedAmplitude
    from asymsafety.scattering.form_factor import GravitonFormFactor
    from asymsafety.scattering.scale import EnergyScale, FixedScale

    params = parse_kv_pairs(args.param)
    guess = {k: float(v) for k, v in parse_kv_pairs(args.guess).items()}
    system = build_truncation(args.truncation, params)
    if "g" not in system.coupling_names:
        raise ValueError(
            f"Truncation {args.truncation!r} has no 'g' coupling; "
            "graviton scattering needs the Newton coupling."
        )

    fp, traj = _build_trajectory(system, guess)

    scale = (
        EnergyScale(xi=args.xi_scale)
        if args.scale == "energy"
        else FixedScale(k_fixed=args.xi_scale)
    )
    form_factor = GravitonFormFactor(traj, scale=scale)
    amplitude = GravitonMediatedAmplitude(
        form_factor, include_xi="xi" in system.coupling_names
    )

    lo, hi, n = _parse_srange(args.s_range)
    s = np.geomspace(lo, hi, n)
    abs_M_as = np.abs(amplitude.amplitude_vs_s(s, args.angle, dressed=True))
    abs_M_gr = np.abs(amplitude.amplitude_vs_s(s, args.angle, dressed=False))
    f_vals = np.asarray(form_factor.f(s), dtype=float)

    ir = amplitude.ir_limit(cos_theta=args.angle)
    uv = amplitude.uv_limit(cos_theta=args.angle)

    arrays = {
        "s": s,
        "sqrt_s": np.sqrt(s),
        "abs_M_as": abs_M_as,
        "abs_M_gr": abs_M_gr,
        "form_factor": f_vals,
    }
    metadata: dict = {
        "truncation": args.truncation,
        "params": params,
        "couplings": system.coupling_names,
        "fixed_point": {k: float(v) for k, v in fp.location.items()},
        "newton_constant": form_factor.newton_constant(),
        "scale": args.scale,
        "xi_scale": args.xi_scale,
        "cos_theta": args.angle,
        "ir_ratio_to_gr": ir["ratio"],
        "ir_matches_gr": bool(ir["matches_gr"]),
        "uv_bounded": bool(uv["bounded"]),
    }

    if args.checks:
        from asymsafety.scattering import consistency as C

        report = C.consistency_report(amplitude, cos_theta=args.angle)
        metadata["consistency"] = {
            "unitarity_as_bounded": bool(report["unitarity"]["as_bounded"]),
            "unitarity_gr_unbounded": bool(report["unitarity"]["gr_unbounded"]),
            "uv_finite": bool(report["uv_finiteness"]["passed"]),
            "froissart_exponent": report["froissart"]["growth_exponent"],
            "no_ghost_pole": bool(report["causality"]["passed"]),
            "crossing": bool(report["crossing"]["passed"]),
            # Newtonian recovery is part of the verdict: a setup whose
            # trajectory never reaches the IR plateau must not report a
            # clean bill while ir_ratio_to_gr sits far from 1.
            "ir_newtonian_recovery": bool(ir["matches_gr"]),
            "all_passed": bool(report["all_passed"]) and bool(ir["matches_gr"]),
        }

    if args.compare_string:
        from asymsafety.scattering.bridge import ScatteringBridge

        verdict = ScatteringBridge(amplitude, cos_theta=args.angle).verify()
        metadata["bridge"] = {
            "as_physically_consistent": bool(
                verdict["as_physically_consistent"]
            ),
            "distinct_from_strings": bool(verdict["distinct_from_strings"]),
            "summary": verdict["summary"],
        }

    out = write_results(args.output, arrays, metadata)
    print(
        f"Sampled |M| over {n} energies for truncation {args.truncation!r} "
        f"(IR/GR ratio {ir['ratio']:.3f}, UV bounded={uv['bounded']}) → {out}"
    )
    if args.checks:
        c = metadata["consistency"]
        print(
            f"  consistency: all_passed={c['all_passed']} "
            f"(ir_newtonian_recovery={c['ir_newtonian_recovery']})"
        )
    if args.compare_string:
        print(f"  bridge: {metadata['bridge']['summary'][:80]}...")
    return 0
