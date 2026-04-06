#!/usr/bin/env python3
"""Generate all documentation figures for the Asymptotic Safety Explorer.

Usage:
    PYTHONPATH=src python3 scripts/generate_figures.py
    PYTHONPATH=src python3 scripts/generate_figures.py --output-dir docs/images --format png

The generated images are committed to docs/images/ so that the README
renders on GitHub without running this script.  Re-run whenever the
visualization code changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Headless backend — must be set before any pyplot import.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ok: list[str] = []
_fail: list[tuple[str, str]] = []


def _save(fig: plt.Figure, name: str, out_dir: Path, fmt: str) -> None:
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    _ok.append(str(path))


def _run(label: str, fn, name: str, out_dir: Path, fmt: str) -> None:
    t0 = time.time()
    try:
        fig = fn()
        _save(fig, name, out_dir, fmt)
        elapsed = time.time() - t0
        print(f"  OK  {name}.{fmt}  ({elapsed:.1f}s)")
    except Exception as exc:
        elapsed = time.time() - t0
        _fail.append((name, str(exc)))
        print(f"  FAIL {name}: {exc}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def gen_asymptotic_safety_concept(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import asymptotic_safety_concept
    _run("Asymptotic safety concept", asymptotic_safety_concept,
         "asymptotic_safety_concept", out, fmt)


def gen_wetterich_equation(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import wetterich_equation_diagram
    _run("Wetterich equation", wetterich_equation_diagram,
         "wetterich_equation", out, fmt)


def gen_regulator_comparison(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import regulator_comparison
    _run("Regulator comparison", regulator_comparison,
         "regulator_comparison", out, fmt)


def gen_fp_stability_concept(out: Path, fmt: str) -> None:
    from asymsafety.visualization.conceptual import fixed_point_stability_concept
    _run("FP stability concept", fixed_point_stability_concept,
         "fp_stability_concept", out, fmt)


def gen_cross_analogue_bridge(out: Path, fmt: str) -> None:
    from asymsafety.visualization.bridge_diagram import cross_analogue_bridge_diagram
    _run("Cross-analogue bridge", cross_analogue_bridge_diagram,
         "cross_analogue_bridge", out, fmt)


def gen_hydraulic_analogy(out: Path, fmt: str) -> None:
    from asymsafety.visualization.bridge_diagram import hydraulic_analogy_diagram
    _run("Hydraulic analogy", hydraulic_analogy_diagram,
         "hydraulic_analogy", out, fmt)


def gen_eh_phase_portrait(out: Path, fmt: str) -> None:
    from asymsafety.visualization.phase_portrait import annotated_eh_phase_portrait
    _run("EH phase portrait", annotated_eh_phase_portrait,
         "eh_phase_portrait", out, fmt)


def gen_running_couplings(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.visualization.phase_portrait import flow_diagram

        system = build_eh_beta_system(d=4)
        integrator = FlowIntegrator(system)
        trajs = []
        for ic in [
            {"g": 0.5, "lambda": 0.1},
            {"g": 0.8, "lambda": 0.05},
            {"g": 0.3, "lambda": 0.2},
        ]:
            try:
                trajs.append(integrator.integrate(ic, t_span=(-8, 8)))
            except Exception:
                pass
        return flow_diagram(trajs, title="Running Couplings ($d=4$, Einstein–Hilbert)")
    _run("Running couplings", _make, "running_couplings", out, fmt)


def gen_3d_flow(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.analysis.fixed_points import FixedPointFinder
        from asymsafety.analysis.flow import FlowIntegrator
        from asymsafety.gui.visualization_3d import flow_trajectories_3d

        system = build_eh_beta_system(d=4)
        finder = FixedPointFinder(system)
        fps = []
        for guess in [{"g": 0.01, "lambda": 0.01}, {"g": 0.7, "lambda": 0.14}]:
            try:
                fps.append(finder.find_fixed_point(guess))
            except Exception:
                pass

        integrator = FlowIntegrator(system)
        trajs = []
        for ic in [
            {"g": 0.3, "lambda": 0.05},
            {"g": 0.8, "lambda": 0.1},
            {"g": 0.5, "lambda": -0.1},
            {"g": 1.0, "lambda": 0.2},
        ]:
            try:
                trajs.append(integrator.integrate(ic, t_span=(-5, 5)))
            except Exception:
                pass

        # For 2-coupling EH, use "g" twice for 3rd axis (will be zero-padded)
        return flow_trajectories_3d(
            trajs, "g", "lambda", "g",
            fixed_points=fps if fps else None,
            show_eigenvectors=False,
        )
    _run("3D flow", _make, "3d_flow", out, fmt)


def gen_hydraulic_network(out: Path, fmt: str) -> None:
    def _make():
        from asymsafety.beta.einstein_hilbert import build_eh_beta_system
        from asymsafety.hydraulic.mapping import RGToHydraulicMapper
        from asymsafety.hydraulic.visualization import plot_hydraulic_network

        system = build_eh_beta_system(d=4)
        mapper = RGToHydraulicMapper(system)
        network = mapper.build_network(
            reference_point={"g": 0.7, "lambda": 0.14},
        )
        return plot_hydraulic_network(network)
    _run("Hydraulic network", _make, "hydraulic_network", out, fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    gen_asymptotic_safety_concept,
    gen_wetterich_equation,
    gen_regulator_comparison,
    gen_fp_stability_concept,
    gen_cross_analogue_bridge,
    gen_hydraulic_analogy,
    gen_eh_phase_portrait,
    gen_running_couplings,
    gen_3d_flow,
    gen_hydraulic_network,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="docs/images",
                        help="Directory for generated images (default: docs/images)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output image format (default: png)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply project style
    from asymsafety.visualization.style import apply_style
    apply_style()

    print(f"Generating figures → {out_dir}/ ({args.format})\n")
    t0 = time.time()

    for gen_fn in ALL_GENERATORS:
        gen_fn(out_dir, args.format)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {len(_ok)} succeeded, {len(_fail)} failed.")
    if _fail:
        print("\nFailed figures:")
        for name, err in _fail:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
