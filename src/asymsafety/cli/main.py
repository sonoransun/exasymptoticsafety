"""``asymsafety`` console-script entry point.

Subcommands:

    asymsafety eval --truncation TRUNC --grid 'g:0:1:50' --output FILE
    asymsafety scan --truncation TRUNC --param-range n_scalars=1:8:8 \\
                    --output FILE
    asymsafety amplitude --truncation eh --s-range 1e-2:1e8:200 \\
                    --checks --compare-string --output FILE
"""

from __future__ import annotations

import argparse
import sys

from asymsafety.cli import amplitude_cmd, eval_cmd, scan_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asymsafety",
        description=(
            "Asymptotic-safety FRG toolkit: batch evaluation and parameter "
            "scans over beta-function truncations."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    eval_cmd.add_subparser(sub)
    scan_cmd.add_subparser(sub)
    amplitude_cmd.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
