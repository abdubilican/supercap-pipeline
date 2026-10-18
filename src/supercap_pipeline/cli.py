"""
cli.py
------
Command-line entry point:

    supercap-run  <folder>  [-o output_dir]  [--gcd-basis BASIS]

<folder> can be a single measurement folder (Output.txt + info.json) or any
parent of one -- a cell, cell-group or whole experiment folder. Every
measurement found below it is analyzed as CV or GCD based on its content.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .gcd_loader import VALID_BASES
from .pipeline import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="supercap-run",
        description="Supercapacitor CV/GCD analysis for one measurement or a whole experiment folder.",
    )
    parser.add_argument(
        "folder",
        help=(
            "A measurement folder containing Output.txt and info.json "
            "(e.g. '.../A7029/01_Cyclic Voltametry'), or any parent folder "
            "(cell / cell group / experiment) -- all measurements below it are run."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="DIR",
        help="Output directory for CSVs and PDFs (default: auto-derived from folder path).",
    )
    parser.add_argument(
        "--gcd-basis",
        choices=sorted(VALID_BASES),
        default=None,
        help=(
            "Mass normalization for GCD specific values. Required for GCD "
            "measurements unless info.json has gcdConfig.basis; overrides it if given."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f"folder not found: {folder}")

    run_experiment(folder, output_dir=args.output, gcd_basis=args.gcd_basis)


if __name__ == "__main__":
    main()
