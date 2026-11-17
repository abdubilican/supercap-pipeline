"""
cli.py
------
Command-line entry point:

    supercap-run  <folder or export file>  [-o output_dir]

<folder> can be a single measurement folder (Output.txt or xlsx export plus
Input-Variables.txt) or any parent of one -- a cell, cell-group or whole
experiment folder. Every measurement found below it is evaluated as CV or
GCD, decided per file from its step types (VS -> CV, CC -> GCD).
Results: <output_dir>/<sample-id>/cv/ and .../gcd/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run_experiment

# accepted for backwards compatibility only (ignored, see pipeline.run_gcd)
LEGACY_GCD_BASES = ("cell", "single_electrode", "total_active_mass")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="supercap-run",
        description="Supercapacitor CV/GCD analysis for one measurement or a whole experiment folder.",
    )
    parser.add_argument(
        "folder",
        help=(
            "A measurement folder containing Output.txt or a PAT-Tester .xlsx export "
            "(e.g. '.../A7029/01_Cyclic Voltametry'), any parent folder "
            "(cell / cell group / experiment) -- all measurements below it are run -- "
            "or a single export file (.xlsx / Output.txt)."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="DIR",
        help=("Output root; results go to DIR/<sample-id>/cv and DIR/<sample-id>/gcd "
              "(default: an 'output' folder next to each measurement folder)."),
    )
    parser.add_argument(
        "--gcd-basis",
        choices=LEGACY_GCD_BASES,
        default=None,
        help="Ignored (kept for compatibility): the lab-guide GCD method fixes the mass basis.",
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
    if not folder.exists():
        parser.error(f"folder or file not found: {folder}")

    run_experiment(folder, output_dir=args.output, gcd_basis=args.gcd_basis)


if __name__ == "__main__":
    main()
