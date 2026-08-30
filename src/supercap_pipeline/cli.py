"""
cli.py
------
Command-line entry point:

    supercap-run  <cv_folder>  [-o output_dir]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="supercap-run",
        description="Supercapacitor CV analysis: CV plots + F/g vs scan rate.",
    )
    parser.add_argument(
        "cv_folder",
        help=(
            "Path to the measurement folder containing Output.txt and info.json "
            "(e.g. '.../A7029/01_Cyclic Voltametry')"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="DIR",
        help="Output directory for CSVs and PDFs (default: auto-derived from folder path).",
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

    run(cv_folder=args.cv_folder, output_dir=args.output)


if __name__ == "__main__":
    main()
