"""
results_io.py
-------------
Writers shared by the CV and GCD outputs: CSV in the pipeline's format,
results workbooks, and the provenance string (version + git commit).
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


def _plain_number(x: float) -> str:
    """Positional notation (never 1e-05) with a decimal point, e.g. -0.00001."""
    return np.format_float_positional(x, precision=10, unique=True, trim="-")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    CSV with ';' as separator, decimal point, no exponent notation, UTF-8
    with BOM (so '±' is shown correctly in Excel). Read back with
    pd.read_csv(sep=';').
    """
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig", float_format=_plain_number)
    return path


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """One xlsx with one sheet per entry (numbers as numbers, no formulas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    return path


def pipeline_version() -> str:
    """Package version plus the git commit of the source tree (if available)."""
    try:
        ver = metadata.version("supercap-pipeline")
    except metadata.PackageNotFoundError:
        ver = "unknown"
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            ver += f" ({commit}{'+dirty' if dirty else ''})"
    except (OSError, subprocess.SubprocessError):
        pass
    return ver
