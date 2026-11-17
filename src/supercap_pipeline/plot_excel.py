"""
plot_excel.py
-------------
One Origin-ready Excel file per plot (lab guide GCD 6.3; also used for CV):

Sheet "Data", wide format: per curve one x column followed by its y column(s)
(extra y columns for a second axis or extra quantities), curves side by side.
    row 1 = long name   (e.g. "Voltage")
    row 2 = unit        (e.g. "V")
    row 3 = comment     (curve label, e.g. "0.1 A/g, cycle 4")
    row 4+ = data       (numbers as numbers; shorter curves leave empty cells)
Sheet "Meta": sample, source, plot file, masses, mass basis of every axis,
cycles used, number of curves, column roles (X/Y), date, pipeline version.
No merged cells, no formulas.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from openpyxl import Workbook

from .results_io import pipeline_version


@dataclass
class Column:
    """One data column: long name (row 1), unit (row 2), values; comment (row 3) optional."""
    name: str
    unit: str
    values: Sequence[float]
    comment: str = ""


@dataclass
class Curve:
    """One curve of a plot: its x column and one or more y columns."""
    label: str
    x: Column
    ys: list[Column] = field(default_factory=list)


def _cell(values: Sequence, r: int):
    if r >= len(values):
        return None
    v = values[r]
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def write_plot_excel(path: Path, curves: list[Curve], meta: dict) -> Path:
    """
    Write `curves` to <path> (sheet "Data", 3 header rows, data from row 4)
    and `meta` plus the curve layout to sheet "Meta". Returns the path.
    """
    cols: list[tuple[Column, str]] = []
    roles: list[str] = []
    for c in curves:
        for i, col in enumerate([c.x, *c.ys]):
            cols.append((col, col.comment or c.label))
            roles.append("X" if i == 0 else "Y")
    arrays = [list(col.values) for col, _ in cols]
    n_rows = max((len(a) for a in arrays), default=0)

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Data")
    ws.append([col.name for col, _ in cols])
    ws.append([col.unit for col, _ in cols])
    ws.append([comment for _, comment in cols])
    for r in range(n_rows):
        ws.append([_cell(a, r) for a in arrays])

    info = {
        **meta,
        "plot_file": meta.get("plot_file", path.with_suffix(".png").name),
        "curves": len(curves),
        "curve_labels": " | ".join(c.label for c in curves),
        "column_roles": " ".join(roles),
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": pipeline_version(),
    }
    wm = wb.create_sheet("Meta")
    wm.append(["key", "value"])
    for k, v in info.items():
        wm.append([k, v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
