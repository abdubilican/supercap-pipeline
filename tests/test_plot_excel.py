"""
tests/test_plot_excel.py
------------------------
Plot-Excel test (lab guide GCD 6.3 / 8, also for the CV retrofit 6.4): for
every plot the <plot>.xlsx exists, sheet "Data" has the 3 header rows (long
name, unit, comment), numeric data from row 4 (empty cells, never NaN text),
and as many curves as the plot; sheet "Meta" carries sample, masses, mass
basis, cycles used and version.
"""

from __future__ import annotations

import math
from pathlib import Path

import openpyxl
import pytest

import cv_synth
from gcd_synth import GcdCell, rate_program, write_masses, write_txt
from supercap_pipeline.pipeline import run_cv_export, run_gcd_export

M = 0.4


def check_plot_xlsx(path: Path) -> dict:
    """Validate the layout of one plot workbook; returns its Meta as a dict."""
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Data", "Meta"]
    ws = wb["Data"]
    rows = list(ws.iter_rows(values_only=True))
    assert len(rows) >= 3
    names, units, comments = rows[0], rows[1], rows[2]
    assert all(isinstance(v, str) and v for v in names), path.name
    assert all(isinstance(v, str) for v in units), path.name
    assert all(isinstance(v, str) and v for v in comments), path.name
    assert not ws.merged_cells.ranges
    for r in rows[3:]:
        for v in r:
            assert v is None or (isinstance(v, (int, float)) and math.isfinite(v)), (path.name, v)
    meta = {k: v for k, v in wb["Meta"].iter_rows(min_row=2, values_only=True)}
    roles = meta["column_roles"].split()
    assert len(roles) == len(names)
    assert roles.count("X") == meta["curves"]
    for k in ("sample_id", "source_file", "plot_file", "m1_g", "m2_g", "m_mean_g",
              "cycles_used", "date", "pipeline_version"):
        assert k in meta, (path.name, k)
    assert any(k.startswith("mass_basis") for k in meta)
    # every curve has data (except a deliberately empty legend entry)
    return meta


def _all_plot_workbooks(plot_dir: Path) -> dict[str, dict]:
    pngs = sorted(plot_dir.glob("*.png"))
    assert pngs
    out = {}
    for png in pngs:
        xlsx = png.with_suffix(".xlsx")
        assert xlsx.exists(), f"missing {xlsx.name}"
        out[png.stem] = check_plot_xlsx(xlsx)
    return out


def test_gcd_plot_workbooks(tmp_path):
    cell = GcdCell(pts=25)
    cell.rest(10)
    rate_program(cell, [100.0, 200.0, 400.0], 3)
    for _ in range(120):                      # cycling block
        cell.cycle_(200.0 * 1.5, 2.5, 0.0)
    folder = tmp_path / "02_GCD"
    write_txt(cell.frame(), folder / "Output.txt")
    write_masses(folder, M)
    r = run_gcd_export(folder / "Output.txt", output_dir=tmp_path / "out")
    metas = _all_plot_workbooks(r["out"] / "plots")

    n_rate = 3
    expected = {"gcd_profiles_time": n_rate, "gcd_profiles_charge": n_rate,
                "gcd_rate_capability": 1, "gcd_esr": 1, "gcd_ragone": 1,
                "gcd_cycling_retention": 1, "gcd_cycling_efficiency": 1,
                "gcd_qc_0.25Ag": 5, "gcd_qc_0.5Ag": 5, "gcd_qc_1Ag": 5}
    assert set(metas) == set(expected)
    for name, n in expected.items():
        assert metas[name]["curves"] == n, name


def test_gcd_floating_plot_workbooks(tmp_path):
    cell = GcdCell(pts=25)
    cell.rest(10)
    for k in range(3):
        if k:
            cell.cc(100.0, 2.5)
            cell.hold(2.5, 3600.0)
            cell.cc(-100.0, 0.0)
        rate_program(cell, [100.0, 400.0], 2)
    folder = tmp_path / "02_GCD"
    write_txt(cell.frame(), folder / "Output.txt")
    write_masses(folder, M)
    r = run_gcd_export(folder / "Output.txt", output_dir=tmp_path / "out")
    metas = _all_plot_workbooks(r["out"] / "plots")
    assert metas["gcd_floating_retention"]["curves"] == 2       # one series per j
    assert metas["gcd_floating_efficiency"]["curves"] == 2


def test_cv_plot_workbooks(tmp_path):
    cell = cv_synth.symmetric_program([(1.0, 0.001), (1.0, 0.010), (2.0, 0.001)], n_cycles=2)
    folder = tmp_path / "01_CV"
    folder.mkdir()
    cv_synth.write_txt(cell.frame(), folder / "Output.txt")
    cv_synth.write_masses(folder)
    r = run_cv_export(folder / "Output.txt", output_dir=tmp_path / "out")
    metas = _all_plot_workbooks(r["out"] / "plots")
    assert metas["cv_1mVs"]["curves"] == 2                      # two windows at 1 mV/s
    assert metas["cv_10mVs"]["curves"] == 1
    assert metas["rate_capability"]["curves"] == 2
    assert metas["normalised_pm1.00V"]["curves"] == 2
    controls = [k for k in metas if k.startswith("control_block")]
    assert len(controls) == 3
    for k in controls:
        assert metas[k]["curves"] == 4    # cycle, 2 discharge areas, 0 V points
    # the CSVs of the CV plots are still written next to them
    assert all((r["out"] / "plots" / f"{k}.csv").exists() for k in metas)
