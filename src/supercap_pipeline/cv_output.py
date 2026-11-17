"""
cv_output.py
------------
Results tables of the CV evaluation (lab guide 6.1): results.xlsx with the
sheets Results (one row per block and polarity), Summary (one row per
block), Meta (masses, provenance, tolerances) and Warnings (every warning
of the run), plus results.csv (= the Results sheet).

The Excel row numbers in Results point into the sheet "Output" of the raw
export (or the line numbers of Output.txt), so every Q0 can be recomputed
by hand from V_a, V_b, Qcd_a, Qcd_b.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv_config as cfg

SEGMENTS = {  # (discharge, charge) segment names per polarity
    ("symmetric", "+"): ("+U->0 (down sweep)", "0->+U (up sweep)"),
    ("symmetric", "-"): ("-U->0 (up sweep)", "0->-U (down sweep)"),
    ("one-sided", "+"): ("U->0 (down sweep)", "0->U (up sweep)"),
}


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


def _block_cols(b: dict, sample_id: str, source: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_file": source,
        "block": b["block"],
        "occurrence": b["occurrence"],
        "window_type": b["window_type"],
        "U_max_V": b["U_max"],
        "nu_nom_mVs": b["nu_nom_mVs"],
        "nu_meas_mVs": round(b["nu_meas_Vs"] * 1000.0, 4),
        "n_cycles": b["n_cycles"],
        "n_complete_cycles": b["n_complete"],
        "up_cycle": b["up"]["cycle"], "up_step": b["up"]["step"],
        "down_cycle": b["down"]["cycle"], "down_step": b["down"]["step"],
    }


def _sweep_cols(prefix: str, s: dict) -> dict:
    return {
        f"{prefix}_rows": f"{s['row_start']}-{s['row_end']}",
        f"{prefix}_row_a": s["row_a"], f"{prefix}_row_b": s["row_b"],
        f"{prefix}_V_a": s["V_a"], f"{prefix}_V_b": s["V_b"],
        f"{prefix}_Qcd_a_mAh": s["Qcd_a"], f"{prefix}_Qcd_b_mAh": s["Qcd_b"],
        f"{prefix}_Qcd_first_mAh": s["Qcd_first"],
        f"{prefix}_Q0_mAh": s["Q0"], f"{prefix}_Q_end_mAh": s["Q_end"],
    }


def results_table(blocks: list[dict], sample_id: str, source: str) -> pd.DataFrame:
    """Sheet "Results": one row per block and polarity."""
    rows = []
    for b in blocks:
        for p, d in b["pol"].items():
            seg_dis, seg_ch = SEGMENTS[(b["window_type"], p)]
            rows.append({
                **_block_cols(b, sample_id, source),
                "polarity": p,
                **_sweep_cols("up", b["up"]),
                **_sweep_cols("down", b["down"]),
                "discharge_segment": seg_dis,
                "charge_segment": seg_ch,
                "Q_dis_mAh": d["Q_dis"],
                "Q_ch_mAh": d["Q_ch"],
                "C_F": d["C"],
                "C_cell_sp_Fg": d["C_cell_sp"],
                "C_el_Fg": d["C_el"],
                "C_el_exact_Fg": d["C_el_exact"],
                "eta": d["eta"],
                "C_est_F": d.get("C_est", np.nan),
                "flags": "; ".join(b["flags"] + d["flags"]),
            })
    return pd.DataFrame(rows)


def summary_table(blocks: list[dict], sample_id: str, source: str) -> pd.DataFrame:
    """Sheet "Summary": one row per block."""
    rows = []
    for b in blocks:
        pol = b["pol"]
        flags = list(b["flags"]) + [f"{p}: {f}" for p, d in pol.items() for f in d["flags"]]
        rows.append({
            **_block_cols(b, sample_id, source),
            "C_plus_F": pol["+"]["C"],
            "C_minus_F": pol["-"]["C"] if "-" in pol else np.nan,
            "C_F": b["C"],
            "C_cell_sp_Fg": b["C_cell_sp"],
            "C_el_Fg": b["C_el"],
            "C_el_exact_Fg": b["C_el_exact"],
            "eta_plus": pol["+"]["eta"],
            "eta_minus": pol["-"]["eta"] if "-" in pol else np.nan,
            "polarity_dev_pct": b["pol_dev"] * 100.0,
            "C_prev_cycle_F": b["C_prev"] if b["C_prev"] is not None else np.nan,
            "stability": b["stability_text"],
            "reported_for_window": b["reported_for_window"],
            "flags": "; ".join(flags),
        })
    return pd.DataFrame(rows)


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


def meta_table(masses: dict, sample_id: str, source: str, n_dropped: int) -> pd.DataFrame:
    """Sheet "Meta": masses, provenance and every tolerance used."""
    items = [
        ("sample_id", sample_id),
        ("source_file", source),
        ("m1_g", masses["m1_g"]),
        ("m2_g", masses["m2_g"]),
        ("m_mean_g", masses["m_mean_g"]),
        ("m1_as_read", masses.get("m1_raw", "")),
        ("m2_as_read", masses.get("m2_raw", "")),
        ("active_fraction_applied", masses.get("active_fraction", "none")),
        ("mass_source_file", masses["source"]),
        ("evaluation_date", _dt.datetime.now().isoformat(timespec="seconds")),
        ("pipeline_version", pipeline_version()),
        ("transition_sweeps_dropped", n_dropped),
        ("method", "lab guide 'Specific Capacitance from CV Data (PAT-Tester)': "
                   "C = 3.6*Q_dis/U_max, C_cell,sp = C/(m1+m2), C_el = 4C/(m1+m2)"),
    ]
    items += [(f"tol.{k}", v) for k, v in cfg.tolerances().items()]
    return pd.DataFrame(items, columns=["key", "value"]).astype({"value": str})


def write_results(out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame,
                  meta: pd.DataFrame, warnings: list[str]) -> list[Path]:
    """Write results.xlsx (Results, Summary, Meta, Warnings) and results.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "results.xlsx"
    csv = out_dir / "results.csv"
    with pd.ExcelWriter(xlsx) as xw:
        results.to_excel(xw, sheet_name="Results", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        meta.to_excel(xw, sheet_name="Meta", index=False)
        pd.DataFrame({"warning": warnings or ["(none)"]}).to_excel(
            xw, sheet_name="Warnings", index=False)
    write_csv(results, csv)
    return [xlsx, csv]
