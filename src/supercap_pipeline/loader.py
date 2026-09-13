"""
loader.py
---------
Loads a cyclic voltammetry experiment folder and returns a unified DataFrame.

The Output.txt time-series is used directly.  Scan speed and voltage window
are detected from the data itself (|dV/dt| and max|V12|) so the analysis is
not tied to any particular Lua script step-numbering scheme.

Returned DataFrame columns
--------------------------
  t, cycle, step, type, V12, I12          -- raw from Output.txt
  scan_speed_mVs                          -- snapped to {1, 10, 100} mV/s
  voltage_window_V                        -- snapped to {1.0, 1.5, 2.0, 2.5} V
  I12_spec_mAg                            -- I12 / m_lightest  [mA/g]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STANDARD_SCAN_SPEEDS_MVS: list[float] = [1.0, 10.0, 100.0]  # mV/s
STANDARD_WINDOWS_V: list[float] = [1.0, 1.5, 2.0, 2.5]      # half-window (V_max)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment(cv_folder: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load one CV measurement folder.

    Parameters
    ----------
    cv_folder : path to the folder containing Output.txt and info.json

    Returns
    -------
    df   : time-series DataFrame (see module docstring for columns)
    meta : metadata dict with cell/electrode info
    """
    cv_folder = Path(cv_folder)
    meta = _load_metadata(cv_folder)
    df = _load_output_txt(cv_folder, meta)
    df = _assign_cv_parameters(df)
    log.info(
        "Loaded %d rows from %s; cycles: %s; scan speeds: %s mV/s; windows: %s V",
        len(df),
        cv_folder.name,
        sorted(df.loc[df["cycle"] > 0, "cycle"].unique()),
        sorted(df["scan_speed_mVs"].dropna().unique()),
        sorted(df["voltage_window_V"].dropna().unique()),
    )
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def read_info_json(folder: Path) -> dict:
    """
    Load the raw `*info.json` metadata file from a measurement folder.

    Shared by the CV loader (this module) and the GCD loader so both read
    cell composition (electrodes, masses, electrolyte) the same way.
    """
    info_files = list(folder.glob("*info.json"))
    if not info_files:
        raise FileNotFoundError(f"No info.json found in {folder}")

    with open(info_files[0], encoding="utf-8") as fh:
        return json.load(fh)


def composition_field(info: dict, key: str, subkey: str | None = "componentName"):
    """
    Look up one `testCell.composition.fields[...]` entry by its definition key.

    Returns the field's scalar value, or `value[subkey]` if the value is a
    nested component dict (e.g. Electrode1 -> {"componentName": "..."}).
    `subkey=None` returns the raw value (used for plain numbers like masses).
    """
    fields = info["testCell"]["composition"]["fields"]
    try:
        v = next(
            f["value"] for f in fields
            if f.get("definition", {}).get("key") == key
        )
    except StopIteration:
        return None
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get(subkey) if subkey is not None else v
    return v


def _load_metadata(folder: Path) -> dict:
    info = read_info_json(folder)

    cell_tag = info["testCell"]["cellTag"]
    exp_name = info["experiment"]["experimentName"]

    def _get(key: str, subkey: str = "componentName"):
        return composition_field(info, key, subkey)

    e1_name    = _get("Electrode1")
    e2_name    = _get("Electrode2")
    m1         = _get("Weight1", subkey=None)
    m2         = _get("Weight2", subkey=None)
    electrolyte = _get("Electrolyte")

    m1 = float(m1) if m1 is not None else None
    m2 = float(m2) if m2 is not None else None

    masses     = [m for m in [m1, m2] if m is not None and m > 0]
    m_lightest = min(masses) if masses else 1.0

    log.info(
        "Cell %s | e1=%s (%.4f g) | e2=%s (%.4f g) | lightest=%.4f g",
        cell_tag, e1_name, m1 or 0, e2_name, m2 or 0, m_lightest,
    )

    return dict(
        cell_tag    = cell_tag,
        exp_name    = exp_name,
        e1_name     = e1_name    or "?",
        e2_name     = e2_name    or "?",
        m1          = m1         or 1.0,
        m2          = m2,
        m_lightest  = m_lightest,
        electrolyte = electrolyte,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output.txt loading
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are text/flags rather than numeric measurements. Shared with
# gcd_loader.py so both techniques parse the tester's Output.txt identically.
TESTER_STR_COLUMNS: set[str] = {"type", "c/d", "p/g", "con", "I-Range", "over", "buf"}


def read_tester_txt(path: Path) -> pd.DataFrame:
    """
    Parse one tester Output.txt file (tab-separated, CRLF, row 2 = units).

    Shared by the CV loader (this module) and the GCD loader. Strips column
    names and string-typed cell values, coerces every other column to numeric
    (invalid/blank -> NaN), sorts by time and resets the index.

    Does NOT drop rows with NaN in any particular column -- callers decide
    which columns are required for their technique.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")

    df = pd.read_csv(path, sep="\t", skiprows=[1], low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if col in TESTER_STR_COLUMNS:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("t").reset_index(drop=True)
    return df


def _load_output_txt(folder: Path, meta: dict) -> pd.DataFrame:
    df = read_tester_txt(folder / "Output.txt")

    df = (
        df.dropna(subset=["t", "V12"])
          .sort_values("t")
          .reset_index(drop=True)
    )
    df["cycle"] = df["cycle"].fillna(0).astype(int)
    df["step"]  = df["step"].fillna(0).astype(int)

    # I12 in Output.txt is in mA
    df["I12_spec_mAg"] = df["I12"] / meta["m_lightest"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CV parameter detection
# ─────────────────────────────────────────────────────────────────────────────

def _assign_cv_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every cycle > 0 that contains VS (voltage sweep) data, detect:
      - scan_speed_mVs  from the median |dV/dt| of VS points
      - voltage_window_V from the 99th-percentile |V12|

    Both values are snapped to the nearest standard value.
    """
    df = df.copy()
    df["scan_speed_mVs"]  = np.nan
    df["voltage_window_V"] = np.nan

    vs_mask = df["type"] == "VS"

    for cyc in sorted(df.loc[df["cycle"] > 0, "cycle"].unique()):
        mask  = (df["cycle"] == cyc) & vs_mask
        sub   = df.loc[mask].sort_values("t")

        if len(sub) < 10:
            continue

        # ── scan speed from |dV/dt| ──────────────────────────────────────────
        dV   = sub["V12"].diff()
        dt   = sub["t"].diff()
        rate = (dV.abs() / dt).dropna()
        rate = rate[(rate > 5e-5) & (rate < 2.0)]   # filter noise & outliers
        if rate.empty:
            continue
        rate_mVs = rate.median() * 1000   # V/s → mV/s
        snap_speed = float(
            min(STANDARD_SCAN_SPEEDS_MVS, key=lambda s: abs(s - rate_mVs))
        )

        # ── voltage window from max |V12| ────────────────────────────────────
        v_max = sub["V12"].abs().quantile(0.99)
        snap_window = float(
            min(STANDARD_WINDOWS_V, key=lambda w: abs(w - v_max))
        )

        df.loc[df["cycle"] == cyc, "scan_speed_mVs"]  = snap_speed
        df.loc[df["cycle"] == cyc, "voltage_window_V"] = snap_window

    return df
