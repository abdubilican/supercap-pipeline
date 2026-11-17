"""
loader.py
---------
Tester-file helpers shared by the GCD loader and the technique detection in
pipeline.py: info.json metadata access and the tab-separated Output.txt
parser.

The CV evaluation reads its inputs through cv_io.py (xlsx or Output.txt
with Excel-row mapping, masses from Input-Variables.txt).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def read_info_json(folder: Path) -> dict:
    """
    Load the raw `*info.json` metadata file from a measurement folder.

    Used by the GCD loader to read cell composition (electrodes, masses,
    electrolyte).
    """
    info_files = list(folder.glob("*info.json"))
    if not info_files:
        raise FileNotFoundError(f"No info.json found in {folder}")

    # utf-8-sig: tolerate a BOM, which Windows editors/PowerShell often add
    with open(info_files[0], encoding="utf-8-sig") as fh:
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


# ─────────────────────────────────────────────────────────────────────────────
# Output.txt loading
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are text/flags rather than numeric measurements.
TESTER_STR_COLUMNS: set[str] = {"type", "c/d", "p/g", "con", "I-Range", "over", "buf"}


def read_tester_txt(path: Path) -> pd.DataFrame:
    """
    Parse one tester Output.txt file (tab-separated, CRLF, row 2 = units).

    Strips column names and string-typed cell values, coerces every other
    column to numeric (invalid/blank -> NaN), sorts by time and resets the
    index.

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
