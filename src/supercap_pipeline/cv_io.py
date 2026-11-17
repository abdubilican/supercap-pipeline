"""
cv_io.py
--------
Input side of the CV evaluation (lab guide section 1).

read_export()  reads a PAT-Tester (EL-Cell) export -- the .xlsx with sheet
               "Output", or the tab-separated Output.txt of the same layout --
               and keeps the spreadsheet row number of every data row in the
               column `excel_row`, so every result can be checked by hand.
read_masses()  reads m1 and m2 from the Input-Variables.txt of the measurement.

Both files are only read, never written.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv_config as cfg

log = logging.getLogger("supercap_pipeline.cv").getChild("io")

# Columns the CV evaluation reads (lab guide 1.1). |Qstep|, |Qcd|, Qc and Qd
# are deliberately absent: they lose the sign or are not split at 0 V.
NUMERIC_COLUMNS = ("t", "t_step", "cycle", "step", "V12", "I12", "Qcd")
STRING_COLUMNS = ("type", "c/d")
REQUIRED_COLUMNS = NUMERIC_COLUMNS + STRING_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Measurement export
# ─────────────────────────────────────────────────────────────────────────────

def read_export(path: str | Path) -> pd.DataFrame:
    """
    Read one PAT-Tester export into a DataFrame with the columns
    t, t_step, cycle, step, type, c/d, V12, I12, Qcd and excel_row.

    Layout (both .xlsx sheet "Output" and Output.txt): row 1 = column names,
    row 2 = units, data from row 3, so  excel_row = pandas index + 3  of the
    unfiltered table. For Output.txt the "Excel row" is the line number.
    Column names and string cells are stripped of the trailing spaces the
    tester writes ("VS ", "c "); columns are accessed by name only.
    Rows without t or V12 are dropped afterwards; excel_row is kept.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    if path.suffix.lower() in (".xlsx", ".xlsm"):
        raw = _read_xlsx(path)
    else:
        raw = _read_txt(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: required column(s) missing: {missing}")

    df = raw[list(REQUIRED_COLUMNS)].copy()
    df.insert(0, "excel_row", np.arange(len(df), dtype=np.int64) + cfg.EXCEL_FIRST_DATA_ROW)

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip().fillna("")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_raw = len(df)
    df = df.dropna(subset=["t", "V12"]).reset_index(drop=True)
    df["cycle"] = df["cycle"].fillna(-1).astype(int)
    df["step"] = df["step"].fillna(-1).astype(int)
    log.info("Read %s: %d data rows (%d without t/V12 ignored)",
             path.name, len(df), n_raw - len(df))
    return df


def _wanted(name) -> bool:
    return str(name).strip() in REQUIRED_COLUMNS


def _read_txt(path: Path) -> pd.DataFrame:
    """Tab-separated Output.txt; blank lines kept so line numbers stay exact."""
    df = pd.read_csv(
        path, sep="\t", header=0, skiprows=[1], usecols=_wanted,
        skip_blank_lines=False, dtype=str, encoding_errors="replace",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_xlsx(path: Path) -> pd.DataFrame:
    """
    Sheet "Output" of the xlsx export, read in one pass by calamine (fast,
    Rust) or, if python-calamine is not installed, by openpyxl in read-only
    mode. Empty rows are kept so list position = Excel row - 1.
    """
    if importlib.util.find_spec("python_calamine") is not None:
        from python_calamine import CalamineWorkbook

        wb = CalamineWorkbook.from_path(str(path))
        if cfg.OUTPUT_SHEET not in wb.sheet_names:
            raise ValueError(f"{path.name}: no sheet named '{cfg.OUTPUT_SHEET}' "
                             f"(sheets: {wb.sheet_names})")
        rows = wb.get_sheet_by_name(cfg.OUTPUT_SHEET).to_python(skip_empty_area=False)
        engine = "calamine"
    else:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if cfg.OUTPUT_SHEET not in wb.sheetnames:
                raise ValueError(f"{path.name}: no sheet named '{cfg.OUTPUT_SHEET}' "
                                 f"(sheets: {wb.sheetnames})")
            rows = list(wb[cfg.OUTPUT_SHEET].iter_rows(min_row=1, values_only=True))
        finally:
            wb.close()
        engine = "openpyxl"
    log.info("Reading %s with %s", path.name, engine)

    if len(rows) < 2:
        raise ValueError(f"{path.name}: sheet '{cfg.OUTPUT_SHEET}' has no data rows")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    keep = [i for i, h in enumerate(header) if h in REQUIRED_COLUMNS]
    data = [[r[i] if i < len(r) else None for i in keep] for r in rows[2:]]
    return pd.DataFrame(data, columns=[header[i] for i in keep], dtype=object)


def is_cv_export(path: Path) -> bool:
    """True if `path` is an .xlsx with a sheet "Output" (cheap: sheet names only)."""
    if path.suffix.lower() not in (".xlsx", ".xlsm") or path.name.startswith("~$"):
        return False
    try:
        if importlib.util.find_spec("python_calamine") is not None:
            from python_calamine import CalamineWorkbook
            return cfg.OUTPUT_SHEET in CalamineWorkbook.from_path(str(path)).sheet_names
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            return cfg.OUTPUT_SHEET in wb.sheetnames
        finally:
            wb.close()
    except Exception:  # unreadable / not a workbook
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Masses
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_RE = re.compile(r"[\[(]?\s*(mg|g)\s*[\])]?\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?$")
_ACTIVE_FRACTION_KEYS = ("active fraction", "active_fraction", "active material fraction")


def find_mass_file(export_path: Path) -> Path:
    """
    The Input-Variables.txt belonging to an export: in the same folder; if
    there are several, the one whose name starts with the export's stem.
    """
    folder = export_path.parent
    candidates = sorted(folder.glob(cfg.MASS_FILE_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"No Input-Variables.txt next to {export_path.name} in {folder} -- "
            "masses m1/m2 are required, no default is used.")
    if len(candidates) == 1:
        return candidates[0]
    matching = [p for p in candidates if p.name.startswith(export_path.stem)]
    if len(matching) == 1:
        return matching[0]
    raise FileNotFoundError(
        f"Several Input-Variables files in {folder} and none clearly belongs to "
        f"{export_path.name}: {[p.name for p in candidates]}")


def read_masses(path: str | Path) -> dict:
    """
    Read the active masses m1, m2 of electrode 1 and 2 from Input-Variables.txt.

    Format written by the EL-Cell software: tab-separated, header row with
    columns "Key" and "Value", a units row, then one row per variable
    (e.g. "m1 | 0.00138"). Plain "key value", "key = value" or "key: value"
    lines are accepted too. If a key appears several times, the last entry
    wins (logged). Unit: taken from the units row, the key ("m1 [mg]") or the
    value ("1.38 mg") if given; otherwise g is assumed (logged). Values are
    converted to g and must lie in [MASS_MIN_G, MASS_MAX_G].
    m_mean = (m1 + m2) / 2.

    Raises FileNotFoundError / ValueError if the file or a mass is missing --
    there is never a default mass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mass file not found: {path}")

    entries = _parse_key_values(path)
    if not entries:
        raise ValueError(f"{path.name}: no key/value entries found")

    result: dict = {"source": str(path)}
    for name in ("m1", "m2"):
        hits = [e for e in entries if e["key"].lower() == name]
        if not hits:
            raise ValueError(f"{path.name}: mass '{name}' missing -- evaluation stopped "
                             "(no default mass is used).")
        if len({h["raw"] for h in hits}) > 1:
            log.warning("%s: '%s' given %d times with different values %s; using the last (%s)",
                        path.name, name, len(hits), [h["raw"] for h in hits], hits[-1]["raw"])
        e = hits[-1]
        value = _to_float(e["raw"])
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{path.name}: mass '{name}' = {e['raw']!r} is not a positive number")
        unit = e["unit"]
        if unit is None:
            unit = "g"
            log.info("%s: no unit given for %s; assuming g (EL-Cell convention)", path.name, name)
        grams = value / 1000.0 if unit.lower() == "mg" else value
        if not (cfg.MASS_MIN_G <= grams <= cfg.MASS_MAX_G):
            raise ValueError(
                f"{path.name}: {name} = {value} {unit} = {grams:g} g is outside the plausible "
                f"range {cfg.MASS_MIN_G:g}-{cfg.MASS_MAX_G:g} g -- check the unit.")
        result[f"{name}_g"] = grams
        result[f"{name}_raw"] = f"{e['raw']} {unit}"

    other = sorted({e["key"] for e in entries} - {"m1", "m2"})
    if other:
        log.info("%s: other variables present and not used: %s", path.name, other)

    fraction = [e for e in entries if e["key"].lower() in _ACTIVE_FRACTION_KEYS]
    if fraction:
        f = _to_float(fraction[-1]["raw"])
        if f is None or not (0 < f <= 1):
            raise ValueError(f"{path.name}: active fraction {fraction[-1]['raw']!r} not in (0, 1]")
        result["m1_g"] *= f
        result["m2_g"] *= f
        result["active_fraction"] = f
        log.warning("%s: explicit active-material fraction %g applied to m1 and m2", path.name, f)

    result["m_mean_g"] = (result["m1_g"] + result["m2_g"]) / 2.0
    log.info("Masses from %s: m1 = %.6g g, m2 = %.6g g, m_mean = %.6g g",
             path.name, result["m1_g"], result["m2_g"], result["m_mean_g"])
    return result


def _to_float(s: str) -> float | None:
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _split_unit(text: str) -> tuple[str, str | None]:
    """'m1 [mg]' -> ('m1', 'mg');  '1.38 mg' -> ('1.38', 'mg');  'm1' -> ('m1', None)."""
    m = _UNIT_RE.search(text)
    if m and m.start() > 0:
        return text[:m.start()].strip(), m.group(1).lower()
    return text.strip(), None


def _parse_key_values(path: Path) -> list[dict]:
    """Return [{key, raw, unit}] in file order."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    rows = [[c.strip() for c in ln.split("\t")] for ln in lines if ln.strip()]
    entries: list[dict] = []

    header_idx = next(
        (i for i, r in enumerate(rows)
         if any(c.lower() == "key" for c in r) and any(c.lower() == "value" for c in r)),
        None,
    )
    if header_idx is not None:
        hdr = [c.lower() for c in rows[header_idx]]
        k_col, v_col = hdr.index("key"), hdr.index("value")
        body = rows[header_idx + 1:]
        column_unit = None
        if body and (len(body[0]) <= k_col or not body[0][k_col]):
            # units row: empty Key cell; may carry a unit for the Value column
            u = body[0][v_col].strip().lower() if len(body[0]) > v_col else ""
            column_unit = u if u in ("mg", "g") else None
            body = body[1:]
        for r in body:
            if len(r) <= max(k_col, v_col) or not r[k_col]:
                continue
            key, key_unit = _split_unit(r[k_col])
            raw, val_unit = _split_unit(r[v_col])
            entries.append({"key": key, "raw": raw, "unit": val_unit or key_unit or column_unit})
        return entries

    for ln in lines:
        m = re.match(r"^\s*([^=:\t]+?)\s*[=:\t]\s*(.+?)\s*$", ln) or \
            re.match(r"^\s*(\S+(?:\s*[\[(]\s*m?g\s*[\])])?)\s+(.+?)\s*$", ln)
        if not m:
            continue
        key, key_unit = _split_unit(m.group(1))
        raw, val_unit = _split_unit(m.group(2))
        if _NUMBER_RE.match(raw.replace(" ", "")):
            entries.append({"key": key, "raw": raw, "unit": val_unit or key_unit})
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Sample ID
# ─────────────────────────────────────────────────────────────────────────────

def sample_id_for(export_path: Path) -> str:
    """xlsx: file name without extension. Output.txt: cellTag from info.json, else folder name."""
    if export_path.suffix.lower() in (".xlsx", ".xlsm"):
        return export_path.stem
    for info in sorted(export_path.parent.glob("*info.json")):
        try:
            with open(info, encoding="utf-8-sig") as fh:
                tag = json.load(fh).get("testCell", {}).get("cellTag")
            if tag:
                return str(tag).strip()
        except (OSError, ValueError):
            pass
    return export_path.parent.name
