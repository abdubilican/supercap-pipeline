"""
export_io.py
------------
Input side shared by the CV and the GCD evaluation (lab guides, section 1).

read_export()  reads a PAT-Tester (EL-Cell) export -- the .xlsx with sheet
               "Output", or the tab-separated Output.txt of the same layout --
               and keeps the spreadsheet row number of every data row in the
               column `excel_row`, so every result can be checked by hand.
               Which columns are read is given by a ColumnSet (CV_COLUMNS,
               GCD_COLUMNS, TYPE_COLUMNS).
read_masses()  reads m1 and m2 from the Input-Variables.txt of the measurement.

Both files are only read, never written.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("supercap_pipeline.io")

# ── Export layout and mass file (shared; listed in the Meta sheets) ──────────
OUTPUT_SHEET = "Output"            # sheet holding the raw data in the xlsx export
EXCEL_FIRST_DATA_ROW = 3           # row 1 = names, row 2 = units -> data from row 3
MASS_FILE_PATTERN = "*Input-Variables*.txt"
MASS_MIN_G = 1e-4                  # plausibility range of one electrode mass [g];
MASS_MAX_G = 1.0                   # outside it the unit is probably not g -> error


@dataclass(frozen=True)
class ColumnSet:
    """
    Columns to read: numeric, string, the subset that may be absent, and the
    columns whose missing values drop a row (default t and V12).
    """
    numeric: tuple[str, ...]
    strings: tuple[str, ...]
    optional: tuple[str, ...] = ()
    dropna_on: tuple[str, ...] = ("t", "V12")

    @property
    def all(self) -> tuple[str, ...]:
        return self.numeric + self.strings


# CV (lab guide CV 1.1). |Qstep|, |Qcd|, Qc and Qd are deliberately absent:
# they lose the sign or are not split at 0 V.
CV_COLUMNS = ColumnSet(("t", "t_step", "cycle", "step", "V12", "I12", "Qcd"), ("type", "c/d"))
# GCD (lab guide GCD 1.1): charge from Qcd, energy from W12; never |Qstep|,
# |Qcd|, Qc, Qd or P. `con` is only used to cross-check voltage holds.
GCD_COLUMNS = ColumnSet(("t", "t_step", "cycle", "step", "V12", "I12", "Qcd", "W12"),
                        ("type", "c/d", "con"), optional=("con",))
# Only the step type: used to decide CV vs GCD per file.
TYPE_COLUMNS = ColumnSet((), ("type",))
# PEIS (lab guide PEIS 1.1): Output-Impedance.txt. The half-cell impedance
# columns are optional (NaN in two-electrode cells); no row is dropped here --
# the PEIS module drops the NaN end row of every spectrum itself and logs it.
IMPEDANCE_FILE = "Output-Impedance.txt"
HALF_CELL_COLUMNS = ("Re(Z1)", "Im(Z1)", "Re(Z2)", "Im(Z2)", "Re(ZR)", "Im(ZR)",
                     "Re(Z1R)", "Im(Z1R)", "Re(Z2R)", "Im(Z2R)")
PEIS_COLUMNS = ColumnSet(
    ("freq", "t", "step", "V12", "dV12", "I12", "dI12", "Re(Z12)", "Im(Z12)", "|Z12|",
     "phi(Z12)", "SFDR(Z12)", "Cs", "dt") + HALF_CELL_COLUMNS,
    ("type",), optional=("dI12", "dt") + HALF_CELL_COLUMNS, dropna_on=())


# ─────────────────────────────────────────────────────────────────────────────
# Measurement export
# ─────────────────────────────────────────────────────────────────────────────

def read_export(path: str | Path, columns: ColumnSet = CV_COLUMNS) -> pd.DataFrame:
    """
    Read one PAT-Tester export into a DataFrame with the columns of
    `columns` (default: the CV set t, t_step, cycle, step, type, c/d, V12,
    I12, Qcd) plus excel_row.

    Layout (both .xlsx sheet "Output" and Output.txt): row 1 = column names,
    row 2 = units, data from row 3, so  excel_row = pandas index + 3  of the
    unfiltered table. For Output.txt the "Excel row" is the line number.
    Column names and string cells are stripped of the trailing spaces the
    tester writes ("VS ", "CC ", "c "); columns are accessed by name only.
    Rows without t or V12 are dropped afterwards; excel_row is kept.
    Optional columns that are absent are filled with "" (strings) or NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    wanted = set(columns.all)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        raw = _read_xlsx(path, wanted)
    else:
        raw = _read_txt(path, wanted)

    missing = [c for c in columns.all if c not in raw.columns and c not in columns.optional]
    if missing:
        raise ValueError(f"{path.name}: required column(s) missing: {missing}")
    for c in columns.optional:
        if c not in raw.columns:
            raw[c] = "" if c in columns.strings else np.nan
            log.info("%s: optional column '%s' not present", path.name, c)

    df = raw[list(columns.all)].copy()
    df.insert(0, "excel_row", np.arange(len(df), dtype=np.int64) + EXCEL_FIRST_DATA_ROW)

    for col in columns.strings:
        df[col] = df[col].astype("string").str.strip().fillna("")
    for col in columns.numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_raw = len(df)
    key = [c for c in columns.dropna_on if c in df.columns]
    if key:
        df = df.dropna(subset=key).reset_index(drop=True)
    for c in ("cycle", "step"):
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype(int)
    log.info("Read %s: %d data rows (%d without %s ignored)",
             path.name, len(df), n_raw - len(df), "/".join(key) or "-")
    return df


def step_types(path: str | Path) -> set[str]:
    """The set of values in the `type` column, e.g. {"Rest", "VS"} or {"Rest", "CC"}."""
    return set(read_export(path, TYPE_COLUMNS)["type"]) - {""}


def _read_txt(path: Path, wanted: set[str]) -> pd.DataFrame:
    """Tab-separated Output.txt; blank lines kept so line numbers stay exact."""
    df = pd.read_csv(
        path, sep="\t", header=0, skiprows=[1], usecols=lambda c: str(c).strip() in wanted,
        skip_blank_lines=False, dtype=str, encoding_errors="replace",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_xlsx(path: Path, wanted: set[str]) -> pd.DataFrame:
    """
    Sheet "Output" of the xlsx export, read in one pass by calamine (fast,
    Rust) or, if python-calamine is not installed, by openpyxl in read-only
    mode. Empty rows are kept so list position = Excel row - 1.
    """
    if importlib.util.find_spec("python_calamine") is not None:
        from python_calamine import CalamineWorkbook

        wb = CalamineWorkbook.from_path(str(path))
        if OUTPUT_SHEET not in wb.sheet_names:
            raise ValueError(f"{path.name}: no sheet named '{OUTPUT_SHEET}' "
                             f"(sheets: {wb.sheet_names})")
        rows = wb.get_sheet_by_name(OUTPUT_SHEET).to_python(skip_empty_area=False)
        engine = "calamine"
    else:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if OUTPUT_SHEET not in wb.sheetnames:
                raise ValueError(f"{path.name}: no sheet named '{OUTPUT_SHEET}' "
                                 f"(sheets: {wb.sheetnames})")
            rows = list(wb[OUTPUT_SHEET].iter_rows(min_row=1, values_only=True))
        finally:
            wb.close()
        engine = "openpyxl"
    log.info("Reading %s with %s", path.name, engine)

    if len(rows) < 2:
        raise ValueError(f"{path.name}: sheet '{OUTPUT_SHEET}' has no data rows")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    keep = [i for i, h in enumerate(header) if h in wanted]
    data = [[r[i] if i < len(r) else None for i in keep] for r in rows[2:]]
    return pd.DataFrame(data, columns=[header[i] for i in keep], dtype=object)


def is_export(path: Path) -> bool:
    """True if `path` is an .xlsx with a sheet "Output" (cheap: sheet names only)."""
    if path.suffix.lower() not in (".xlsx", ".xlsm") or path.name.startswith("~$"):
        return False
    try:
        if importlib.util.find_spec("python_calamine") is not None:
            from python_calamine import CalamineWorkbook
            return OUTPUT_SHEET in CalamineWorkbook.from_path(str(path)).sheet_names
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            return OUTPUT_SHEET in wb.sheetnames
        finally:
            wb.close()
    except Exception:  # unreadable / not a workbook
        return False


def measurement_exports(path: str | Path) -> list[Path]:
    """
    The export files of one measurement: `path` itself if it is a file; for
    a folder every xlsx with a sheet "Output", or Output.txt if there is none.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    xlsx = [p for p in sorted(path.glob("*.xls[xm]")) if is_export(p)]
    return xlsx or [path / "Output.txt"]


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
    candidates = sorted(folder.glob(MASS_FILE_PATTERN))
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
        if not (MASS_MIN_G <= grams <= MASS_MAX_G):
            raise ValueError(
                f"{path.name}: {name} = {value} {unit} = {grams:g} g is outside the plausible "
                f"range {MASS_MIN_G:g}-{MASS_MAX_G:g} g -- check the unit.")
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


_DIAMETER_KEYS = ("d", "diameter", "electrode diameter", "d_el", "del", "electrode_diameter")
_AREA_KEYS = ("a", "area", "electrode area", "a_el", "ael", "electrode_area")
_LEN_UNIT_RE = re.compile(r"^\s*(.*?)\s*[\[(]?\s*(mm2|mm²|mm\^2|cm2|cm²|cm\^2|mm|cm)\s*[\])]?\s*$",
                          re.IGNORECASE)


def read_electrode_area(path: str | Path) -> dict | None:
    """
    Electrode area from Input-Variables.txt (lab guide PEIS 1.2), if the file
    holds an electrode diameter or area. Keys: d / diameter (unit mm or cm,
    default mm), A / area (unit cm2 or mm2, default cm2); the unit may be in
    the key ("d [mm]") or the value ("18 mm"). A = pi (d/2)^2.
    Returns {"area_cm2", "diameter_mm", "source"} or None if neither is given.
    """
    path = Path(path)
    if not path.exists():
        return None
    for e in reversed(_parse_key_values(path)):
        km = _LEN_UNIT_RE.match(e["key"])
        key, kunit = (km.group(1), km.group(2)) if km else (e["key"], None)
        vm = _LEN_UNIT_RE.match(e["raw"])
        raw, vunit = (vm.group(1), vm.group(2)) if vm else (e["raw"], None)
        unit = (vunit or kunit or "").lower().replace("²", "2").replace("^2", "2")
        value = _to_float(raw)
        if value is None or value <= 0:
            continue
        k = key.strip().lower()
        if k in _DIAMETER_KEYS and unit in ("", "mm", "cm"):
            d_mm = value * 10.0 if unit == "cm" else value
            area = np.pi * (d_mm / 20.0) ** 2
            src = f"{path.name}: {e['key']} = {e['raw']} (diameter {d_mm:g} mm)"
            log.info("Electrode area from %s -> %.4g cm2", src, area)
            return {"area_cm2": area, "diameter_mm": d_mm, "source": src}
        if k in _AREA_KEYS and unit in ("", "cm2", "mm2"):
            area = value / 100.0 if unit == "mm2" else value
            src = f"{path.name}: {e['key']} = {e['raw']}"
            log.info("Electrode area from %s -> %.4g cm2", src, area)
            return {"area_cm2": area, "diameter_mm": float(np.sqrt(area / np.pi) * 20.0),
                    "source": src}
    return None


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
