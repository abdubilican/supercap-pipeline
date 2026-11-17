"""
peis_output.py
--------------
Results workbook of the PEIS evaluation (lab guide PEIS 7.1):
output/<sample-id>/peis/results.xlsx with the sheets

  Spectra        one row per spectrum: label, order, rows, DC voltage, A_set, range,
                 points used/excluded, stationarity, section-4 values, KK, fit, flags
  Points         every point: row, f, Z', Z'', |Z|, phi, SFDR, dV12, derived values,
                 excluded + reason, KK residuals, fit values and residuals
  Reversibility  (only with repeated DC voltages) first vs last spectrum
  Comparison     GCD / CV comparison
  Flags          every flag / warning / info with spectrum, row and explanation
  Meta           masses, area and its source, all settings, versions
  Warnings       every warning of the run (log)

plus results_<sheet>.csv per sheet. Row numbers are the line numbers of
Output-Impedance.txt (= Excel rows), so every value can be checked by hand.
"""

from __future__ import annotations

import datetime as _dt
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import peis_config as cfg
from .results_io import pipeline_version, write_csv, write_workbook

SPECTRA_COLUMNS = [
    "label", "order", "step", "row_start", "row_end", "V_dc", "occurrence", "A_set_mV",
    "f_max", "f_min", "points_per_decade", "n_points", "n_used", "n_excluded", "n_inductive",
    "I_first_mA", "I_last_mA", "I_change_mA",
    "arc", "R_s_circle", "R_c_circle", "a", "b", "r", "f_apex", "C_c",
    "f_knee", "Zre_knee", "mZim_knee", "row_knee", "Zre_1k", "f_1k", "row_1k",
    "f_min_used", "C_s_fmin", "C_re_fmin", "f_Cs_01", "C_s_01",
    "C_el", "C_cell_sp", "C_el_re", "C_cell_sp_re", "f0", "tau0", "f0_decades_above_fmin",
    "phi_fmin", "kk_M", "kk_mu", "kk_converged", "kk_max_res_pct", "kk_f_max_res",
    "fit_circuit", "fit_success",
]
FIT_PARAMS = ["L", "R_s", "R_c", "Q_c", "alpha_c", "R_ion", "Q_dl", "alpha_dl"]
SPECTRA_TAIL = ["C_c_eff", "EDR", "fit_max_res_pct",
                "R_s_area_Ohm_cm2", "R_c_area_Ohm_cm2", "R_ion_area_Ohm_cm2",
                "Zre_1k_area_Ohm_cm2", "EDR_area_Ohm_cm2", "R_s_circle_area_Ohm_cm2",
                "R_c_circle_area_Ohm_cm2", "flags"]
POINT_COLUMNS = [
    "label", "spectrum", "excel_row", "f", "Zre", "Zim", "Zabs", "phi", "SFDR", "dV_mV", "I_mA",
    "omega", "mZim", "Z2", "C_s", "C_re", "C_im", "Cs_uF", "Cs_dev", "inductive", "excluded",
    "exclude_reason", "kk_res_re", "kk_res_im", "fit_Zre", "fit_Zim", "fit_res_re", "fit_res_im",
]
UNITS = {
    "V_dc": "V", "A_set_mV": "mV", "f_max": "Hz", "f_min": "Hz", "I_first_mA": "mA",
    "I_last_mA": "mA", "I_change_mA": "mA", "R_s_circle": "Ohm", "R_c_circle": "Ohm", "a": "Ohm",
    "b": "Ohm", "r": "Ohm", "f_apex": "Hz", "C_c": "F", "f_knee": "Hz", "Zre_knee": "Ohm",
    "mZim_knee": "Ohm", "Zre_1k": "Ohm", "f_1k": "Hz", "f_min_used": "Hz", "C_s_fmin": "F",
    "C_re_fmin": "F", "f_Cs_01": "Hz", "C_s_01": "F", "C_el": "F/g", "C_cell_sp": "F/g",
    "C_el_re": "F/g", "C_cell_sp_re": "F/g", "f0": "Hz", "tau0": "s", "phi_fmin": "deg",
    "kk_max_res_pct": "%", "kk_f_max_res": "Hz", "C_c_eff": "F", "EDR": "Ohm",
    "fit_max_res_pct": "%", "f": "Hz", "Zre": "Ohm", "Zim": "Ohm", "Zabs": "Ohm", "phi": "deg",
    "SFDR": "dB", "dV_mV": "mV", "I_mA": "mA", "omega": "rad/s", "mZim": "Ohm", "Z2": "Ohm^2",
    "C_s": "F", "C_re": "F", "C_im": "F", "Cs_uF": "uF", "kk_res_re": "%", "kk_res_im": "%",
    "fit_Zre": "Ohm", "fit_Zim": "Ohm", "fit_res_re": "%", "fit_res_im": "%",
}

EXPLANATIONS = [
    ("point excluded", "Excluded by the rules of section 3.1 (amplitude, SFDR, inductive run); "
                       "kept in the tables, not used for evaluation, KK test or fit."),
    ("KK residual", "Lin-KK residual |dZ|/|Z| above 1 % (warning) or 2 % (flag): the point may "
                    "violate causality/linearity/stationarity."),
    ("C_s differs", "The calculated C_s = -1/(omega Z'') does not match the instrument's Cs."),
    ("no high-frequency arc", "No maximum of -Z'' within the arc points; R_c = 0, fit without arc."),
    ("R_s upper bound", "-Z'' does not cross zero; R_s is the smallest measured Z'."),
    ("arc circle", "The fitted circle does not reach the real axis."),
    ("tau0 uncertain", "The C'' maximum lies within half a decade of f_min; extend the lower "
                       "frequency limit."),
    ("tau0 not determined", "C'' still rises at f_min; extend the lower frequency limit."),
    ("capacitive line tilted", "phi(f_min) above -80 deg: the low-frequency line is not vertical "
                               "(dispersion / leakage)."),
    ("fit failed", "The equivalent-circuit fit did not converge; graphical values are kept."),
    ("fit: relative uncertainty", "A fit parameter is poorly determined (> 10 %)."),
    ("fit: max residual", "The fit deviates by more than 5 % at some point."),
    ("fit:", "Fit parameter at a bound or outside the physical range."),
    ("fit and graphical", "The fitted R_s / R_c disagree with the circle fit."),
    ("not reversible", "A repeated DC voltage gives a different spectrum (> 5 %)."),
    ("GCD ESR", "Comparison with the GCD ESR at the lowest current density."),
    ("GCD C", "Comparison with the GCD capacitance at the lowest current density."),
    ("CV C", "Comparison with the CV capacitance at the slowest scan rate."),
]


def explain(text: str) -> str:
    for start, expl in EXPLANATIONS:
        if text.startswith(start):
            return expl
    return ""


def _with_units(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: f"{c} ({UNITS[c]})" for c in df.columns if c in UNITS})


def spectra_table(spectra: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Sheet "Spectra": one row per spectrum."""
    s = spectra.copy()
    fit_cols = []
    for n in FIT_PARAMS:
        for suf in ("", "_conf", "_rel_unc"):
            c = f"fit_{n}{suf}"
            if c in s:
                fit_cols.append(c)
    per = flags[flags["scope"].isin(["spectrum", "point"])].groupby("spectrum")
    s["flags"] = s["label"].map(lambda lab: "; ".join(
        f"{r.severity}: {r.flag}" + (f" (row {int(r.row)})" if pd.notna(r.row) else "")
        for r in per.get_group(lab).itertuples()) if lab in per.groups else "")
    cols = [c for c in SPECTRA_COLUMNS if c in s] + fit_cols + [c for c in SPECTRA_TAIL if c in s]
    out = s[cols].rename(columns={f"fit_{n}_rel_unc": f"fit_{n}_rel_unc (-)" for n in FIT_PARAMS})
    return _with_units(out)


def points_table(points: pd.DataFrame) -> pd.DataFrame:
    """Sheet "Points": every point of every spectrum."""
    return _with_units(points[[c for c in POINT_COLUMNS if c in points]])


def meta_table(masses: dict, area: dict, sample_id: str, source: str, spectra: pd.DataFrame,
               linkk_note: str) -> pd.DataFrame:
    """Sheet "Meta": masses, area and its source, all settings, versions."""
    try:
        imp_ver = metadata.version("impedance")
    except metadata.PackageNotFoundError:
        imp_ver = "not installed"
    items = [
        ("sample_id", sample_id), ("source_file", source),
        ("m1_g", masses["m1_g"]), ("m2_g", masses["m2_g"]), ("m_mean_g", masses["m_mean_g"]),
        ("mass_source_file", masses["source"]),
        ("area_cm2", area["area_cm2"]), ("electrode_diameter_mm", area["diameter_mm"]),
        ("area_source", area["source"]),
        ("evaluation_date", _dt.datetime.now().isoformat(timespec="seconds")),
        ("pipeline_version", pipeline_version()),
        ("impedance.py", imp_ver), ("lin_kk", f"own implementation (guide 3.3); {linkk_note}"),
        ("method", "lab guide 'Impedance Spectroscopy (PEIS) of Supercapacitors (PAT-Tester)'"),
        ("spectra", "; ".join(f"{r.label}: step {r.step}, rows {r.row_start}-{r.row_end}"
                              for r in spectra.itertuples())),
    ]
    items += [(f"setting.{k}", v) for k, v in cfg.settings().items()]
    return pd.DataFrame(items, columns=["key", "value"]).astype({"value": str})


def write_results(out_dir: Path, sheets: dict[str, pd.DataFrame]) -> list[Path]:
    """results.xlsx with all non-empty sheets (Flags/Meta/Comparison always) + csv per sheet."""
    keep = {k: v for k, v in sheets.items()
            if k in ("Flags", "Meta", "Comparison", "Warnings") or not v.empty}
    paths = [write_workbook(out_dir / "results.xlsx", keep)]
    for name, df in keep.items():
        paths.append(write_csv(df, out_dir / f"results_{name}.csv"))
    return paths
