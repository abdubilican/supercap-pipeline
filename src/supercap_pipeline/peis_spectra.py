"""
peis_spectra.py
---------------
Reading and preparing PEIS spectra (lab guide PEIS sections 1-3, 4.1):

  read_impedance_export()  Output-Impedance.txt via the shared reader; NaN end
                           rows dropped, other step types ignored (both logged)
  split_spectra()          one spectrum = contiguous rows with the same step and
                           type PEIS, ended by the NaN end row
  label_spectra()          DC voltage, occurrence, label ("0 V (1)"), file-name
                           slug ("0V_1"), colour / marker / line style
  exclude_points()         amplitude, SFDR and inductive-run rules (with reason)
  derived_columns()        omega, -Z'', |Z|^2, C_s, C', C'' and the Cs check

Internal column names: f, Zre, Zim, Zabs, phi, SFDR, Cs_uF, dV_mV, I_mA, dI_mA.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import peis_config as cfg
from .export_io import HALF_CELL_COLUMNS, PEIS_COLUMNS, read_export
from .plot_style import sequential_colors

log = logging.getLogger("supercap_pipeline.peis").getChild("spectra")

RENAME = {"freq": "f", "Re(Z12)": "Zre", "Im(Z12)": "Zim", "|Z12|": "Zabs", "phi(Z12)": "phi",
          "SFDR(Z12)": "SFDR", "Cs": "Cs_uF", "dV12": "dV_mV", "I12": "I_mA", "dI12": "dI_mA"}
OCCURRENCE_STYLES = [  # (marker, line style, filled) for the 1st, 2nd, ... spectrum at one voltage
    ("o", "-", True), ("s", "--", False), ("^", ":", True), ("D", "-.", False),
    ("v", (0, (5, 1)), True), ("P", (0, (1, 1)), False),
]


def read_impedance_export(path: str | Path) -> pd.DataFrame:
    """
    Read Output-Impedance.txt (lab guide 1.1) with the shared reader: names
    and strings stripped, "NaN" parsed as missing, excel_row = line number.
    Adds `spectrum` (running id: a new spectrum starts at a change of step
    number or after a NaN end row), then drops the NaN end rows (freq or
    Z12 missing) and rows of other step types (both logged). Half-cell
    columns (Z1, Z2, ZR, Z1R, Z2R) are only reported if they hold data.
    """
    df = read_export(path, PEIS_COLUMNS)
    nan_row = df["freq"].isna() | df["Re(Z12)"].isna() | df["Im(Z12)"].isna()
    step = df["step"].to_numpy()
    new = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        new[1:] = (step[1:] != step[:-1]) | nan_row.to_numpy()[:-1]
    df["spectrum"] = np.cumsum(new)

    used = [c for c in HALF_CELL_COLUMNS if df[c].notna().any()]
    if used:
        log.info("Half-cell columns contain data (not used for the two-electrode "
                 "evaluation): %s", ", ".join(used))
    if nan_row.any():
        log.info("Dropped %d NaN end row(s) (Excel rows %s)", int(nan_row.sum()),
                 ", ".join(map(str, df.loc[nan_row, "excel_row"].tolist())))
    df = df[~nan_row]
    other = df["type"] != "PEIS"
    if other.any():
        log.warning("Ignored %d row(s) of other step types: %s", int(other.sum()),
                    df.loc[other, "type"].value_counts().to_dict())
        df = df[~other]
    df = df.drop(columns=list(HALF_CELL_COLUMNS)).rename(columns=RENAME)
    return df.reset_index(drop=True)


def split_spectra(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spectra (lab guide 2): one row per spectrum with order index (by t),
    step, Excel rows, number of points, DC voltage = median V12 rounded to
    DC_VOLTAGE_ROUND_V, A_set = median dV12, f_max, f_min, points per decade,
    I12 of the first and last point (stationarity, 3.2).
    Returns (spectra, points) with points sorted by spectrum order and
    frequency from high to low; spectrum ids are renumbered by order (1, 2, ...).
    """
    first_t = df.groupby("spectrum")["t"].min().sort_values()
    order = {sid: k + 1 for k, sid in enumerate(first_t.index)}
    pts = df.assign(spectrum=df["spectrum"].map(order))
    g = pts.groupby("spectrum", sort=True)
    spec = pd.DataFrame({
        "spectrum": g.size().index,
        "order": g.size().index,
        "step": g["step"].first().to_numpy(),
        "row_start": g["excel_row"].min().to_numpy(),
        "row_end": g["excel_row"].max().to_numpy(),
        "n_points": g.size().to_numpy(),
        "V_dc": np.round(g["V12"].median().to_numpy() / cfg.DC_VOLTAGE_ROUND_V)
                * cfg.DC_VOLTAGE_ROUND_V + 0.0,
        "A_set_mV": g["dV_mV"].median().to_numpy(),
        "f_max": g["f"].max().to_numpy(),
        "f_min": g["f"].min().to_numpy(),
        "I_first_mA": g["I_mA"].first().to_numpy(),
        "I_last_mA": g["I_mA"].last().to_numpy(),
        "t_start": g["t"].min().to_numpy(),
    })
    spec["V_dc"] = spec["V_dc"].round(6)
    spec["points_per_decade"] = (spec["n_points"] - 1) / np.log10(spec["f_max"] / spec["f_min"])
    spec["I_change_mA"] = spec["I_last_mA"] - spec["I_first_mA"]
    pts = pts.sort_values(["spectrum", "f"], ascending=[True, False], kind="stable") \
             .reset_index(drop=True)
    log.info("Detected %d spectra: %s", len(spec), "; ".join(
        f"step {r.step}: {r.n_points} points {r.f_max:g} Hz -> {r.f_min:g} Hz at {r.V_dc:g} V"
        for r in spec.itertuples()))
    return spec, pts


def _slug_voltage(v: float) -> str:
    return f"{v:g}".replace("-", "m").replace(".", "p") + "V"


def label_spectra(spec: pd.DataFrame) -> pd.DataFrame:
    """
    Labels (lab guide 2): DC voltage plus occurrence number if that voltage
    occurs more than once ("0 V (1)", "0.5 V", "0 V (2)"); file-name slug
    ("0V_1", "0p5V"). Colour by DC voltage (sequential palette, low -> high);
    repeated voltages keep the hue and get another marker / line style /
    fill (1st: filled circle, solid; 2nd: open square, dashed; ...).
    """
    s = spec.copy()
    s["occurrence"] = s.groupby("V_dc").cumcount() + 1
    s["n_at_voltage"] = s.groupby("V_dc")["spectrum"].transform("size")
    rep = s["n_at_voltage"] > 1
    s["label"] = [f"{v:g} V" + (f" ({k})" if r else "")
                  for v, k, r in zip(s["V_dc"], s["occurrence"], rep)]
    s["slug"] = [_slug_voltage(v) + (f"_{k}" if r else "")
                 for v, k, r in zip(s["V_dc"], s["occurrence"], rep)]
    colors = sequential_colors(s["V_dc"].tolist())
    s["color"] = s["V_dc"].map(colors)
    sty = [OCCURRENCE_STYLES[(k - 1) % len(OCCURRENCE_STYLES)] for k in s["occurrence"]]
    s["marker"] = [m for m, _, _ in sty]
    s["linestyle"] = [ls for _, ls, _ in sty]
    s["filled"] = [fl for _, _, fl in sty]
    return s


def exclude_points(pts: pd.DataFrame, spec: pd.DataFrame,
                   include_L: bool = cfg.FIT_INCLUDE_L) -> pd.DataFrame:
    """
    Point exclusion (lab guide 3.1), per spectrum with A_set = median dV12:
      amplitude:  |dV12 - A_set| / A_set > AMP_TOL
      signal:     SFDR(Z12) < SFDR_MIN
      inductance: Im(Z12) > 0 for the contiguous run starting at the highest
                  frequency (kept if include_L, then fitted with L0)
    Adds `inductive`, `excluded` and `exclude_reason`. Excluded points stay
    in all tables but are not used for evaluation, KK test or fit.
    """
    p = pts.copy()
    a_set = p["spectrum"].map(spec.set_index("spectrum")["A_set_mV"])
    amp = (p["dV_mV"] - a_set).abs() / a_set > cfg.AMP_TOL
    sfdr = p["SFDR"] < cfg.SFDR_MIN
    pos = (p["Zim"] > 0).astype(int)
    p["inductive"] = pos.groupby(p["spectrum"]).cumprod().astype(bool)
    reasons = []
    for r, a, s, ind, aset in zip(p.itertuples(), amp, sfdr, p["inductive"], a_set):
        why = []
        if a:
            why.append(f"amplitude {r.dV_mV:.1f} mV (A_set {aset:.1f} mV)")
        if s:
            why.append(f"SFDR {r.SFDR:.1f} dB < {cfg.SFDR_MIN:g} dB")
        if ind and not include_L:
            why.append("inductive (Im(Z) > 0)")
        reasons.append("; ".join(why))
    p["exclude_reason"] = reasons
    p["excluded"] = p["exclude_reason"] != ""
    labels = spec.set_index("spectrum")["label"] if "label" in spec else None
    for sid, g in p[p["excluded"]].groupby("spectrum"):
        name = labels[sid] if labels is not None else sid
        log.info("Spectrum %s: %d point(s) excluded: %s", name, len(g),
                 "; ".join(f"row {r.excel_row} ({r.exclude_reason})" for r in g.itertuples()))
    if include_L and p["inductive"].any():
        log.info("%d inductive point(s) kept (FIT_INCLUDE_L): fitted with L0", int(p["inductive"].sum()))
    return p


def derived_columns(pts: pd.DataFrame) -> pd.DataFrame:
    """
    Derived quantities per point (lab guide 4.1):
      omega = 2 pi f;  -Z'';  |Z|^2 = Z'^2 + Z''^2
      C_s = -1 / (omega Z'')                 series capacitance  [F]
      C'  = -Z'' / (omega |Z|^2)             real part of the complex capacitance
      C'' =  Z'  / (omega |Z|^2)             imaginary part
    Cs_dev = (C_s - Cs_instrument) / C_s with Cs_instrument = Cs[uF] * 1e-6.
    """
    p = pts.copy()
    p["omega"] = 2.0 * np.pi * p["f"]
    p["mZim"] = -p["Zim"]
    p["Z2"] = p["Zre"] ** 2 + p["Zim"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        p["C_s"] = -1.0 / (p["omega"] * p["Zim"])
        p["C_re"] = -p["Zim"] / (p["omega"] * p["Z2"])
        p["C_im"] = p["Zre"] / (p["omega"] * p["Z2"])
        p["Cs_dev"] = (p["C_s"] - p["Cs_uF"] * 1e-6) / p["C_s"]
    return p
