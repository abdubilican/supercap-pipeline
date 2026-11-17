"""
gcd_output.py
-------------
Results workbook of the GCD evaluation (lab guide GCD 6.1):
output/<sample-id>/gcd/results.xlsx with the sheets

  Cycles    one row per complete cycle, all section-3 values, row numbers, flags
  Rate      one row per rate block = its selected cycle, retention, stability
  Cycling   (only if present) per evaluated cycle: retention, ESR/ESR0, eta, end of life
  Floating  (only if present) per check-up and current density, vs hold time
  Holds     (only if present) per voltage hold: voltage, duration, leakage current
  Flags     every flag / warning with block, cycle and explanation
  Meta      masses, detected program, thresholds, provenance
  Warnings  every warning of the run (log)

plus results_<sheet>.csv per sheet. Row numbers are Excel rows of the sheet
"Output" (or line numbers of Output.txt), so every value can be checked by hand.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import gcd_config as cfg
from .flags import FLAG, INFO, WARNING, sort_flags
from .gcd_segmentation import CYCLING, FLOATING, RATE
from .results_io import pipeline_version, write_csv, write_workbook

CYCLE_COLUMNS = [
    "block", "occurrence", "test_type", "checkup", "cycle", "cycle_in_block", "instr_cycle",
    "step_ch", "step_dis", "j_Ag", "j_block_Ag", "I_abs_mA", "U_max", "U_min", "U_end",
    "row_ch_start", "row_ch_end", "row_dis_start", "row_dis_end", "row_a", "row_b",
    "n_ch", "n_dis", "has_rest",
    "U_before", "I_before_mA", "U1", "t_U1_ms", "I_dis_mA", "dU_IR", "dI_A", "ESR",
    "Q_ch", "Q_dis", "t_ch", "t_dis", "E_ch", "E_dis",
    "C", "C_E", "C_cell_sp", "C_el", "C_E_cell_sp", "C_E_el", "C_slope",
    "E_sp", "P_sp", "P_max_sp", "eta_C", "eta_E", "I_leak_mA", "flags",
]
UNITS = {  # column -> unit (header suffix in the CSV / xlsx)
    "j_Ag": "A/g", "j_block_Ag": "A/g", "I_abs_mA": "mA", "U_max": "V", "U_min": "V", "U_end": "V",
    "U_before": "V", "I_before_mA": "mA", "U1": "V", "t_U1_ms": "ms", "I_dis_mA": "mA",
    "dU_IR": "V", "dI_A": "A", "ESR": "Ohm", "Q_ch": "mAh", "Q_dis": "mAh", "t_ch": "s",
    "t_dis": "s", "E_ch": "mWh", "E_dis": "mWh", "C": "F", "C_E": "F", "C_cell_sp": "F/g",
    "C_el": "F/g", "C_E_cell_sp": "F/g", "C_E_el": "F/g", "C_slope": "F", "E_sp": "Wh/kg",
    "P_sp": "W/kg", "P_max_sp": "W/kg", "I_leak_mA": "mA", "C_prev": "F",
}
RATE_VALUES = ["U_max", "U_min", "U_before", "U1", "t_U1_ms", "dU_IR", "ESR", "Q_ch", "Q_dis",
               "t_ch", "t_dis", "E_ch", "E_dis", "C", "C_E", "C_cell_sp", "C_el", "C_E_cell_sp",
               "C_E_el", "C_slope", "E_sp", "P_sp", "P_max_sp", "eta_C", "eta_E", "I_leak_mA"]

EXPLANATIONS = [  # (start of the flag text, explanation)
    ("starts outside window", "The cycle does not start within 50 mV of U_min (e.g. first cycle "
                              "after an OCV rest); never used as a result."),
    ("charge does not end", "The charge does not end within 5 mV of the block's U_max; not used."),
    ("discharge does not reach", "The discharge ends more than 5 mV above U_min; "
                               "not used."),
    ("discharge incomplete", "The discharge never reaches U_min; not used."),
    ("no complete cycle", "The block has no complete cycle; it is listed but not evaluated."),
    ("IR drop read late", "U1 was logged more than 10 ms after the switch; the IR drop and ESR "
                          "include part of the capacitive drop."),
    ("eta_C", "Coulombic efficiency below 0.97 (warning)."),
    ("low efficiency", "Coulombic efficiency below 0.90: leakage or side reactions."),
    ("curve shape", "C (from Q_dis) and C_E (from E_dis) differ by more than 5 %: the discharge "
                    "curve is not linear."),
    ("resistance-limited", "The IR drop exceeds 20 % of U_max; the cell is resistance-limited "
                           "at this current."),
    ("logging too coarse", "Fewer than 50 points in the discharge; increase the logging rate."),
    ("non-standard current density", "j is not within 2 % of a value in STANDARD_J; check the "
                                     "program and the current range."),
    ("not stable", "C of the last and second-to-last complete cycle differ by more than 1 %."),
    ("single complete cycle", "Only one complete cycle in the block; the stability check is "
                              "not possible."),
    ("ESR not constant", "ESR varies by more than 10 % across the rate series."),
    ("ESR spread", "ESR spread across the rate series (within the 10 % limit)."),
    ("GCD C_el", "Comparison with the CV result of the same sample (lowest j vs lowest scan "
                 "rate); flagged above 15 %."),
    ("CV C_el", "Comparison with the CV result of the same sample."),
    ("end of life", "Retention below 80 % or ESR/ESR0 above 2."),
]


def explain(text: str) -> str:
    for start, expl in EXPLANATIONS:
        if text.startswith(start):
            return expl
    return ""


def _with_units(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: f"{c} ({UNITS[c]})" for c in df.columns if c in UNITS})


def cycles_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Sheet "Cycles": the used (complete) cycles."""
    t = ev[ev["used"]].copy()
    t["flags"] = t["cycle_flags"].map(lambda f: "; ".join(f"{s}: {x}" for s, x in f))
    t["row_a"] = t["row_a"].astype(int)
    t["row_b"] = t["row_b"].astype(int)
    return _with_units(t[CYCLE_COLUMNS])


def rate_table(rate: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Sheet "Rate": one row per rate block (selected cycle)."""
    if rate.empty:
        return pd.DataFrame()
    flags_by_cycle = ev.set_index("cycle")["cycle_flags"]
    rows = []
    for r in rate.to_dict("records"):
        f = list(r["block_flags"])
        if np.isfinite(r.get("selected_cycle", np.nan)):
            f += flags_by_cycle.get(int(r["selected_cycle"]), [])
        row = {
            "block": r["block"], "occurrence": r["occurrence"],
            "j_Ag": r.get("j_block_Ag", r.get("j_Ag")), "j_standard_Ag": r.get("j_standard_Ag"),
            "I_mA": r.get("I_block_mA", r.get("I_mA")),
            "n_cycles": r["n_cycles"], "n_complete": r["n_complete"],
            "selected_cycle": r["selected_cycle"], "instr_cycle": r.get("instr_cycle"),
            "prev_cycle": r.get("prev_cycle"),
            "row_ch_start": r.get("row_ch_start"), "row_ch_end": r.get("row_ch_end"),
            "row_dis_start": r.get("row_dis_start"), "row_dis_end": r.get("row_dis_end"),
            "row_a": r.get("row_a"), "row_b": r.get("row_b"),
            **{k: r.get(k, np.nan) for k in RATE_VALUES},
            "C_prev": r.get("C_prev", np.nan),
            "stability_pct": 100.0 * r.get("stability", np.nan),
            "retention_pct": r.get("retention_pct", np.nan),
            "flags": "; ".join(f"{s}: {x}" for s, x in f),
        }
        rows.append(row)
    return _with_units(pd.DataFrame(rows))


def flags_table(ev: pd.DataFrame, rate: pd.DataFrame, series_flags: list,
                cv_cmp: dict | None, cycling: list[dict], floating: list[dict]) -> pd.DataFrame:
    """
    Sheet "Flags": every flag with block, cycle and explanation. Rate blocks:
    each flagged cycle individually (incomplete cycles and the selected
    cycle). Long-term blocks: one line per flag type with the number of
    affected cycles and the first one.
    """
    rows = []

    def add(sev, scope, text, block=None, cycle=None, test=None, j=None):
        rows.append(dict(severity=sev, scope=scope, block=block, cycle=cycle, test_type=test,
                         j_Ag=j, flag=text, explanation=explain(text)))

    sel = {}
    for r in rate.to_dict("records") if not rate.empty else []:
        sel[r["block"]] = r.get("selected_cycle")
        for s, t in r["block_flags"]:
            add(s, "block", t, r["block"], r.get("selected_cycle"), RATE, r.get("j_block_Ag"))
    for b, g in ev.groupby("block"):
        test = g["test_type"].iloc[0]
        j = float(g["j_block_Ag"].iloc[0])
        if test == RATE:
            for r in g.itertuples():
                if not r.used or r.cycle == sel.get(b):
                    for s, t in r.cycle_flags:
                        add(s, "cycle", t, b, r.cycle, test, j)
        else:
            agg: dict[tuple, list] = {}
            for r in g.itertuples():
                for s, t in r.cycle_flags:
                    key = (s, t.split(" (")[0].split(":")[0].split(" =")[0])
                    agg.setdefault(key, []).append(r.cycle)
            for (s, kind), cyc in agg.items():
                add(s, "block", f"{kind}: {len(cyc)} of {len(g)} cycles (first: cycle {cyc[0]})",
                    b, cyc[0], test, j)
    for s, t in series_flags:
        add(s, "rate series", t, test=RATE)
    if cv_cmp:
        add(cv_cmp["severity"], "sample", cv_cmp["text"], cv_cmp["gcd_block"], test=RATE,
            j=cv_cmp["gcd_j"])
    for c in cycling:
        if c["eol_criterion"] != "not reached":
            add(INFO, "block", f"end of life at cycle {c['eol_eol_x']:.1f} ({c['eol_criterion']})",
                c["block"], test=CYCLING, j=c["j"])
    for f in floating:
        if f["eol_criterion"] != "not reached":
            add(INFO, "series", f"end of life at {f['eol_eol_x']:.1f} h hold time "
                               f"({f['eol_criterion']})", test=FLOATING, j=f["j"])
    return sort_flags(pd.DataFrame(rows, columns=["severity", "scope", "block", "cycle",
                                                  "test_type", "j_Ag", "flag", "explanation"]))


def meta_table(masses: dict, sample_id: str, source: str, ev: pd.DataFrame,
               n_holds: int, cv_cmp: dict | None, cycling: list[dict],
               floating: list[dict]) -> pd.DataFrame:
    """Sheet "Meta": masses, detected program, thresholds, provenance."""
    items = [
        ("sample_id", sample_id), ("source_file", source),
        ("m1_g", masses["m1_g"]), ("m2_g", masses["m2_g"]), ("m_mean_g", masses["m_mean_g"]),
        ("m1_as_read", masses.get("m1_raw", "")), ("m2_as_read", masses.get("m2_raw", "")),
        ("active_fraction_applied", masses.get("active_fraction", "none")),
        ("mass_source_file", masses["source"]),
        ("mass_basis", "j = I / m_mean (one electrode); C_el = 2C/m_mean; C_cell,sp = C/(2 m_mean); "
                       "E_sp, P_sp per total active mass 2 m_mean"),
        ("evaluation_date", _dt.datetime.now().isoformat(timespec="seconds")),
        ("pipeline_version", pipeline_version()),
        ("method", "lab guide 'Specific Capacitance from GCD Data (PAT-Tester)': "
                   "C = 3.6 Q_dis/(U1 - U_min), C_E = 7.2 E_dis/(U1^2 - U_min^2)"),
        ("cycles_total", len(ev)), ("cycles_used", int(ev["used"].sum()) if len(ev) else 0),
        ("voltage_holds", n_holds),
    ]
    if len(ev):
        for b, g in ev.groupby("block"):
            r = g.iloc[0]
            items.append((f"program.block_{b}",
                          f"{r.test_type}, j = {r.j_block_Ag:.4g} A/g (|I| = {r.I_block_mA:.4g} mA), "
                          f"U {r.U_min:.2f} -> {r.U_max_block:.4f} V, {len(g)} cycles "
                          f"({int(g['used'].sum())} used), occurrence {r.occurrence}"
                          + (f", check-up {r.checkup}" if r.test_type == FLOATING else "")))
        items.append(("U_max_V (blocks)", ", ".join(f"{u:.4f}" for u in
                                                     ev.groupby("block")["U_max_block"].first())))
        items.append(("U_min_V (blocks)", ", ".join(f"{u:.2f}" for u in
                                                     ev.groupby("block")["U_min"].first())))
    for c in cycling:
        items.append((f"cycling.block_{c['block']}",
                      f"reference cycle {c['reference']} ({c['reference_how']}); end of life: "
                      + ("not reached" if c["eol_criterion"] == "not reached"
                         else f"cycle {c['eol_eol_x']:.1f} ({c['eol_criterion']})")))
    for f in floating:
        items.append((f"floating.j_{f['j']:.3g}",
                      "end of life: " + ("not reached" if f["eol_criterion"] == "not reached"
                                         else f"{f['eol_eol_x']:.1f} h ({f['eol_criterion']})")))
    if cv_cmp:
        items.append(("cv_comparison", cv_cmp["text"]))
        items.append(("cv_results_file", cv_cmp["cv_file"]))
    items += [(f"threshold.{k}", v) for k, v in cfg.thresholds().items()]
    return pd.DataFrame(items, columns=["key", "value"]).astype({"value": str})


def write_results(out_dir: Path, sheets: dict[str, pd.DataFrame]) -> list[Path]:
    """results.xlsx with all non-empty sheets (Flags and Meta always) + results_<sheet>.csv."""
    keep = {k: v for k, v in sheets.items() if k in ("Flags", "Meta", "Warnings") or not v.empty}
    paths = [write_workbook(out_dir / "results.xlsx", keep)]
    for name, df in keep.items():
        paths.append(write_csv(df, out_dir / f"results_{name}.csv"))
    return paths
