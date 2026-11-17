"""
gcd_segmentation.py
-------------------
Universal segmentation of a GCD measurement (lab guide GCD section 2):
steps, half-cycles, rests, voltage holds, cycles, blocks, test types and the
cycles to evaluate. Everything is derived from the data in file order; the
instrument's cycle and step numbers are carried along for traceability only.

Data flow:
    find_steps(df) -> classify_steps() [find_half_cycles, find_rests,
    detect_holds] -> build_cycles() -> detect_blocks() -> classify_tests()
    -> mark_complete() -> select_cycles()
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import gcd_config as cfg

log = logging.getLogger("supercap_pipeline.gcd").getChild("segmentation")

CHARGE, DISCHARGE, REST, HOLD, OTHER = "charge", "discharge", "rest", "hold", "other"
RATE, CYCLING, FLOATING = "rate", "cycling", "floating"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def snap(x, grid: float = cfg.U_LIMIT_GRID_V, shift: float = cfg.U_LIMIT_SNAP_SHIFT):
    """
    Lower voltage limit from an end-of-discharge voltage: the nearest point
    of the U_LIMIT_GRID_V grid, with the rounding window shifted up by
    shift*grid because the tester stops at or below the limit
    (grid 0.05 V: -0.0375 ... +0.0125 V -> 0.0 V;  1.238 V -> 1.25 V).
    """
    v = np.asarray(x, dtype=float)
    return np.round((v + shift * grid) / grid) * grid + 0.0


def cluster_values(values, rel_tol: float = cfg.CURRENT_REL_TOL) -> np.ndarray:
    """
    Label values that agree within `rel_tol` (relative to the cluster's first,
    i.e. smallest, value) with the same integer; labels follow ascending value.
    """
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    labels = np.empty(len(v), dtype=int)
    lab, ref = -1, None
    for i in order:
        if ref is None or abs(v[i] - ref) > rel_tol * abs(ref):
            lab += 1
            ref = v[i]
        labels[i] = lab
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def find_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contiguous runs of rows with the same step number and type (a reset of
    t_step also starts a new run). Per step: positional rows i0..i1, Excel
    rows, instrument cycle/step, type, con, c/d values, number of points,
    first/last/min/max/median V12, median I12, start/end time t and duration.
    """
    n = len(df)
    step = df["step"].to_numpy()
    typ = df["type"].to_numpy()
    ts = df["t_step"].to_numpy(dtype=float)
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = (step[1:] != step[:-1]) | (typ[1:] != typ[:-1]) | (np.diff(ts) < 0)
    sid = np.cumsum(new) - 1
    g = df.assign(_sid=sid).groupby("_sid", sort=True)
    steps = pd.DataFrame({
        "i0": np.flatnonzero(new),
        "row_start": g["excel_row"].first().to_numpy(),
        "row_end": g["excel_row"].last().to_numpy(),
        "cycle": g["cycle"].first().to_numpy(),
        "step": g["step"].first().to_numpy(),
        "type": g["type"].first().to_numpy(),
        "con": g["con"].first().to_numpy(),
        "cd": g["c/d"].agg(lambda s: ",".join(sorted(set(s) - {""}))).to_numpy(),
        "n_points": g.size().to_numpy(),
        "V_first": g["V12"].first().to_numpy(),
        "V_last": g["V12"].last().to_numpy(),
        "V_min": g["V12"].min().to_numpy(),
        "V_max": g["V12"].max().to_numpy(),
        "V_median": g["V12"].median().to_numpy(),
        "I_median": g["I12"].median().to_numpy(),
        "t_start": g["t"].first().to_numpy(),
        "t_end": g["t"].last().to_numpy(),
    })
    steps["i1"] = np.append(steps["i0"].to_numpy()[1:] - 1, n - 1)
    steps["duration_s"] = steps["t_end"] - steps["t_start"]
    # last logging step |V[i1] - V[i1-1]|: how far the logged end point can lie
    # beyond the limit (points logged after the stop condition was met)
    V = df["V12"].to_numpy(dtype=float)
    i0, i1 = steps["i0"].to_numpy(), steps["i1"].to_numpy()
    steps["dV_last"] = np.where(i1 > i0, np.abs(V[i1] - V[np.maximum(i1 - 1, 0)]), 0.0)
    return steps


def find_half_cycles(steps: pd.DataFrame) -> np.ndarray:
    """
    Kind of every CC step (lab guide 2): charge if median I12 > 0, discharge
    if median I12 < 0. Mismatches with the c/d column are logged.
    Returns an array of kinds ("charge"/"discharge"/"" for non-CC steps).
    """
    cc = (steps["type"] == "CC").to_numpy()
    med = steps["I_median"].to_numpy()
    kind = np.where(cc & (med > 0), CHARGE, np.where(cc & (med < 0), DISCHARGE, ""))
    expected = np.where(kind == CHARGE, "c", np.where(kind == DISCHARGE, "d", ""))
    bad = (kind != "") & (steps["cd"].to_numpy() != expected)
    if bad.any():
        ex = steps[bad].head(5)
        log.warning("%d CC step(s) whose c/d disagrees with the current sign, e.g. Excel rows %s",
                    int(bad.sum()), ", ".join(f"{a}-{b} (c/d '{c}')" for a, b, c in
                                              zip(ex.row_start, ex.row_end, ex.cd)))
    return kind


def find_rests(steps: pd.DataFrame) -> np.ndarray:
    """Rest steps: type "Rest"."""
    return (steps["type"].str.lower() == "rest").to_numpy()


def detect_holds(df: pd.DataFrame, steps: pd.DataFrame) -> np.ndarray:
    """
    Voltage holds (floating, lab guide 2.1), detected from the data: a step
    in which V12 stays within +-HOLD_V_TOL_V of its median for at least
    HOLD_MIN_DURATION_S while |I12| decays (the largest |I| in the first
    10 % of the points above the median |I| of the last 10 %; a rest with
    I = 0 is not a hold). The type/con values of the detected holds are
    logged for cross-checking.
    """
    cand = ((steps["duration_s"] >= cfg.HOLD_MIN_DURATION_S)
            & ((steps["V_max"] - steps["V_median"]) <= cfg.HOLD_V_TOL_V)
            & ((steps["V_median"] - steps["V_min"]) <= cfg.HOLD_V_TOL_V)).to_numpy()
    hold = np.zeros(len(steps), dtype=bool)
    I = df["I12"].to_numpy(dtype=float)
    for k in np.flatnonzero(cand):
        s = steps.iloc[k]
        a = np.abs(I[s.i0:s.i1 + 1])
        m = max(1, len(a) // 10)
        i_start, i_end = np.max(a[:m]), np.median(a[-m:])
        hold[k] = i_start > 0 and i_end < i_start
    if hold.any():
        found = steps[hold].groupby(["type", "con"]).size()
        log.info("Voltage holds detected: %d; type/con of these steps: %s", int(hold.sum()),
                 ", ".join(f"type '{t}' / con '{c}': {n}" for (t, c), n in found.items()))
    return hold


def classify_steps(df: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Add the column `kind` (charge / discharge / rest / hold / other) to `steps`."""
    steps = steps.copy()
    kind = find_half_cycles(steps)
    kind = np.where(find_rests(steps) & (kind == ""), REST, kind)
    hold = detect_holds(df, steps)
    kind = np.where(hold, HOLD, kind)
    kind = np.where(kind == "", OTHER, kind)
    steps["kind"] = kind
    other = steps[steps["kind"] == OTHER]
    if len(other):
        found = other.groupby(["type", "con"]).size()
        log.warning("%d step(s) of other types ignored: %s", len(other),
                    ", ".join(f"type '{t}' / con '{c}': {n}" for (t, c), n in found.items()))
    log.info("Steps: %s", ", ".join(f"{k} {v}" for k, v in steps["kind"].value_counts().items()))
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Cycles
# ─────────────────────────────────────────────────────────────────────────────

def build_cycles(df: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """
    A cycle is a charge followed by the next discharge, possibly with a rest
    step in between (lab guide 2). Data order, not the instrument's cycle
    column. A charge without discharge, a discharge without charge, or a
    charge interrupted by a hold/other step is logged and not a cycle.
    `checkup` = number of voltage holds before the cycle (floating tests).
    Per cycle: |I| (median of the charge), U_max (last V12 of the charge),
    U_end (last V12 of the discharge), V_pre (V12 in the row before the
    charge, i.e. where the cycle starts), points and row numbers.
    """
    V = df["V12"].to_numpy(dtype=float)
    kinds = steps["kind"].to_numpy()
    out = []
    pending_ch, pending_rest = None, None
    n_holds = 0
    orphans = {"charge": 0, "discharge": 0}
    for k, kind in enumerate(kinds):
        if kind == CHARGE:
            if pending_ch is not None:
                orphans["charge"] += 1
            pending_ch, pending_rest = k, None
        elif kind == REST:
            if pending_ch is not None:
                pending_rest = k
        elif kind == DISCHARGE:
            if pending_ch is None:
                orphans["discharge"] += 1
                continue
            out.append((pending_ch, pending_rest, k, n_holds))
            pending_ch, pending_rest = None, None
        else:  # hold / other
            if pending_ch is not None:
                orphans["charge"] += 1
            pending_ch, pending_rest = None, None
            if kind == HOLD:
                n_holds += 1
    if orphans["charge"] or orphans["discharge"]:
        log.info("Half-cycles not part of a cycle: %d charge(s) without discharge, "
                 "%d discharge(s) without preceding charge", orphans["charge"], orphans["discharge"])

    if not out:
        return pd.DataFrame()
    ch, rest, dis, checkup = (np.array(x) for x in zip(*out))
    rest = np.array([-1 if r is None else r for r in rest])
    S = steps
    cyc = pd.DataFrame({
        "cycle": np.arange(1, len(ch) + 1),
        "ch": ch, "rest": rest, "dis": dis, "checkup": checkup,
        "instr_cycle": S["cycle"].to_numpy()[ch],
        "step_ch": S["step"].to_numpy()[ch],
        "step_dis": S["step"].to_numpy()[dis],
        "ch_i0": S["i0"].to_numpy()[ch], "ch_i1": S["i1"].to_numpy()[ch],
        "dis_i0": S["i0"].to_numpy()[dis], "dis_i1": S["i1"].to_numpy()[dis],
        "row_ch_start": S["row_start"].to_numpy()[ch], "row_ch_end": S["row_end"].to_numpy()[ch],
        "row_dis_start": S["row_start"].to_numpy()[dis],
        "row_dis_end": S["row_end"].to_numpy()[dis],
        "has_rest": rest >= 0,
        "I_abs_mA": np.abs(S["I_median"].to_numpy()[ch]),
        "U_max": S["V_last"].to_numpy()[ch],
        "U_end": S["V_last"].to_numpy()[dis],
        "n_ch": S["n_points"].to_numpy()[ch],
        "n_dis": S["n_points"].to_numpy()[dis],
        "dV_last_ch": S["dV_last"].to_numpy()[ch],
        "dV_last_dis": S["dV_last"].to_numpy()[dis],
    })
    i0 = cyc["ch_i0"].to_numpy()
    cyc["V_pre"] = np.where(i0 > 0, V[np.maximum(i0 - 1, 0)], V[i0])
    cyc["dV_last_pre"] = np.where(i0 > 1, np.abs(V[np.maximum(i0 - 1, 0)] - V[np.maximum(i0 - 2, 0)]),
                                  0.0)
    log.info("Built %d cycles (%d with a rest between charge and discharge)",
             len(cyc), int(cyc["has_rest"].sum()))
    return cyc


# ─────────────────────────────────────────────────────────────────────────────
# Blocks and test types
# ─────────────────────────────────────────────────────────────────────────────

def voltage_tolerance(cycles: pd.DataFrame) -> np.ndarray:
    """
    Per-cycle tolerance for "ends at the limit": VOLTAGE_TOL_V (5 mV), widened
    to the larger last logging step of the cycle's charge and discharge. With
    coarse logging the tester records a point or two after the stop
    condition, so the logged end voltage lies beyond the limit by up to about
    one logging step and jitters by that much from cycle to cycle. With fine
    logging (steps < 5 mV) this is exactly the 5 mV of the guide.
    """
    return np.maximum.reduce([np.full(len(cycles), cfg.VOLTAGE_TOL_V),
                              cycles["dV_last_ch"].to_numpy(),
                              cycles["dV_last_dis"].to_numpy()])


def detect_blocks(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    A block is a run of consecutive cycles with the same |I| (within
    CURRENT_REL_TOL) and the same voltage limits: U_max (last V12 of the
    charge) agrees within voltage_tolerance() of both cycles; U_min changes
    only if the snapped end-of-discharge voltage changes AND the end voltages
    differ by more than that tolerance (several points can be logged after
    the stop condition, so the end voltage alone jitters by more than one
    logging step). A voltage hold always ends a block. A combination that occurs again later
    is a new block; `occurrence` counts its repetitions.
    Adds per cycle: block, occurrence, cycle_in_block, and the block values
    I_block_mA, U_max_block, U_end_block (medians) and U_min (lower limit:
    the end voltage closest to the limit -- the largest one in the block --
    snapped to U_LIMIT_GRID_V).
    """
    c = cycles.copy()
    I = c["I_abs_mA"].to_numpy()
    umax = c["U_max"].to_numpy()
    uend = c["U_end"].to_numpy()
    tol = voltage_tolerance(c)
    pair_tol = np.maximum(tol[1:], tol[:-1])
    new = np.ones(len(c), dtype=bool)
    new[1:] = ((np.abs(np.diff(I)) > cfg.CURRENT_REL_TOL * I[:-1])
               | (np.abs(np.diff(umax)) > pair_tol)
               | ((np.diff(snap(uend)) != 0) & (np.abs(np.diff(uend)) > pair_tol))
               | (np.diff(c["checkup"].to_numpy()) != 0))
    block = np.cumsum(new)
    c["block"] = block
    c["cycle_in_block"] = c.groupby("block").cumcount() + 1
    g = c.groupby("block")
    for col, src in (("I_block_mA", "I_abs_mA"), ("U_max_block", "U_max"), ("U_end_block", "U_end")):
        c[col] = g[src].transform("median")
    c["U_min"] = snap(g["U_end"].transform("max"))

    # occurrence: same |I| (1 %), U_max (5 mV) and U_min seen in an earlier block
    first = c.groupby("block").first()
    occ, seen = {}, []
    for b, row in first.iterrows():
        n = sum(1 for (i, u, m) in seen
                if abs(row.I_block_mA - i) <= cfg.CURRENT_REL_TOL * i
                and abs(row.U_max_block - u) <= cfg.VOLTAGE_TOL_V and row.U_min == m)
        occ[b] = n + 1
        seen.append((row.I_block_mA, row.U_max_block, row.U_min))
    c["occurrence"] = c["block"].map(occ)
    log.info("Detected %d block(s)", len(first))
    return c


def classify_tests(cycles: pd.DataFrame, has_holds: bool) -> pd.DataFrame:
    """
    Test type per block (lab guide 2): cycling if the block has at least
    N_CYCLING cycles; floating if the file contains voltage holds (the CC
    cycles between two holds form one check-up); otherwise rate.
    """
    c = cycles.copy()
    n = c.groupby("block")["cycle"].transform("size")
    c["n_cycles_block"] = n
    c["test_type"] = np.where(n >= cfg.N_CYCLING, CYCLING, FLOATING if has_holds else RATE)
    return c


def mark_complete(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Complete cycle (lab guide 2): the charge ends at U_max (within
    voltage_tolerance() of the block median -- 5 mV with fine logging), the
    discharge reaches U_min (its last V12 at most VOLTAGE_TOL_V above U_min;
    overshooting below is normal), and the cycle starts near U_min (V12 in
    the row before the charge within START_TOL_V plus the last logging step
    there). Adds `complete` and `incomplete_reason`. The U_min crossing
    itself is located in the evaluation.
    """
    c = cycles.copy()
    start_bad = np.abs(c["V_pre"] - c["U_min"]) > cfg.START_TOL_V + c["dV_last_pre"]
    umax_bad = np.abs(c["U_max"] - c["U_max_block"]) > voltage_tolerance(c)
    uend_bad = c["U_end"] > c["U_min"] + cfg.VOLTAGE_TOL_V
    reason = np.where(start_bad, "starts outside window",
                      np.where(umax_bad, "charge does not end at U_max",
                               np.where(uend_bad, "discharge does not reach U_min", "")))
    c["incomplete_reason"] = reason
    c["complete"] = reason == ""
    for _, r in c[~c["complete"]].iterrows():
        log.info("Cycle %d (block %d, Excel rows %d-%d): %s (starts at %.4f V, U_min %.2f V)",
                 r.cycle, r.block, r.row_ch_start, r.row_dis_end, r.incomplete_reason,
                 r.V_pre, r.U_min)
    return c


def select_cycles(evaluated: pd.DataFrame) -> dict[int, dict]:
    """
    Rate blocks: the last complete (and evaluable) cycle, plus the
    second-to-last for the stability check (None if there is only one).
    Returns {block: {"last": cycle number, "prev": cycle number or None,
    "n_complete": int}}.
    """
    out = {}
    ok = evaluated[evaluated["used"]]
    for b, g in ok.groupby("block"):
        cyc = g["cycle"].to_numpy()
        out[int(b)] = {"last": int(cyc[-1]), "prev": int(cyc[-2]) if len(cyc) > 1 else None,
                       "n_complete": len(cyc)}
    return out
