"""
cv_analysis.py
--------------
Charges, capacitances and quality checks of the CV evaluation (lab guide
sections 3-5).

Sign convention: Qcd is the signed charge since the start of the step
(positive in up sweeps, negative in down sweeps). Q = Qcd - Qcd[first row
of the sweep]. Signs are kept as computed -- a negative discharge charge is
a real result and makes the polarity "not evaluable", it is never abs()'d.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import cv_config as cfg
from .cv_segmentation import SYMMETRIC, expected_sweep_duration, select_last_cycle

log = logging.getLogger("supercap_pipeline.cv").getChild("analysis")

POLARITIES = ("+", "-")


# ─────────────────────────────────────────────────────────────────────────────
# 0 V crossing
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_zero_crossing(V: np.ndarray, Q: np.ndarray) -> dict | None:
    """
    First 0 V crossing of a sweep and the charge there.

    Crossing: row a with V[a] * V[a+1] < 0, b = a + 1, and
        Q0 = Q_a - V_a * (Q_b - Q_a) / (V_b - V_a)        (linear interpolation)
    A row with V exactly 0 is a crossing at that row: a = b, Q0 = Q_a.
    Returns dict(a, b, frac, Q0, n_crossings) with a, b as positions in V,
    frac = position of 0 V between a and b (0..1), or None if V never
    changes sign. If there are several crossings the first is used.
    """
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    events = [(int(i), int(i) + 1) for i in np.flatnonzero(V[:-1] * V[1:] < 0)]
    events += [(int(i), int(i)) for i in np.flatnonzero(V == 0)]
    if not events:
        return None
    events.sort()
    a, b = events[0]
    if a == b:
        frac, q0 = 0.0, float(Q[a])
    else:
        frac = -V[a] / (V[b] - V[a])
        q0 = float(Q[a] - V[a] * (Q[b] - Q[a]) / (V[b] - V[a]))
    return dict(a=a, b=b, frac=float(frac), Q0=q0, n_crossings=len(events))


def sweep_charges(df: pd.DataFrame, sweep: pd.Series, find_crossing: bool) -> dict:
    """
    Charge record of one sweep: Q = Qcd - Qcd[first row], Q_end = Q in the
    last row and -- if `find_crossing` -- the interpolated Q0 at 0 V with the
    Excel rows a, b of the crossing and the current I0 interpolated there.
    Warns if |Qcd[first row]| > QCD_START_REL_TOL * |Qcd[last row]|.
    """
    seg = df.iloc[sweep.i0:sweep.i1 + 1]
    V = seg["V12"].to_numpy(dtype=float)
    I = seg["I12"].to_numpy(dtype=float)
    qcd = seg["Qcd"].to_numpy(dtype=float)
    rows = seg["excel_row"].to_numpy()
    Q = qcd - qcd[0]
    warnings = []
    if abs(qcd[0]) > cfg.QCD_START_REL_TOL * abs(qcd[-1]):
        msg = (f"Qcd at sweep start (Excel row {rows[0]}) = {qcd[0]:.5g} mAh is more than "
               f"{cfg.QCD_START_REL_TOL:.0%} of the end value {qcd[-1]:.5g} mAh")
        log.warning(msg)
        warnings.append("Qcd not ~0 at sweep start")

    rec = dict(
        cycle=int(sweep.cycle), step=int(sweep.step),
        row_start=int(sweep.row_start), row_end=int(sweep.row_end),
        Qcd_first=float(qcd[0]), Q_end=float(Q[-1]),
        row_a=np.nan, row_b=np.nan, V_a=np.nan, V_b=np.nan, Qcd_a=np.nan, Qcd_b=np.nan,
        Q0=np.nan, I0=np.nan, n_crossings=0,
        duration_s=float(sweep.duration_s), nu_meas_Vs=float(sweep.nu_meas_Vs),
        warnings=warnings,
    )
    if not find_crossing:
        return rec
    x = interpolate_zero_crossing(V, Q)
    if x is None:
        log.warning("Sweep Excel rows %d-%d: no 0 V crossing -> not evaluable",
                    sweep.row_start, sweep.row_end)
        rec["warnings"].append("no 0 V crossing")
        return rec
    a, b = x["a"], x["b"]
    if x["n_crossings"] > 1:
        log.warning("Sweep Excel rows %d-%d: %d zero crossings, the first (rows %d/%d) is used",
                    sweep.row_start, sweep.row_end, x["n_crossings"], rows[a], rows[b])
        rec["warnings"].append(f"{x['n_crossings']} 0 V crossings, first used")
    rec.update(
        row_a=int(rows[a]), row_b=int(rows[b]), V_a=float(V[a]), V_b=float(V[b]),
        Qcd_a=float(qcd[a]), Qcd_b=float(qcd[b]),
        Q0=x["Q0"], I0=float(I[a] + x["frac"] * (I[b] - I[a])),
        n_crossings=x["n_crossings"],
    )
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Charges per polarity
# ─────────────────────────────────────────────────────────────────────────────

def charges_symmetric(up: dict, down: dict) -> dict:
    """
    Four charges of a symmetric +-U cycle [mAh] (lab guide 3.1):
      Up sweep  (-U -> +U):  Q(-U->0) = Q0_up               discharge, polarity -
                             Q(0->+U) = Q_end_up - Q0_up    charge,    polarity +
      Down sweep (+U -> -U): Q(+U->0) = -Q0_down            discharge, polarity +
                             Q(0->-U) = Q0_down - Q_end_down charge,   polarity -
    Returns {"+": {Q_dis, Q_ch, I_ref}, "-": {...}}; I_ref is the current at
    the 0 V crossing of the discharge sweep (for the order-of-magnitude check).
    """
    return {
        "+": dict(Q_dis=-down["Q0"], Q_ch=up["Q_end"] - up["Q0"], I_ref=down["I0"]),
        "-": dict(Q_dis=up["Q0"], Q_ch=down["Q0"] - down["Q_end"], I_ref=up["I0"]),
    }


def charges_one_sided(up: dict, down: dict, df: pd.DataFrame, down_sweep: pd.Series) -> dict:
    """
    Charges of a one-sided 0 -> U cycle [mAh] (lab guide 3.2), one polarity:
      Q_ch  =  Q_end of the 0 -> U sweep
      Q_dis = -Q_end of the U -> 0 sweep
    I_ref = |I12| in the middle row of the discharge sweep (there is no 0 V
    crossing in a one-sided window).
    """
    mid = (down_sweep.i0 + down_sweep.i1) // 2
    return {"+": dict(Q_dis=-down["Q_end"], Q_ch=up["Q_end"],
                      I_ref=float(df["I12"].iat[mid]))}


# ─────────────────────────────────────────────────────────────────────────────
# Capacitance
# ─────────────────────────────────────────────────────────────────────────────

def evaluability(q_dis: float, q_ch: float) -> str | None:
    """None if evaluable, else the reason: Q_dis <= 0 or Q_dis < MIN_DISCHARGE_FRACTION * Q_ch."""
    if not np.isfinite(q_dis):
        return "not evaluable (no 0 V crossing)"
    if q_dis <= 0:
        return "not evaluable (Q_dis <= 0)"
    if np.isfinite(q_ch) and q_dis < cfg.MIN_DISCHARGE_FRACTION * q_ch:
        return f"not evaluable (Q_dis < {cfg.MIN_DISCHARGE_FRACTION:.0%} of Q_ch)"
    return None


def capacitance(q_dis: float, q_ch: float, u_max: float, m1_g: float, m2_g: float,
                evaluable: bool = True) -> dict:
    """
    Capacitance of one polarity from its DISCHARGE charge (lab guide 4):
      C         = MAH_TO_COULOMB * Q_dis[mAh] / U_max[V]   [F]  (discharge runs U_max -> 0)
      C_cell_sp = C / (m1 + m2)                            [F/g]
      C_el      = 4 * C / (m1 + m2) = 2 * C / m_mean       [F/g]  (two electrodes in series)
      C_el_exact= C * (1/m1 + 1/m2)  -- only if |m1-m2|/m_mean > MASS_ASYMMETRY_TOL
      eta       = Q_dis / Q_ch                             (same polarity; always reported)
    If not `evaluable`, C, C_cell_sp, C_el (and C_el_exact) are NaN.
    """
    m_mean = (m1_g + m2_g) / 2.0
    eta = q_dis / q_ch if (np.isfinite(q_ch) and q_ch != 0) else np.nan
    if not evaluable:
        return dict(C=np.nan, C_cell_sp=np.nan, C_el=np.nan, C_el_exact=np.nan, eta=eta)
    C = cfg.MAH_TO_COULOMB * q_dis / u_max
    asym = abs(m1_g - m2_g) / m_mean > cfg.MASS_ASYMMETRY_TOL
    return dict(
        C=C,
        C_cell_sp=C / (m1_g + m2_g),
        C_el=4.0 * C / (m1_g + m2_g),
        C_el_exact=C * (1.0 / m1_g + 1.0 / m2_g) if asym else np.nan,
        eta=eta,
    )


def evaluate_cycle(df: pd.DataFrame, sweeps: pd.DataFrame, cycle: pd.Series,
                   masses: dict) -> dict:
    """
    Charges and capacitances of one cycle. Block value C = (C+ + C-) / 2 for a
    symmetric window (NaN if one polarity is not evaluable), C = C+ for a
    one-sided window.
    """
    up_s, down_s = sweeps.loc[cycle.up], sweeps.loc[cycle.down]
    sym = cycle.window_type == SYMMETRIC
    up = sweep_charges(df, up_s, find_crossing=sym)
    down = sweep_charges(df, down_s, find_crossing=sym)
    charges = charges_symmetric(up, down) if sym else charges_one_sided(up, down, df, down_s)

    pol = {}
    for p, ch in charges.items():
        reason = evaluability(ch["Q_dis"], ch["Q_ch"])
        cap = capacitance(ch["Q_dis"], ch["Q_ch"], cycle.U_max,
                          masses["m1_g"], masses["m2_g"], evaluable=reason is None)
        pol[p] = dict(**ch, **cap, not_evaluable=reason)

    Cs = [pol[p]["C"] for p in pol]
    return dict(
        up=up, down=down, pol=pol,
        C=float(np.mean(Cs)) if all(np.isfinite(Cs)) else np.nan,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def quality_checks(res: dict, masses: dict) -> None:
    """
    Per-block checks (lab guide 5); appends flag strings to res["flags"]
    (block level) and res["pol"][p]["flags"] (polarity level), in place.
      - discharge charge positive (Q_dis > 0 and >= 5 % of Q_ch), else not evaluable
      - polarities agree: |C+ - C-| / C <= POLARITY_DEV_TOL
      - efficiency: eta >= EFFICIENCY_MIN (warning only)
      - stability: |C_last - C_prev| / C_last <= STABILITY_TOL  ("n/a" without previous cycle)
      - order of magnitude: C_est = |I_ref| / nu, 0.3 <= C / C_est <= 3
      - sweep duration within DURATION_REL_TOL of 2*U/nu (sym.) or U/nu (one-sided)
    The scan-rate trend needs all blocks: see scan_rate_trend_checks().
    """
    flags = res.setdefault("flags", [])
    flags.extend(res.get("notes", []))

    for p, d in res["pol"].items():
        pf = d.setdefault("flags", [])
        if d["not_evaluable"]:
            pf.append(d["not_evaluable"])
        if np.isfinite(d["eta"]) and d["eta"] < cfg.EFFICIENCY_MIN:
            pf.append(f"eta {d['eta']:.2f} < {cfg.EFFICIENCY_MIN}")
        nu = res["nu_meas_Vs"]
        if np.isfinite(d["C"]) and np.isfinite(d["I_ref"]) and d["I_ref"] != 0 and nu > 0:
            c_est = abs(d["I_ref"]) / 1000.0 / nu
            ratio = d["C"] / c_est
            d["C_est"] = c_est
            if not (cfg.MAGNITUDE_RATIO_MIN <= ratio <= cfg.MAGNITUDE_RATIO_MAX):
                pf.append(f"possible unit error (C/C_est = {ratio:.2g})")
        else:
            d["C_est"] = np.nan

    # polarity agreement
    res["pol_dev"] = np.nan
    if len(res["pol"]) == 2:
        cp, cm = res["pol"]["+"]["C"], res["pol"]["-"]["C"]
        if np.isfinite(cp) and np.isfinite(cm):
            dev = abs(cp - cm) / ((cp + cm) / 2.0)
            res["pol_dev"] = dev
            if dev > cfg.POLARITY_DEV_TOL:
                flags.append(f"polarity deviation {dev:.1%} > {cfg.POLARITY_DEV_TOL:.0%}")

    # stability
    res["stability"] = np.nan
    if res.get("C_prev") is None:
        res["stability_text"] = "n/a"
    elif np.isfinite(res["C"]) and np.isfinite(res["C_prev"]):
        st = abs(res["C"] - res["C_prev"]) / res["C"]
        res["stability"] = st
        res["stability_text"] = f"{st:.2%}"
        if st > cfg.STABILITY_TOL:
            flags.append(f"not stable: C changed {st:.1%} vs previous cycle")
    else:
        res["stability_text"] = "n/a (C not evaluable)"

    # sweep durations
    t_exp = expected_sweep_duration(res["window_type"], res["U_max"], res["nu_nom_mVs"])
    for name in ("up", "down"):
        dev = (res[name]["duration_s"] - t_exp) / t_exp
        if abs(dev) > cfg.DURATION_REL_TOL:
            flags.append(f"{name} sweep duration {dev:+.1%} vs expected {t_exp:.0f} s")
        for w in res[name]["warnings"]:
            flags.append(f"{name} sweep: {w}")


def evaluate_blocks(df: pd.DataFrame, sweeps: pd.DataFrame, cycles: pd.DataFrame,
                    masses: dict) -> list[dict]:
    """
    Evaluate every block (output of detect_blocks): select the last complete
    cycle, compute charges and capacitances for it and for the second-to-last
    complete cycle (stability), run quality_checks and, across blocks,
    scan_rate_trend_checks. Returns one dict per block in measurement order.
    """
    blocks = []
    for block_no, bc in cycles.dropna(subset=["block"]).groupby("block", sort=True):
        sel = select_last_cycle(bc, sweeps)
        last, prev = sel["last"], sel["prev"]
        res = evaluate_cycle(df, sweeps, last, masses)
        res.update(
            block=int(block_no), occurrence=int(last.occurrence),
            window_type=last.window_type, U_max=float(last.U_max),
            nu_nom_mVs=float(last.nu_nom_mVs), nu_meas_Vs=float(last.nu_meas_Vs),
            n_cycles=sel["n_cycles"], n_complete=sel["n_complete"], notes=sel["notes"],
            up_sweep=sweeps.loc[last.up], down_sweep=sweeps.loc[last.down],
            C_prev=evaluate_cycle(df, sweeps, prev, masses)["C"] if prev is not None else None,
        )
        C = res["C"]
        m1, m2 = masses["m1_g"], masses["m2_g"]
        res["C_cell_sp"] = C / (m1 + m2)
        res["C_el"] = 4.0 * C / (m1 + m2)
        res["C_el_exact"] = (C * (1 / m1 + 1 / m2)
                             if abs(m1 - m2) / masses["m_mean_g"] > cfg.MASS_ASYMMETRY_TOL
                             else np.nan)
        quality_checks(res, masses)
        blocks.append(res)
    scan_rate_trend_checks(blocks)
    for b in blocks:
        all_flags = b["flags"] + [f"{p}: {f}" for p, d in b["pol"].items() for f in d["flags"]]
        for f in all_flags:
            log.info("Block %d (%s %.2f V, %g mV/s): %s", b["block"], b["window_type"],
                     b["U_max"], b["nu_nom_mVs"], f)
    return blocks


def scan_rate_trend_checks(blocks: list[dict]) -> None:
    """
    Within one window (type, U_max), C must not increase with nu: a block is
    flagged if its C exceeds the C of the next slower scan rate (largest of
    its occurrences) by more than SCAN_RATE_TREND_TOL. Also marks the block
    whose capacitance is reported for the window: the slowest nu (its last
    occurrence).
    """
    by_window: dict[tuple, list[dict]] = {}
    for b in blocks:
        b["reported_for_window"] = False
        by_window.setdefault((b["window_type"], b["U_max"]), []).append(b)
    for group in by_window.values():
        rates = sorted({b["nu_nom_mVs"] for b in group})
        slowest = [b for b in group if b["nu_nom_mVs"] == rates[0]]
        slowest[-1]["reported_for_window"] = True
        for slow, fast in zip(rates[:-1], rates[1:]):
            ref = [b["C"] for b in group if b["nu_nom_mVs"] == slow and np.isfinite(b["C"])]
            if not ref:
                continue
            c_ref = max(ref)
            for b in group:
                if b["nu_nom_mVs"] == fast and np.isfinite(b["C"]):
                    rise = b["C"] / c_ref - 1.0
                    if rise > cfg.SCAN_RATE_TREND_TOL:
                        b["flags"].append(f"scan-rate trend: C {rise:+.1%} vs {slow:g} mV/s")
