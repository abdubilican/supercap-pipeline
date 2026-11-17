"""
gcd_analysis.py
---------------
Evaluation of GCD cycles (lab guide GCD sections 3-5): IR drop and ESR,
interpolation at U_min, charges/energies from Qcd and W12, capacitance,
energy, power, efficiencies, quality checks, rate summary, comparison with
CV, and the long-term (cycling / floating) evaluation.

All per-cycle quantities are computed in one vectorised pass over index
arrays (start/end row of every half-cycle, crossing rows a/b), so files
with tens of thousands of cycles need no Python loop over rows.

Units: I12 in mA, V12 in V, Qcd in mAh, W12 in mWh, t/t_step in s,
masses in g. m_mean = (m1 + m2) / 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import gcd_config as cfg
from .gcd_segmentation import CYCLING, FLOATING, HOLD, RATE, cluster_values

log = logging.getLogger("supercap_pipeline.gcd").getChild("analysis")

FLAG, WARNING, INFO = "flag", "warning", "info"


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation at U_min and IR drop
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_at(u, V_a, V_b, X_a, X_b):
    """
    Linear interpolation at the voltage u (lab guide 3.2):
        X(u) = X_a + (u - V_a) * (X_b - X_a) / (V_b - V_a)
    used for X = Qcd, W12 and t_step between rows a and b of the U_min crossing.
    """
    return X_a + (u - V_a) * (X_b - X_a) / (V_b - V_a)


def find_crossing(V: np.ndarray, u_min: float) -> tuple[int, int] | None:
    """
    U_min crossing of one discharge (lab guide 3.2): b = the first row with
    V12 <= U_min, a = b - 1 (the last row with V12 > U_min before it).
    Rows after b are ignored (e.g. points slightly below U_min at the end).
    Returns (a, b) as positions in V, or None if the discharge never reaches
    U_min (or already starts below it).
    """
    hit = np.flatnonzero(np.asarray(V, dtype=float) <= u_min)
    if len(hit) == 0 or hit[0] == 0:
        return None
    return int(hit[0] - 1), int(hit[0])


def ir_drop(U_before: float, U1: float, I_before_mA: float, I_dis_mA: float) -> tuple[float, float]:
    """
    IR drop and ESR at the start of the discharge (lab guide 3.1):
        dU_IR = U_before - U1
        ESR   = dU_IR / dI,   dI = |I_before - I_discharge| [A]
    U_before = V12 in the last row before the discharge (end of the charge,
    or end of a rest step), U1 = V12 in the first discharge row. With direct
    switching (no rest) I_before = +|I| and dI = 2|I|; after a rest
    I_before = 0 and dI = |I|. I_discharge = median I12 of the discharge.
    Returns (dU_IR [V], ESR [Ohm]).
    """
    du = U_before - U1
    di = abs(I_before_mA - I_dis_mA) / 1000.0
    return du, du / di if di > 0 else float("nan")


def _segments(i0: np.ndarray, i1: np.ndarray):
    """Flattened row index of the segments [i0, i1]: (idx, seg, pos, offsets)."""
    L = i1 - i0 + 1
    offs = np.cumsum(L) - L
    seg = np.repeat(np.arange(len(L)), L)
    pos = np.arange(int(L.sum())) - np.repeat(offs, L)
    return np.repeat(i0, L) + pos, seg, pos, offs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of every cycle
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cycles(df: pd.DataFrame, cycles: pd.DataFrame, masses: dict) -> pd.DataFrame:
    """
    Section 3 values for every cycle (vectorised):
      3.1  U_before, U1, t_U1 [ms], dU_IR = U_before - U1, ESR = dU_IR / dI
      3.2  rows a/b of the U_min crossing, Qcd/W12/t_step interpolated at U_min
      3.3  Q_ch = Qcd(last charge row); Q_dis = -Qcd(U_min); t_dis = t_step(U_min)
           E_ch = W12(last charge row) - W12(first charge row)
           E_dis = W12(last row before discharge) - W12(U_min)
      3.4  C   = 3.6 * Q_dis / (U1 - U_min)                      [F]
           C_E = 2 * 3.6 * E_dis / (U1^2 - U_min^2)               [F]
           C_cell,sp = C / (2 m_mean), C_el = 2 C / m_mean        [F/g]  (same for C_E)
           C_slope (info) = |I| / |dU/dt|, fit between 80 % and 40 % of (U1 - U_min)
      3.5  E_sp = E_dis / (2 m_mean)                              [Wh/kg]
           P_sp = E_dis * 3600 / t_dis / (2 m_mean)               [W/kg]
           P_max,sp = U_max^2 / (4 ESR) / (2 m_mean) * 1000       [W/kg] (theoretical, matched load)
           eta_C = Q_dis / Q_ch, eta_E = E_dis / E_ch
           I_leak = (Q_ch - Q_dis) * 3600 / (t_ch + t_dis)        [mA]
    `used` = complete cycle whose discharge reaches U_min.
    """
    V = df["V12"].to_numpy(dtype=float)
    I = df["I12"].to_numpy(dtype=float)
    Q = df["Qcd"].to_numpy(dtype=float)
    W = df["W12"].to_numpy(dtype=float)
    TS = df["t_step"].to_numpy(dtype=float)
    ROWS = df["excel_row"].to_numpy()
    m = masses["m_mean_g"]

    c = cycles.copy()
    c0, c1 = c["ch_i0"].to_numpy(), c["ch_i1"].to_numpy()
    d0, d1 = c["dis_i0"].to_numpy(), c["dis_i1"].to_numpy()
    umin = c["U_min"].to_numpy(dtype=float)
    K = len(c)

    # discharge rows, U_min crossing (b = first row <= U_min, a = b - 1)
    idx, seg, pos, offs = _segments(d0, d1)
    BIG = np.iinfo(np.int64).max
    first = np.minimum.reduceat(np.where(V[idx] <= umin[seg], pos, BIG), offs)
    cross = (first != BIG) & (first > 0)
    bb = np.where(cross, d0 + np.where(cross, first, 0), d0)
    aa = np.where(cross, bb - 1, d0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Q_u = np.where(cross, interpolate_at(umin, V[aa], V[bb], Q[aa], Q[bb]), np.nan)
        W_u = np.where(cross, interpolate_at(umin, V[aa], V[bb], W[aa], W[bb]), np.nan)
        t_u = np.where(cross, interpolate_at(umin, V[aa], V[bb], TS[aa], TS[bb]), np.nan)
    I_dis = pd.Series(I[idx]).groupby(seg).median().to_numpy()

    # 3.1 IR drop / ESR
    pre = d0 - 1
    U_before, I_before = V[pre], I[pre]
    U1 = V[d0]
    dU = U_before - U1
    dI = np.abs(I_before - I_dis) / 1000.0

    # 3.3
    Q_ch, t_ch = Q[c1], TS[c1]
    E_ch = W[c1] - W[c0]
    Q_dis, t_dis = -Q_u, t_u
    E_dis = W[pre] - W_u

    with np.errstate(invalid="ignore", divide="ignore"):
        ESR = dU / dI
        C = cfg.MAH_TO_COULOMB * Q_dis / (U1 - umin)
        C_E = 2.0 * cfg.MAH_TO_COULOMB * E_dis / (U1 ** 2 - umin ** 2)
        P_sp = E_dis * 3600.0 / t_dis / (2.0 * m)
        P_max = c["U_max"].to_numpy() ** 2 / (4.0 * ESR) / (2.0 * m) * 1000.0
        eta_C = Q_dis / Q_ch
        eta_E = E_dis / E_ch
        I_leak = (Q_ch - Q_dis) * 3600.0 / (t_ch + t_dis)

        # optional info column: slope between 80 % and 40 % of (U1 - U_min) above U_min
        span = (U1 - umin)[seg]
        vv, xx = V[idx], TS[idx]
        w = ((vv <= umin[seg] + cfg.SLOPE_FIT_UPPER * span)
             & (vv >= umin[seg] + cfg.SLOPE_FIT_LOWER * span)
             & (pos < first[seg])).astype(float)
        n = np.bincount(seg, w, K)
        sx, sy = np.bincount(seg, w * xx, K), np.bincount(seg, w * vv, K)
        sxx, sxy = np.bincount(seg, w * xx * xx, K), np.bincount(seg, w * xx * vv, K)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        C_slope = np.where(n >= cfg.SLOPE_FIT_MIN_POINTS,
                           np.abs(I_dis) / 1000.0 / np.abs(slope), np.nan)

    j = c["I_abs_mA"].to_numpy() / (1000.0 * m)
    c = c.assign(
        j_Ag=j, j_block_Ag=c["I_block_mA"].to_numpy() / (1000.0 * m),
        U_before=U_before, I_before_mA=I_before, U1=U1, t_U1_ms=TS[d0] * 1000.0,
        I_dis_mA=I_dis, dU_IR=dU, dI_A=dI, ESR=ESR,
        row_a=np.where(cross, ROWS[aa], -1), row_b=np.where(cross, ROWS[bb], -1),
        Q_ch=Q_ch, Q_dis=Q_dis, t_ch=t_ch, t_dis=t_dis, E_ch=E_ch, E_dis=E_dis,
        C=C, C_E=C_E,
        C_cell_sp=C / (2.0 * m), C_el=2.0 * C / m,
        C_E_cell_sp=C_E / (2.0 * m), C_E_el=2.0 * C_E / m,
        C_slope=C_slope,
        E_sp=E_dis / (2.0 * m), P_sp=P_sp, P_max_sp=P_max,
        eta_C=eta_C, eta_E=eta_E, I_leak_mA=I_leak,
        reaches_U_min=cross,
    )
    c["used"] = c["complete"] & c["reaches_U_min"]
    for _, r in c[c["complete"] & ~c["reaches_U_min"]].iterrows():
        log.warning("Cycle %d (Excel rows %d-%d): discharge incomplete -- never reaches U_min = "
                    "%.2f V; not used", r.cycle, r.row_dis_start, r.row_dis_end, r.U_min)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def nearest_standard_j(j: float) -> tuple[float, bool]:
    """Nearest STANDARD_J value and whether j is within STANDARD_J_REL_TOL of it."""
    std = np.asarray(cfg.STANDARD_J, dtype=float)
    k = int(np.argmin(np.abs(std - j) / std))
    return float(std[k]), abs(j - std[k]) <= cfg.STANDARD_J_REL_TOL * std[k]


def cycle_checks(ev: pd.DataFrame) -> pd.Series:
    """
    Per-cycle checks (lab guide 4), as a list of (severity, text) per cycle:
      not complete / discharge incomplete; t_U1 > T_U1_MAX_MS "IR drop read late";
      eta_C < ETA_C_WARN (warning), < ETA_C_FLAG "low efficiency";
      |C - C_E| / C > C_VS_CE_TOL (curve shape); dU_IR / U_max > IR_DROP_MAX_FRACTION
      "resistance-limited"; fewer than MIN_DISCHARGE_POINTS discharge points
      "logging too coarse" (warning).
    Cycles that are not used get only the reason (no further checks).
    """
    out = []
    for r in ev.itertuples(index=False):
        f = []
        if not r.complete:
            out.append([(FLAG, f"{r.incomplete_reason} - not used")])
            continue
        if not r.reaches_U_min:
            out.append([(FLAG, "discharge incomplete - not used")])
            continue
        if r.t_U1_ms > cfg.T_U1_MAX_MS:
            f.append((FLAG, f"IR drop read late (t_U1 = {r.t_U1_ms:.1f} ms > {cfg.T_U1_MAX_MS:g} ms)"))
        if r.eta_C < cfg.ETA_C_WARN:
            f.append((WARNING, f"eta_C = {r.eta_C:.3f} < {cfg.ETA_C_WARN}"))
        if r.eta_C < cfg.ETA_C_FLAG:
            f.append((FLAG, f"low efficiency (eta_C = {r.eta_C:.3f} < {cfg.ETA_C_FLAG})"))
        dev = abs(r.C - r.C_E) / r.C
        if dev > cfg.C_VS_CE_TOL:
            f.append((FLAG, f"curve shape: |C - C_E| / C = {dev:.1%} > {cfg.C_VS_CE_TOL:.0%}"))
        ir = r.dU_IR / r.U_max
        if ir > cfg.IR_DROP_MAX_FRACTION:
            f.append((FLAG, f"resistance-limited (dU_IR / U_max = {ir:.1%} > "
                            f"{cfg.IR_DROP_MAX_FRACTION:.0%})"))
        if r.n_dis < cfg.MIN_DISCHARGE_POINTS:
            f.append((WARNING, f"logging too coarse ({r.n_dis} points in the discharge < "
                               f"{cfg.MIN_DISCHARGE_POINTS})"))
        out.append(f)
    return pd.Series(out, index=ev.index, dtype=object)


# ─────────────────────────────────────────────────────────────────────────────
# Rate series
# ─────────────────────────────────────────────────────────────────────────────

def rate_summary(ev: pd.DataFrame, selection: dict[int, dict]) -> tuple[pd.DataFrame, list]:
    """
    One row per rate block = its selected (last complete) cycle (lab guide 2/6.1),
    with the second-to-last cycle's C, the stability value
    |C_last - C_prev| / C_last (flag > STABLE_CYCLE_TOL; single cycle flagged),
    the current-density check against STANDARD_J, and the retention
    C / C(lowest j) [%]. Returns (table, series_flags) where series_flags
    holds the ESR-constancy check of the whole rate series.
    """
    rows = []
    rate_blocks = ev[ev["test_type"] == RATE].groupby("block")
    by_cycle = ev.set_index("cycle")
    for b, g in rate_blocks:
        sel = selection.get(int(b))
        first = g.iloc[0]
        if sel is None:
            rows.append({"block": int(b), "occurrence": int(first.occurrence),
                         "j_Ag": first.j_block_Ag, "I_mA": first.I_block_mA,
                         "n_cycles": len(g), "n_complete": 0, "selected_cycle": np.nan,
                         "block_flags": [(FLAG, "no complete cycle - block not evaluated")]})
            log.warning("Rate block %d: no complete cycle, not evaluated", b)
            continue
        r = by_cycle.loc[sel["last"]]
        flags = []
        std, ok = nearest_standard_j(r.j_block_Ag)
        if not ok:
            flags.append((FLAG, f"non-standard current density, check program and current range "
                                f"(j = {r.j_block_Ag:.3g} A/g, nearest standard {std:g} A/g)"))
        if sel["prev"] is None:
            c_prev, stab = np.nan, np.nan
            flags.append((FLAG, "single complete cycle - stability not checked"))
        else:
            c_prev = by_cycle.loc[sel["prev"], "C"]
            stab = abs(r.C - c_prev) / r.C
            if stab > cfg.STABLE_CYCLE_TOL:
                flags.append((FLAG, f"not stable: C changed {stab:.2%} vs previous cycle "
                                    f"(> {cfg.STABLE_CYCLE_TOL:.0%})"))
        rows.append({**r.to_dict(), "cycle": sel["last"], "selected_cycle": sel["last"],
                     "prev_cycle": sel["prev"], "C_prev": c_prev, "stability": stab,
                     "n_cycles": len(g), "n_complete": sel["n_complete"],
                     "j_standard_Ag": std, "block_flags": flags})
    rate = pd.DataFrame(rows)
    series_flags = []
    if rate.empty or "C" not in rate:
        return rate, series_flags
    ok = rate["C"].notna()
    if ok.any():
        ref = rate[ok].sort_values(["j_block_Ag", "block"]).iloc[0]
        rate["retention_pct"] = 100.0 * rate["C"] / ref["C"]
        rate["retention_ref_block"] = int(ref["block"])
        esr = rate.loc[ok, "ESR"]
        spread = (esr.max() - esr.min()) / esr.min()
        rate.attrs["esr_spread"] = spread
        if spread > cfg.ESR_SPREAD_TOL:
            series_flags.append((FLAG, f"ESR not constant in the rate series: (max - min) / min = "
                                       f"{spread:.1%} > {cfg.ESR_SPREAD_TOL:.0%}"))
        else:
            series_flags.append((INFO, f"ESR spread in the rate series {spread:.1%} "
                                       f"(<= {cfg.ESR_SPREAD_TOL:.0%})"))
    return rate, series_flags


def compare_with_cv(sample_dir: Path, rate: pd.DataFrame) -> dict | None:
    """
    Comparison with CV (lab guide 4): C_el of the rate block with the lowest
    j vs the CV C_el at the lowest scan rate in the matching window -- the
    one-sided window 0->U_max, or +-U_max if there is no one-sided window --
    read from <sample_dir>/cv/results.xlsx (sheet Summary).
    Deviation = (C_el,GCD - C_el,CV) / C_el,CV; > CV_COMPARE_TOL -> flag.
    Returns None if there are no CV results or no matching window.
    """
    cv_file = sample_dir / "cv" / "results.xlsx"
    if rate.empty or "C_el" not in rate or not cv_file.exists():
        return None
    ok = rate[rate["C_el"].notna()]
    if ok.empty:
        return None
    g = ok.sort_values(["j_block_Ag", "block"]).iloc[0]
    try:
        cv = pd.read_excel(cv_file, sheet_name="Summary")
    except (OSError, ValueError) as exc:
        log.warning("CV results %s not readable: %s", cv_file, exc)
        return None
    u = round(float(g["U_max_block"]), 2)
    match = cv[(cv["U_max_V"] - u).abs() <= 0.005]
    one = match[match["window_type"] == "one-sided"]
    sel, wtxt = (one, "0->U_max") if len(one) else (match[match["window_type"] == "symmetric"],
                                                       "+-U_max")
    if sel.empty:
        log.info("CV results found but no window with U_max = %.2f V for the comparison", u)
        return None
    cvrow = sel[sel["nu_nom_mVs"] == sel["nu_nom_mVs"].min()].iloc[-1]
    res = dict(cv_file=str(cv_file), cv_window=f"{wtxt} = {u:g} V", cv_nu_mVs=cvrow["nu_nom_mVs"],
               cv_C_el=cvrow["C_el_Fg"], gcd_j=g["j_block_Ag"], gcd_block=int(g["block"]),
               gcd_C_el=g["C_el"])
    note = f"CV window {wtxt}, GCD window 0->U_max"
    if not np.isfinite(res["cv_C_el"]):
        res.update(deviation=np.nan, severity=INFO,
                   text=f"CV C_el at {wtxt}, {res['cv_nu_mVs']:g} mV/s not evaluable; {note}")
        return res
    dev = (res["gcd_C_el"] - res["cv_C_el"]) / res["cv_C_el"]
    res["deviation"] = dev
    res["severity"] = FLAG if abs(dev) > cfg.CV_COMPARE_TOL else INFO
    res["text"] = (f"GCD C_el at {res['gcd_j']:.3g} A/g = {res['gcd_C_el']:.1f} F/g vs CV "
                   f"{res['cv_C_el']:.1f} F/g ({wtxt}, {res['cv_nu_mVs']:g} mV/s): {dev:+.1%}"
                   f"{' > ' + format(cfg.CV_COMPARE_TOL, '.0%') if abs(dev) > cfg.CV_COMPARE_TOL else ''}"
                   f" ({note})")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Long-term tests
# ─────────────────────────────────────────────────────────────────────────────

def end_of_life(x: np.ndarray, retention: np.ndarray, esr_rel: np.ndarray, start: int) -> dict:
    """
    End of life (lab guide 5.1): the first point after the reference at which
    retention < EOL_RETENTION or ESR/ESR0 > EOL_ESR; the x value (cycle number
    or hold time) is interpolated linearly between that point and the one
    before it. Returns dict(eol_x, criterion, index) or criterion "not reached".
    """
    bad_r = retention < cfg.EOL_RETENTION
    bad_e = esr_rel > cfg.EOL_ESR
    cand = np.flatnonzero((bad_r | bad_e) & (np.arange(len(x)) > start))
    if len(cand) == 0:
        return dict(eol_x=np.nan, criterion="not reached", index=-1)
    i = int(cand[0])
    hits = []
    for name, y, thr, bad in (("retention < 80 %", retention, cfg.EOL_RETENTION, bad_r),
                              ("ESR/ESR0 > 2", esr_rel, cfg.EOL_ESR, bad_e)):
        if bad[i]:
            if i > 0 and y[i] != y[i - 1]:
                xe = x[i - 1] + (thr - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
            else:
                xe = x[i]
            hits.append((xe, name))
    xe, name = min(hits)
    if cfg.EOL_RETENTION != 0.8 or cfg.EOL_ESR != 2.0:
        name = name.replace("80 %", f"{cfg.EOL_RETENTION:.0%}").replace("> 2", f"> {cfg.EOL_ESR:g}")
    return dict(eol_x=float(xe), criterion=name, index=i)


def longterm_cycling(ev: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Cycling test (lab guide 5.1), per cycling block: every used cycle;
    reference C0/ESR0 = the first cycle whose C differs from the next by
    less than STABILISED_TOL (cell stabilised), else cycle 2 of the block;
    retention = C/C0 [%], ESR increase = ESR/ESR0; end of life vs cycle
    number in the block. Returns (table, one summary dict per block).
    """
    tables, summaries = [], []
    for b, g in ev[(ev["test_type"] == CYCLING) & ev["used"]].groupby("block"):
        g = g.sort_values("cycle_in_block")
        C, esr = g["C"].to_numpy(), g["ESR"].to_numpy()
        x = g["cycle_in_block"].to_numpy(dtype=float)
        rel = np.abs(np.diff(C)) / C[:-1]
        stab = np.flatnonzero(rel < cfg.STABILISED_TOL)
        if len(stab):
            ref = int(stab[0])
            how = "first stabilised cycle (|C - C_next| / C < 1 %)"
        else:
            ref = int(np.flatnonzero(x == 2)[0]) if (x == 2).any() else 0
            how = "fallback: cycle 2 of the block"
        retention = C / C[ref]
        esr_rel = esr / esr[ref]
        eol = end_of_life(x, retention, esr_rel, ref)
        log.info("Cycling block %d: reference cycle %d of the block (%s); end of life: %s",
                 b, int(x[ref]), how,
                 "not reached" if eol["criterion"] == "not reached"
                 else f"cycle {eol['eol_x']:.1f} ({eol['criterion']})")
        t = g[["block", "cycle", "cycle_in_block", "instr_cycle", "j_Ag", "C", "C_el", "ESR",
               "eta_C", "eta_E"]].copy()
        t["retention_pct"] = 100.0 * retention
        t["ESR_rel"] = esr_rel
        t["reference_cycle_in_block"] = int(x[ref])
        t["eol_cycle"] = eol["eol_x"]
        t["eol_criterion"] = eol["criterion"]
        tables.append(t)
        summaries.append(dict(block=int(b), j=float(g["j_block_Ag"].iloc[0]), n_used=len(g),
                              n_cycles=int(g["n_cycles_block"].iloc[0]),
                              reference=int(x[ref]), reference_how=how, C0=float(C[ref]),
                              ESR0=float(esr[ref]), **{f"eol_{k}": v for k, v in eol.items()}))
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), summaries


def holds_table(df: pd.DataFrame, steps: pd.DataFrame, masses: dict) -> pd.DataFrame:
    """
    Per voltage hold (lab guide 5.2): rows, hold voltage (median V12),
    duration [h], cumulative hold time [h], and the mean leakage current over
    the last HOLD_LEAK_WINDOW_S of the hold [mA and mA/g per m_mean].
    """
    h = steps[steps["kind"] == HOLD]
    if h.empty:
        return pd.DataFrame()
    t = df["t"].to_numpy(dtype=float)
    I = df["I12"].to_numpy(dtype=float)
    leak = []
    for s in h.itertuples():
        sl = slice(s.i0, s.i1 + 1)
        last = t[sl] >= t[s.i1] - cfg.HOLD_LEAK_WINDOW_S
        leak.append(float(np.mean(I[sl][last])))
    out = pd.DataFrame({
        "hold": np.arange(1, len(h) + 1),
        "row_start": h["row_start"].to_numpy(), "row_end": h["row_end"].to_numpy(),
        "type": h["type"].to_numpy(), "con": h["con"].to_numpy(),
        "V_hold": h["V_median"].to_numpy(),
        "duration_h": h["duration_s"].to_numpy() / 3600.0,
        "leak_mA": leak,
    })
    out["leak_mA_per_g"] = out["leak_mA"] / masses["m_mean_g"]
    out["cumulative_h"] = out["duration_h"].cumsum()
    return out


def longterm_floating(ev: pd.DataFrame, holds: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Floating test (lab guide 5.2): each check-up = the CC cycles between two
    holds; for every current density in a check-up its last complete cycle.
    x = cumulative hold time [h] at the end of the preceding hold (0 for the
    check-up before the first hold). Reference = first check-up (per j);
    retention, ESR increase and end of life vs hold time as in 5.1.
    """
    fl = ev[(ev["test_type"] == FLOATING) & ev["used"]].copy()
    if fl.empty:
        return pd.DataFrame(), []
    cum = np.r_[0.0, holds["cumulative_h"].to_numpy()] if not holds.empty else np.array([0.0])
    fl["j_group"] = cluster_values(fl["j_block_Ag"].to_numpy())
    last = fl.sort_values("cycle").groupby(["checkup", "j_group"]).tail(1)
    tables, summaries = [], []
    for jg, g in last.groupby("j_group"):
        g = g.sort_values("checkup")
        x = cum[np.minimum(g["checkup"].to_numpy(), len(cum) - 1)]
        C, esr = g["C"].to_numpy(), g["ESR"].to_numpy()
        retention, esr_rel = C / C[0], esr / esr[0]
        eol = end_of_life(x, retention, esr_rel, 0)
        j = float(g["j_block_Ag"].median())
        log.info("Floating, j = %.3g A/g: %d check-ups; end of life: %s", j, len(g),
                 "not reached" if eol["criterion"] == "not reached"
                 else f"{eol['eol_x']:.1f} h ({eol['criterion']})")
        t = g[["checkup", "block", "cycle", "instr_cycle", "j_block_Ag", "C", "C_el", "ESR",
               "eta_C", "eta_E"]].copy()
        t["hold_time_h"] = x
        t["retention_pct"] = 100.0 * retention
        t["ESR_rel"] = esr_rel
        t["eol_hold_time_h"] = eol["eol_x"]
        t["eol_criterion"] = eol["criterion"]
        tables.append(t)
        summaries.append(dict(j=j, n_checkups=len(g), C0=float(C[0]), ESR0=float(esr[0]),
                              **{f"eol_{k}": v for k, v in eol.items()}))
    return pd.concat(tables, ignore_index=True), summaries
