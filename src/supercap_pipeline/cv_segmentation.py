"""
cv_segmentation.py
------------------
Universal segmentation of a CV measurement (lab guide section 2): sweeps,
transition steps, scan rates, cycles, windows, blocks and the cycle to
evaluate. Everything is derived from the data; the instrument's cycle and
step numbers are carried along for traceability only.

Data flow:
    find_sweeps(df) -> drop_transition_sweeps() -> pair_cycles()
                    -> detect_blocks() -> select_last_cycle() per block
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from . import cv_config as cfg

log = logging.getLogger("supercap_pipeline.cv").getChild("segmentation")

SYMMETRIC = "symmetric"
ONE_SIDED = "one-sided"
OTHER = "other"


def vertex_tolerance(v_max: float) -> float:
    """Vertex tolerance = max(WINDOW_REL_TOL * |V_max|, WINDOW_ABS_TOL_V)  [V]."""
    return max(cfg.WINDOW_REL_TOL * abs(v_max), cfg.WINDOW_ABS_TOL_V)


def round_sig(x: float, sig: int = cfg.SCAN_RATE_SIG_FIGS) -> float:
    """Round x to `sig` significant figures (NaN for x <= 0)."""
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    return round(x, sig - 1 - int(math.floor(math.log10(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────────────────────

def find_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    A sweep is a contiguous run of rows with the same step number and
    type "VS" (a reset of t_step also starts a new run).

    Per sweep: positional rows i0..i1 in `df`, Excel rows, instrument cycle
    and step, V_start, V_end, V_min, V_max, direction = sign(V_end - V_start)
    (cross-checked with the c/d column), number of points, duration = last
    t_step, measured scan rate nu = |slope| of a linear fit V12(t_step) over
    the middle SCAN_RATE_FIT_FRACTION of the sweep [V/s], and the nominal
    scan rate = nu in mV/s rounded to SCAN_RATE_SIG_FIGS significant figures.
    """
    step = df["step"].to_numpy()
    is_vs = (df["type"] == "VS").to_numpy()
    t_step = df["t_step"].to_numpy(dtype=float)
    V = df["V12"].to_numpy(dtype=float)
    cd = df["c/d"].to_numpy()
    rows = df["excel_row"].to_numpy()
    cyc = df["cycle"].to_numpy()

    n = len(df)
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = (step[1:] != step[:-1]) | (is_vs[1:] != is_vs[:-1]) | (np.diff(t_step) < 0)
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:] - 1, n - 1)

    out = []
    for i0, i1 in zip(starts, ends):
        if not is_vs[i0]:
            continue
        v = V[i0:i1 + 1]
        ts = t_step[i0:i1 + 1]
        npts = len(v)
        direction = int(np.sign(v[-1] - v[0]))
        cd_vals = set(cd[i0:i1 + 1]) - {""}
        cd_expected = {1: "c", -1: "d"}.get(direction)
        nu = _fit_scan_rate(ts, v)
        out.append(dict(
            i0=int(i0), i1=int(i1),
            row_start=int(rows[i0]), row_end=int(rows[i1]),
            cycle=int(cyc[i0]), step=int(step[i0]),
            V_start=float(v[0]), V_end=float(v[-1]),
            V_min=float(v.min()), V_max=float(v.max()),
            dV=float(v[-1] - v[0]),
            direction=direction,
            cd=",".join(sorted(cd_vals)),
            cd_ok=(cd_vals == {cd_expected}) if cd_vals else True,
            n_points=npts,
            duration_s=float(ts[-1]),
            nu_meas_Vs=nu,
            nu_nom_mVs=round_sig(nu * 1000.0),
        ))
    sweeps = pd.DataFrame(out)
    if sweeps.empty:
        raise ValueError("No voltage sweeps (type 'VS') found in the export.")

    bad_cd = sweeps[~sweeps["cd_ok"] & (sweeps["n_points"] >= cfg.MIN_SWEEP_POINTS)]
    for _, s in bad_cd.iterrows():
        log.warning("Sweep at Excel rows %d-%d: direction %+d from V12 disagrees with c/d = '%s'",
                    s.row_start, s.row_end, s.direction, s.cd)
    log.info("Found %d voltage sweeps", len(sweeps))
    return sweeps


def _fit_scan_rate(t: np.ndarray, v: np.ndarray) -> float:
    """|dV/dt| from a least-squares line V12 = a + nu*t_step over the middle part [V/s]."""
    n = len(v)
    cut = int(round(n * (1 - cfg.SCAN_RATE_FIT_FRACTION) / 2))
    sel = slice(cut, n - cut) if n - 2 * cut >= 2 else slice(0, n)
    tt, vv = t[sel], v[sel]
    if len(tt) < 2 or np.ptp(tt) == 0:
        return float("nan")
    return float(abs(np.polyfit(tt, vv, 1)[0]))


def drop_transition_sweeps(sweeps: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transition steps: sweeps with fewer than MIN_SWEEP_POINTS points,
    or whose voltage change |V_end - V_start| is below
    TRANSITION_MIN_FRACTION x the window (V_max - V_min) of the neighbouring
    sweeps (the larger of previous and next). The number dropped is logged.
    """
    span = (sweeps["V_max"] - sweeps["V_min"]).to_numpy()
    prev = np.r_[np.nan, span[:-1]]
    nxt = np.r_[span[1:], np.nan]
    ref = np.fmax(prev, nxt)
    ref = np.where(np.isnan(ref), span, ref)
    too_short = sweeps["n_points"].to_numpy() < cfg.MIN_SWEEP_POINTS
    too_small = sweeps["dV"].abs().to_numpy() < cfg.TRANSITION_MIN_FRACTION * ref
    drop = too_short | too_small

    dropped = sweeps[drop]
    log.info("Dropped %d transition sweep(s) (< %d points or |dV| < %.0f %% of the neighbours' window)",
             len(dropped), cfg.MIN_SWEEP_POINTS, cfg.TRANSITION_MIN_FRACTION * 100)
    for _, s in dropped.iterrows():
        log.info("  dropped sweep Excel rows %d-%d (cycle %d, step %d, %d pts, dV = %.4f V)",
                  s.row_start, s.row_end, s.cycle, s.step, s.n_points, s.dV)
    kept = sweeps[~drop].reset_index(drop=True)
    kept.attrs["n_dropped"] = int(drop.sum())
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Cycles
# ─────────────────────────────────────────────────────────────────────────────

def classify_window(v_min: float, v_max: float) -> tuple[str, float]:
    """
    Window type from the cycle's V_min, V_max (tol = vertex_tolerance(V_max)):
      symmetric +-U : |V_min + V_max| <= tol,  U_max = V_max
      one-sided 0->U: |V_min| <= tol,          U_max = V_max
      otherwise     : "other" (skipped)
    U_max is rounded to U_MAX_ROUND_V for grouping.
    """
    tol = vertex_tolerance(v_max)
    u = round(round(v_max / cfg.U_MAX_ROUND_V) * cfg.U_MAX_ROUND_V, 6)
    if v_max > 0 and abs(v_min + v_max) <= tol:
        return SYMMETRIC, u
    if v_max > 0 and abs(v_min) <= tol:
        return ONE_SIDED, u
    return OTHER, u


def pair_cycles(sweeps: pd.DataFrame) -> pd.DataFrame:
    """
    A cycle is a pair of consecutive sweeps in opposite directions with the
    same window (V_min and V_max agree within the vertex tolerance) and the
    same nominal scan rate. Where two pairings are possible, sweeps that
    directly follow each other (no transition step or rest in between) are
    paired. Sweeps that cannot be paired are logged and
    ignored. Cycles with a window that is neither symmetric nor one-sided
    are kept with window_type "other" and a warning (they are not evaluated).
    """
    cycles = []
    i = 0
    n = len(sweeps)
    while i < n:
        a = sweeps.iloc[i]
        if i + 1 < n and _pairable(a, sweeps.iloc[i + 1]) and not (
            # prefer directly adjacent sweeps: if a and the next sweep are separated
            # (e.g. by a transition step) but the next one pairs with its own
            # immediate neighbour, a belongs to the previous cycle
            not _adjacent(a, sweeps.iloc[i + 1]) and i + 2 < n
            and _adjacent(sweeps.iloc[i + 1], sweeps.iloc[i + 2])
            and _pairable(sweeps.iloc[i + 1], sweeps.iloc[i + 2])
        ):
            b = sweeps.iloc[i + 1]
            up, down = (a, b) if a.direction > 0 else (b, a)
            v_min = min(a.V_min, b.V_min)
            v_max = max(a.V_max, b.V_max)
            wtype, u_max = classify_window(v_min, v_max)
            if wtype == OTHER:
                log.warning("Cycle at Excel rows %d-%d skipped: window %.3f..%.3f V is neither "
                            "symmetric nor one-sided", a.row_start, b.row_end, v_min, v_max)
            cycles.append(dict(
                first=i, second=i + 1,
                up=int(up.name), down=int(down.name),
                V_min=v_min, V_max=v_max,
                window_type=wtype, U_max=u_max,
                nu_nom_mVs=a.nu_nom_mVs,
                nu_meas_Vs=float(np.mean([a.nu_meas_Vs, b.nu_meas_Vs])),
                row_start=int(a.row_start), row_end=int(b.row_end),
            ))
            i += 2
        else:
            log.info("Unpaired sweep at Excel rows %d-%d (%.3f -> %.3f V, %g mV/s) ignored",
                     a.row_start, a.row_end, a.V_start, a.V_end, a.nu_nom_mVs)
            i += 1
    return pd.DataFrame(cycles)


def _adjacent(a: pd.Series, b: pd.Series) -> bool:
    """b starts in the row directly after a ends (nothing in between)."""
    return b.i0 == a.i1 + 1


def _pairable(a: pd.Series, b: pd.Series) -> bool:
    if a.direction == 0 or a.direction != -b.direction:
        return False
    if not (a.nu_nom_mVs == b.nu_nom_mVs):
        return False
    tol = vertex_tolerance(max(abs(a.V_max), abs(b.V_max), abs(a.V_min), abs(b.V_min)))
    return abs(a.V_min - b.V_min) <= tol and abs(a.V_max - b.V_max) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# Blocks
# ─────────────────────────────────────────────────────────────────────────────

def detect_blocks(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    A block is a run of consecutive cycles with the same window type, U_max
    and nominal scan rate. A combination that occurs again later (not
    consecutively) is a new block; `occurrence` counts its repetitions.
    Adds the columns `block` (1, 2, ...; NaN for skipped "other" cycles)
    and `occurrence` to a copy of `cycles`.
    """
    cycles = cycles.copy()
    cycles["block"] = np.nan
    cycles["occurrence"] = np.nan
    seen: dict[tuple, int] = {}
    block = 0
    prev_key = None
    for idx, c in cycles.iterrows():
        if c.window_type == OTHER:
            prev_key = None
            continue
        key = (c.window_type, c.U_max, c.nu_nom_mVs)
        if key != prev_key:
            block += 1
            seen[key] = seen.get(key, 0) + 1
            prev_key = key
        cycles.at[idx, "block"] = block
        cycles.at[idx, "occurrence"] = seen[key]
    log.info("Detected %d block(s)", block)
    return cycles


def expected_sweep_duration(window_type: str, u_max: float, nu_nom_mVs: float) -> float:
    """Expected sweep duration [s]: 2*U_max/nu (symmetric) or U_max/nu (one-sided)."""
    nu = nu_nom_mVs / 1000.0
    return (2.0 if window_type == SYMMETRIC else 1.0) * u_max / nu


def is_complete(cycle: pd.Series, sweeps: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Complete cycle: both sweeps start and end at the vertices (-U/+U or 0/U)
    within vertex_tolerance(U_max), and each sweep lasts the expected
    duration within DURATION_REL_TOL. Returns (complete, reasons).
    """
    u = cycle.U_max
    lo = -u if cycle.window_type == SYMMETRIC else 0.0
    tol = vertex_tolerance(u)
    t_exp = expected_sweep_duration(cycle.window_type, u, cycle.nu_nom_mVs)
    reasons = []
    for name, s, v0, v1 in (("up", sweeps.loc[cycle.up], lo, u),
                            ("down", sweeps.loc[cycle.down], u, lo)):
        if abs(s.V_start - v0) > tol or abs(s.V_end - v1) > tol:
            reasons.append(f"{name} sweep {s.V_start:.3f}->{s.V_end:.3f} V misses a vertex")
        dev = (s.duration_s - t_exp) / t_exp
        if abs(dev) > cfg.DURATION_REL_TOL:
            reasons.append(f"{name} sweep lasts {s.duration_s:.1f} s, expected {t_exp:.1f} s "
                           f"({dev:+.1%})")
    return not reasons, reasons


def select_last_cycle(block_cycles: pd.DataFrame, sweeps: pd.DataFrame) -> dict:
    """
    The cycle to evaluate in one block: the last complete cycle (is_complete),
    plus the second-to-last complete cycle for the stability check.

    Returns dict(last, prev, n_cycles, n_complete, notes) with `last`/`prev`
    as rows of `block_cycles` (prev may be None). If no cycle is complete,
    the last cycle is used anyway and a note says so (never dropped silently).
    """
    complete = []
    notes = []
    for _, c in block_cycles.iterrows():
        ok, why = is_complete(c, sweeps)
        if ok:
            complete.append(c)
        else:
            log.info("Block %d: cycle at Excel rows %d-%d incomplete: %s",
                     int(c.block), c.row_start, c.row_end, "; ".join(why))
    if complete:
        last = complete[-1]
        prev = complete[-2] if len(complete) >= 2 else None
        if prev is None:
            notes.append("single cycle, may not be stabilised")
    else:
        last = block_cycles.iloc[-1]
        prev = None
        notes.append("no complete cycle - last cycle used")
        log.warning("Block %d: no complete cycle, evaluating the last one anyway", int(last.block))
    return dict(last=last, prev=prev, n_cycles=len(block_cycles),
                n_complete=len(complete), notes=notes)
