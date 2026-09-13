"""
gcd_analysis.py
---------------
Compute galvanostatic charge/discharge (GCD) cycle metrics from the raw
Output.txt DataFrame produced by gcd_loader.load_gcd_experiment().

Pipeline of this module
------------------------
1. segment_steps()       -- group rows into (cycle, step) steps, classify
                             each as charge/discharge/rest/other, drop the
                             trailing incomplete step at file end.
2. flag_ocv_start()       -- flag the very first charge step if it starts
                             far from 0V (OCV rest beforehand, not a fully
                             discharged cell) -> excluded from aggregates.
3. assign_rate_stages()   -- cluster cycles by |I12| into rate stages
                             (~2% relative tolerance).
4. per discharge step:
     compute_ir_drop()            -- ESR / IR-drop via linear extrapolation
     compute_capacitance_slope()  -- IEC 62391 slope method
     compute_capacitance_integral() -- C = 2*I*INT(U dt)/U_eff^2
     compute_energy_power()       -- E, P_mean, P_max
     compute_efficiencies()       -- vs. the preceding charge step
     compute_three_electrode()    -- if V1R/V2R present
     compute_relaxation()         -- voltage recovery during a following Rest
5. compute_gcd_cycle_table()  -- orchestrates the above into one tidy
                                  DataFrame, one row per complete cycle.
6. compute_gcd_rate_aggregate() -- mean +/- std per rate stage.
7. build_qc_report()          -- text summary of everything worth a human's
                                  attention (dropped steps, over-flags, low
                                  fit quality, implausible efficiencies, ...).

Units & normalization
----------------------
Internally everything stays in the tester's native units (mA, V, s, mWh,
mW) -- only compute_gcd_cycle_table() converts to the reported specific
units at the very end, using meta["basis"] (REQUIRED, no default):

    basis == "single_electrode" : C_spec = 2*C_cell / m_one_electrode
    basis == "total_active_mass": C_spec = 4*C_cell / m_total
    basis == "cell"             : C_spec =   C_cell / m_total

Specific energy/power/current density use the same normalizing mass as
capacitance (m_one_electrode for "single_electrode", m_total otherwise),
without the 2x/4x capacitance doubling factor (that factor only converts
cell capacitance to electrode capacitance; current/energy are not doubled).
Because 1 mWh/g == 1 Wh/kg and 1 mW/g == 1 W/kg numerically, dividing the
tester's native mWh/mW values by grams directly yields Wh/kg and W/kg.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Configurable defaults (all overridable via function kwargs) ─────────────
DEFAULT_IR_N_POINTS           = 4        # IEC/lab practice: 3-5 points
DEFAULT_SLOPE_WINDOW          = (0.40, 0.80)  # fraction of U_max, IEC 62391
DEFAULT_MIN_SLOPE_POINTS      = 5
DEFAULT_OCV_TOL_V             = 0.05
DEFAULT_RATE_CLUSTER_TOL      = 0.02     # 2% relative
DEFAULT_V_CONSISTENCY_TOL_V   = 0.01
DEFAULT_R2_WARN_THRESHOLD     = 0.98
DEFAULT_LINEARITY_WARN_THRESHOLD = 0.10  # 10% relative |slope-integral| dev
DEFAULT_COULOMB_EFF_PLAUSIBLE = (0.80, 1.05)


# ─────────────────────────────────────────────────────────────────────────────
# Step segmentation & classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_step(type_: str, c_d: str) -> str:
    if type_ == "Rest":
        return "rest"
    if type_ == "CC" and c_d == "c":
        return "charge"
    if type_ == "CC" and c_d == "d":
        return "discharge"
    return "other"


def segment_steps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group the raw GCD DataFrame into contiguous (cycle, step) steps.

    Returns
    -------
    df       : copy of `df`, sorted by t, with an added integer `step_id`
               column (0..N-1, chronological order of first appearance)
    steps_df : one row per step_id: step_id, cycle, step, type, c_d, kind,
               n_points, t_start, t_end, duration_s, i_mean_mA,
               i_abs_mean_mA, complete (bool).

    The step containing the LAST row of the file is marked complete=False
    (the data stream can end mid-step) and is excluded from every
    downstream per-cycle metric. This is logged as a warning.
    """
    df = df.sort_values("t").reset_index(drop=True).copy()

    new_step = (df["cycle"] != df["cycle"].shift()) | (df["step"] != df["step"].shift())
    df["step_id"] = new_step.cumsum() - 1

    rows = []
    for step_id, g in df.groupby("step_id", sort=True):
        rows.append(dict(
            step_id       = int(step_id),
            cycle         = int(g["cycle"].iloc[0]),
            step          = int(g["step"].iloc[0]),
            type          = g["type"].iloc[0],
            c_d           = g["c/d"].iloc[0],
            kind          = classify_step(g["type"].iloc[0], g["c/d"].iloc[0]),
            n_points      = len(g),
            t_start       = float(g["t"].iloc[0]),
            t_end         = float(g["t"].iloc[-1]),
            duration_s    = float(g["t"].iloc[-1] - g["t"].iloc[0]),
            i_mean_mA     = float(g["I12"].mean()),
            i_abs_mean_mA = float(g["I12"].abs().mean()),
        ))
    steps_df = pd.DataFrame(rows)
    steps_df["complete"] = True

    last_step_id = int(df["step_id"].iloc[-1])
    steps_df.loc[steps_df["step_id"] == last_step_id, "complete"] = False

    dropped = steps_df[~steps_df["complete"]]
    for _, r in dropped.iterrows():
        log.warning(
            "Dropping incomplete trailing step cycle=%d step=%d (%s, c/d=%s): "
            "file ends mid-step (%d points, %.3fs).",
            r["cycle"], r["step"], r["type"], r["c_d"], r["n_points"], r["duration_s"],
        )
    return df, steps_df


def flag_ocv_start(
    df: pd.DataFrame,
    steps_df: pd.DataFrame,
    tol_v: float = DEFAULT_OCV_TOL_V,
) -> pd.DataFrame:
    """
    Flag the very first charge step if V12 at its start is not within
    `tol_v` of 0V -- it began from an open-circuit voltage, not a fully
    discharged cell, so its charge/discharge pair must be excluded from
    aggregate statistics (kept, flagged, in the per-cycle table).
    """
    steps_df = steps_df.copy()
    steps_df["ocv_start"] = False

    charge_steps = steps_df[steps_df["kind"] == "charge"].sort_values("step_id")
    if charge_steps.empty:
        return steps_df

    first = charge_steps.iloc[0]
    v0 = float(df.loc[df["step_id"] == first["step_id"], "V12"].iloc[0])
    if abs(v0) > tol_v:
        steps_df.loc[steps_df["step_id"] == first["step_id"], "ocv_start"] = True
        log.info(
            "First charge step (cycle=%d) starts at V12=%.3fV (tol=%.3fV): "
            "flagged ocv_start, excluded from aggregate statistics.",
            first["cycle"], v0, tol_v,
        )
    return steps_df


def assign_rate_stages(
    steps_df: pd.DataFrame,
    tol: float = DEFAULT_RATE_CLUSTER_TOL,
) -> pd.DataFrame:
    """
    Cluster cycles by their (complete) charge step's mean |I12| into rate
    stages, using a greedy 1D clustering on sorted currents with `tol`
    relative tolerance between consecutive cluster members.

    Adds `rate_group` (int, ascending by current) and `i_nominal_mA`
    (cluster median) to every step belonging to that cycle.
    """
    steps_df = steps_df.copy()
    charge = steps_df[(steps_df["kind"] == "charge") & steps_df["complete"]]
    charge = charge.sort_values("i_abs_mean_mA")

    clusters: list[list[int]] = []
    cluster_currents: list[list[float]] = []
    for _, row in charge.iterrows():
        i = row["i_abs_mean_mA"]
        if clusters and abs(i - cluster_currents[-1][-1]) / cluster_currents[-1][-1] <= tol:
            clusters[-1].append(int(row["cycle"]))
            cluster_currents[-1].append(i)
        else:
            clusters.append([int(row["cycle"])])
            cluster_currents.append([i])

    cycle_to_group: dict[int, int] = {}
    cycle_to_inom: dict[int, float] = {}
    for gi, (cyc_list, i_list) in enumerate(zip(clusters, cluster_currents)):
        i_nom = float(np.median(i_list))
        for c in cyc_list:
            cycle_to_group[c] = gi
            cycle_to_inom[c] = i_nom

    steps_df["rate_group"]   = steps_df["cycle"].map(cycle_to_group)
    steps_df["i_nominal_mA"] = steps_df["cycle"].map(cycle_to_inom)
    return steps_df


# ─────────────────────────────────────────────────────────────────────────────
# Low-level numerics
# ─────────────────────────────────────────────────────────────────────────────

def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS line fit y = a + b*x. Returns (intercept a, slope b, R^2)."""
    b, a = np.polyfit(x, y, 1)
    y_pred = a + b * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), r2


def step_frame(df: pd.DataFrame, step_id: int) -> pd.DataFrame:
    return df.loc[df["step_id"] == step_id].sort_values("t_step")


def _step_integrals(g: pd.DataFrame) -> dict:
    """Shared trapezoidal integrals for one step's (t_step, V12, I12)."""
    t = g["t_step"].to_numpy(dtype=float)
    v = g["V12"].to_numpy(dtype=float)
    i_abs = g["I12"].abs().to_numpy(dtype=float)

    duration_s      = float(t[-1] - t[0])
    i_mean_mA       = float(i_abs.mean())
    integral_v_dt_Vs = float(np.trapezoid(v, t))
    q_mAh           = float(np.trapezoid(i_abs, t) / 3600.0)
    e_mWh           = float(i_mean_mA * integral_v_dt_Vs / 3600.0)

    return dict(
        duration_s=duration_s, i_mean_mA=i_mean_mA,
        integral_v_dt_Vs=integral_v_dt_Vs, q_mAh=q_mAh, e_mWh=e_mWh,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IR-drop / ESR
# ─────────────────────────────────────────────────────────────────────────────

def compute_ir_drop(
    df: pd.DataFrame,
    steps_df: pd.DataFrame,
    step_id: int,
    n_points: int = DEFAULT_IR_N_POINTS,
) -> dict:
    """
    IR-drop at the start of `step_id` (normally a discharge step): fit a
    line through the first `n_points` (V12 vs t_step) and extrapolate to
    t_step=0 -- this is the cell voltage right at current reversal, before
    the first (unreliable, coarse-sampled) measured point.

    ΔU_IR = V(end of the immediately preceding step) - V0_extrapolated.
    ESR variant is derived from the preceding step's kind: a direct charge
    -> discharge reversal changes the current by 2*|I| (ESR = ΔU/(2|I|));
    a Rest step in between changes it by only |I| (ESR = ΔU/|I|).
    """
    g = step_frame(df, step_id)
    n = min(n_points, len(g))
    if n < 2:
        raise ValueError(f"step_id={step_id}: not enough points for IR-drop fit (n={n}).")

    t = g["t_step"].to_numpy(dtype=float)[:n]
    v = g["V12"].to_numpy(dtype=float)[:n]
    v0_extrap, _slope, r2 = _linfit(t, v)

    prev_rows = steps_df[steps_df["step_id"] == step_id - 1]
    if prev_rows.empty:
        raise ValueError(f"step_id={step_id}: no preceding step (file starts here).")
    prev = prev_rows.iloc[0]
    prev_g = step_frame(df, int(prev["step_id"]))
    v_prev_end = float(prev_g["V12"].iloc[-1])

    i_dis_mA = float(g["I12"].mean())
    ir_drop_V = v_prev_end - v0_extrap

    if prev["kind"] == "rest":
        esr_variant = "from_rest"
        denom_mA = abs(i_dis_mA)
    else:
        esr_variant = "direct_reversal"
        denom_mA = 2.0 * abs(i_dis_mA)
    esr_Ohm = abs(ir_drop_V) / (denom_mA / 1000.0) if denom_mA > 0 else np.nan

    return dict(
        step_id=step_id, v0_extrap_V=v0_extrap, ir_fit_r2=r2, ir_n_points=n,
        v_prev_end_V=v_prev_end, prev_step_id=int(prev["step_id"]), prev_kind=prev["kind"],
        ir_drop_V=ir_drop_V, esr_variant=esr_variant, esr_Ohm=esr_Ohm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Capacitance
# ─────────────────────────────────────────────────────────────────────────────

def compute_capacitance_slope(
    df: pd.DataFrame,
    step_id: int,
    u_max_V: float,
    window: tuple[float, float] = DEFAULT_SLOPE_WINDOW,
    min_points: int = DEFAULT_MIN_SLOPE_POINTS,
) -> dict:
    """
    IEC 62391 slope method: C = |I| / |dV/dt|, linear regression of V12 vs
    t_step over the window where V falls within
    [window[0]*u_max_V, window[1]*u_max_V].

    `u_max_V` is the reference voltage (normally the charge step's terminal
    voltage, i.e. `compute_ir_drop(...)["v_prev_end_V"]`) -- shared with
    compute_capacitance_integral() so both methods use the same reference.
    """
    g = step_frame(df, step_id)
    lo, hi = window[0] * u_max_V, window[1] * u_max_V
    sub = g.loc[(g["V12"] >= lo) & (g["V12"] <= hi)]
    i_mA = float(g["I12"].abs().mean())

    if len(sub) < min_points:
        log.warning(
            "step_id=%d: slope-method window [%.0f%%..%.0f%% Umax] has only "
            "%d points (< %d) -- capacitance unreliable.",
            step_id, window[0] * 100, window[1] * 100, len(sub), min_points,
        )
        return dict(c_slope_F=np.nan, slope_r2=np.nan, n_points_used=len(sub),
                    window_lo_V=lo, window_hi_V=hi)

    t = sub["t_step"].to_numpy(dtype=float)
    v = sub["V12"].to_numpy(dtype=float)
    _a, b, r2 = _linfit(t, v)
    c_slope_F = (i_mA / 1000.0) / abs(b) if b != 0 else np.nan

    return dict(c_slope_F=c_slope_F, slope_r2=r2, n_points_used=len(sub),
                window_lo_V=lo, window_hi_V=hi)


def compute_capacitance_integral(
    df: pd.DataFrame,
    step_id: int,
    u_max_V: float,
    ir_drop_V: float,
) -> dict:
    """
    Integral method: C = 2*|I| * INT(U dt) / U_eff^2, U_eff = u_max_V - ir_drop_V.
    """
    g = step_frame(df, step_id)
    integ = _step_integrals(g)
    u_eff_V = u_max_V - ir_drop_V

    if u_eff_V <= 0:
        c_integral_F = np.nan
    else:
        c_integral_F = 2.0 * (integ["i_mean_mA"] / 1000.0) * integ["integral_v_dt_Vs"] / (u_eff_V ** 2)

    return dict(c_integral_F=c_integral_F, u_eff_V=u_eff_V, **integ)


# ─────────────────────────────────────────────────────────────────────────────
# Energy / power
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_power(
    df: pd.DataFrame,
    step_id: int,
    esr_Ohm: float,
    u_eff_V: float,
) -> dict:
    """
    E = |I| * INT(U dt) (already computed by _step_integrals as e_mWh),
    P_mean = E / t_dis, P_max = U_eff^2 / (4*ESR).
    """
    g = step_frame(df, step_id)
    integ = _step_integrals(g)

    p_mean_mW = (integ["e_mWh"] * 3600.0 / integ["duration_s"]) if integ["duration_s"] > 0 else np.nan
    if esr_Ohm and esr_Ohm > 0 and not np.isnan(esr_Ohm):
        p_max_mW = (u_eff_V ** 2) / (4.0 * esr_Ohm) * 1000.0
    else:
        p_max_mW = np.nan

    return dict(e_mWh=integ["e_mWh"], p_mean_mW=p_mean_mW, p_max_mW=p_max_mW,
                duration_s=integ["duration_s"])


# ─────────────────────────────────────────────────────────────────────────────
# Efficiencies (discharge vs. the immediately preceding charge)
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiencies(
    df: pd.DataFrame,
    charge_step_id: int,
    discharge_step_id: int,
) -> dict:
    ic   = _step_integrals(step_frame(df, charge_step_id))
    idis = _step_integrals(step_frame(df, discharge_step_id))

    coulomb_eff_time   = idis["duration_s"] / ic["duration_s"] if ic["duration_s"] > 0 else np.nan
    coulomb_eff_charge = idis["q_mAh"] / ic["q_mAh"] if ic["q_mAh"] > 0 else np.nan
    energy_eff         = idis["e_mWh"] / ic["e_mWh"] if ic["e_mWh"] > 0 else np.nan

    return dict(
        t_charge_s=ic["duration_s"], t_discharge_s=idis["duration_s"],
        coulomb_eff_time=coulomb_eff_time, coulomb_eff_charge=coulomb_eff_charge,
        q_charge_mAh=ic["q_mAh"], q_discharge_mAh=idis["q_mAh"],
        e_charge_mWh=ic["e_mWh"], e_discharge_mWh=idis["e_mWh"],
        energy_eff=energy_eff,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Three-electrode (working/counter vs. reference)
# ─────────────────────────────────────────────────────────────────────────────

def compute_three_electrode(
    df: pd.DataFrame,
    step_id: int,
    u_max_V: float,
    window: tuple[float, float] = DEFAULT_SLOPE_WINDOW,
    min_points: int = DEFAULT_MIN_SLOPE_POINTS,
    v_consistency_tol_V: float = DEFAULT_V_CONSISTENCY_TOL_V,
) -> dict | None:
    """
    Working/counter electrode capacitance (slope method on V1R/V2R, same
    window as the cell-level slope method) + potential windows. Returns
    None if V1R/V2R are entirely NaN for this step (no reference electrode).
    Warns (does not raise) if V12 != V1R - V2R beyond tolerance.
    """
    g = step_frame(df, step_id)
    if g["V1R"].isna().all() or g["V2R"].isna().all():
        return None

    i_mA = float(g["I12"].abs().mean())
    lo, hi = window[0] * u_max_V, window[1] * u_max_V
    mask = (g["V12"] >= lo) & (g["V12"] <= hi)
    sub = g.loc[mask]

    def _slope_c(col: str) -> tuple[float, float, int]:
        if len(sub) < min_points:
            return np.nan, np.nan, len(sub)
        t = sub["t_step"].to_numpy(dtype=float)
        v = sub[col].to_numpy(dtype=float)
        _a, b, r2 = _linfit(t, v)
        c = (i_mA / 1000.0) / abs(b) if b != 0 else np.nan
        return c, r2, len(sub)

    c1_F, r2_1, n1 = _slope_c("V1R")
    c2_F, r2_2, n2 = _slope_c("V2R")

    v12_residual = (g["V1R"] - g["V2R"]) - g["V12"]
    max_abs_err_V = float(v12_residual.abs().max())
    if max_abs_err_V > v_consistency_tol_V:
        log.warning(
            "step_id=%d: V12 != V1R-V2R (max abs err=%.4fV > tol=%.4fV).",
            step_id, max_abs_err_V, v_consistency_tol_V,
        )

    return dict(
        c1_F=c1_F, c1_r2=r2_1, c1_n_points=n1,
        c2_F=c2_F, c2_r2=r2_2, c2_n_points=n2,
        v1r_min_V=float(g["V1R"].min()), v1r_max_V=float(g["V1R"].max()),
        v2r_min_V=float(g["V2R"].min()), v2r_max_V=float(g["V2R"].max()),
        v12_consistency_max_abs_err_V=max_abs_err_V,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relaxation
# ─────────────────────────────────────────────────────────────────────────────

def compute_relaxation(
    df: pd.DataFrame,
    steps_df: pd.DataFrame,
    discharge_step_id: int,
) -> dict | None:
    """
    Voltage recovery during a Rest step immediately following
    `discharge_step_id`. Returns None if no such Rest step exists.
    """
    rest_rows = steps_df[steps_df["step_id"] == discharge_step_id + 1]
    if rest_rows.empty or rest_rows.iloc[0]["kind"] != "rest":
        return None

    rest_id = int(rest_rows.iloc[0]["step_id"])
    g = step_frame(df, rest_id)
    v_start = float(g["V12"].iloc[0])
    v_end   = float(g["V12"].iloc[-1])

    return dict(
        relax_step_id=rest_id,
        relax_duration_s=float(g["t_step"].iloc[-1] - g["t_step"].iloc[0]),
        relax_dV_V=v_end - v_start,
        relax_v_end_V=v_end,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalizing_mass_g(basis: str, mass_total_g: float, mass_electrode_g: float) -> float:
    """Mass used for current density / energy / power (no doubling factor)."""
    return mass_electrode_g if basis == "single_electrode" else mass_total_g


def _normalize_capacitance(c_cell_F: float, basis: str, mass_total_g: float, mass_electrode_g: float) -> float:
    if basis == "single_electrode":
        return 2.0 * c_cell_F / mass_electrode_g
    if basis == "total_active_mass":
        return 4.0 * c_cell_F / mass_total_g
    return c_cell_F / mass_total_g  # "cell"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: per-cycle table
# ─────────────────────────────────────────────────────────────────────────────

def compute_gcd_cycle_table(
    df: pd.DataFrame,
    meta: dict,
    ir_n_points: int = DEFAULT_IR_N_POINTS,
    slope_window: tuple[float, float] = DEFAULT_SLOPE_WINDOW,
    min_slope_points: int = DEFAULT_MIN_SLOPE_POINTS,
    ocv_tol_V: float = DEFAULT_OCV_TOL_V,
    rate_cluster_tol: float = DEFAULT_RATE_CLUSTER_TOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full per-cycle GCD analysis.

    Parameters
    ----------
    df   : raw DataFrame from gcd_loader.load_gcd_experiment()
    meta : metadata dict from the same (REQUIRES meta["basis"],
           meta["mass_total_g"], meta["mass_per_electrode_g"])

    Returns
    -------
    cycle_df : tidy DataFrame, one row per complete (charge+discharge)
               cycle > 0. See module docstring for the normalization rules.
    steps_df : the full step-segmentation table (for QC / plotting).
    """
    basis            = meta["basis"]
    mass_total_g     = meta["mass_total_g"]
    mass_electrode_g = meta["mass_per_electrode_g"]
    norm_mass_g      = _normalizing_mass_g(basis, mass_total_g, mass_electrode_g)

    df2, steps_df = segment_steps(df)
    steps_df = flag_ocv_start(df2, steps_df, tol_v=ocv_tol_V)
    steps_df = assign_rate_stages(steps_df, tol=rate_cluster_tol)

    tester_cycles = meta.get("tester_cycles")

    rows: list[dict] = []
    for cycle, cyc_steps in steps_df[steps_df["cycle"] > 0].groupby("cycle"):
        charge_rows = cyc_steps[(cyc_steps["kind"] == "charge") & cyc_steps["complete"]]
        dis_rows    = cyc_steps[(cyc_steps["kind"] == "discharge") & cyc_steps["complete"]]
        if charge_rows.empty or dis_rows.empty:
            log.warning("cycle=%d: no complete charge+discharge pair, skipping.", cycle)
            continue

        charge_id = int(charge_rows.iloc[0]["step_id"])
        dis_id    = int(dis_rows.iloc[0]["step_id"])

        ir    = compute_ir_drop(df2, steps_df, dis_id, n_points=ir_n_points)
        u_max = ir["v_prev_end_V"]

        slope = compute_capacitance_slope(df2, dis_id, u_max, window=slope_window, min_points=min_slope_points)
        integ = compute_capacitance_integral(df2, dis_id, u_max, ir["ir_drop_V"])
        energy = compute_energy_power(df2, dis_id, ir["esr_Ohm"], integ["u_eff_V"])
        eff   = compute_efficiencies(df2, charge_id, dis_id)
        three = compute_three_electrode(df2, dis_id, u_max, window=slope_window, min_points=min_slope_points)
        relax = compute_relaxation(df2, steps_df, dis_id)

        c_slope_F, c_integral_F = slope["c_slope_F"], integ["c_integral_F"]
        if np.isfinite(c_slope_F) and np.isfinite(c_integral_F) and (c_slope_F + c_integral_F) != 0:
            linearity_rel_dev = abs(c_slope_F - c_integral_F) / ((c_slope_F + c_integral_F) / 2.0)
        else:
            linearity_rel_dev = np.nan

        over_values = sorted(
            v for v in df2.loc[df2["step_id"].isin(cyc_steps["step_id"]), "over"].unique()
            if v not in ("--", "nan")
        )

        row = dict(
            cell_tag=meta.get("cell_tag"), exp_name=meta.get("exp_name"),
            cycle=int(cycle),
            rate_group=int(dis_rows.iloc[0]["rate_group"]),
            i_nominal_mA=float(dis_rows.iloc[0]["i_nominal_mA"]),
            i_charge_mA=float(charge_rows.iloc[0]["i_abs_mean_mA"]),
            i_discharge_mA=float(dis_rows.iloc[0]["i_abs_mean_mA"]),
            basis=basis,
            ocv_start=bool(charge_rows.iloc[0]["ocv_start"]),
            over_flags=",".join(over_values) if over_values else "",
            u_max_V=u_max,
            ir_drop_V=ir["ir_drop_V"], ir_fit_r2=ir["ir_fit_r2"], ir_n_points=ir["ir_n_points"],
            esr_Ohm=ir["esr_Ohm"], esr_variant=ir["esr_variant"],
            c_cell_slope_F=c_slope_F, slope_r2=slope["slope_r2"], slope_n_points=slope["n_points_used"],
            c_cell_integral_F=c_integral_F, u_eff_V=integ["u_eff_V"],
            capacitance_linearity_rel_dev=linearity_rel_dev,
            C_spec_slope=_normalize_capacitance(c_slope_F, basis, mass_total_g, mass_electrode_g),
            C_spec_integral=_normalize_capacitance(c_integral_F, basis, mass_total_g, mass_electrode_g),
            current_density_Ag=(dis_rows.iloc[0]["i_abs_mean_mA"] / 1000.0) / norm_mass_g,
            E_discharge_Whkg=energy["e_mWh"] / norm_mass_g,
            P_mean_Wkg=energy["p_mean_mW"] / norm_mass_g,
            P_max_Wkg=energy["p_max_mW"] / norm_mass_g,
            t_charge_s=eff["t_charge_s"], t_discharge_s=eff["t_discharge_s"],
            coulomb_eff_time=eff["coulomb_eff_time"], coulomb_eff_charge=eff["coulomb_eff_charge"],
            energy_eff=eff["energy_eff"],
            q_charge_mAh=eff["q_charge_mAh"], q_discharge_mAh=eff["q_discharge_mAh"],
        )

        if three is not None:
            row.update({
                "C1_spec": _normalize_capacitance(three["c1_F"], basis, mass_total_g, mass_electrode_g),
                "C2_spec": _normalize_capacitance(three["c2_F"], basis, mass_total_g, mass_electrode_g),
                "c1_r2": three["c1_r2"], "c2_r2": three["c2_r2"],
                "v1r_min_V": three["v1r_min_V"], "v1r_max_V": three["v1r_max_V"],
                "v2r_min_V": three["v2r_min_V"], "v2r_max_V": three["v2r_max_V"],
                "v12_consistency_max_abs_err_V": three["v12_consistency_max_abs_err_V"],
            })

        if relax is not None:
            row.update({
                "relax_duration_s": relax["relax_duration_s"],
                "relax_dV_V": relax["relax_dV_V"],
                "relax_v_end_V": relax["relax_v_end_V"],
            })

        if tester_cycles is not None:
            match = tester_cycles[tester_cycles["cycle"] == cycle]
            if not match.empty:
                for col in match.columns:
                    if col != "cycle":
                        row[col] = match.iloc[0][col]

        rows.append(row)

    cycle_df = pd.DataFrame(rows)
    if not cycle_df.empty:
        cycle_df = cycle_df.sort_values("cycle").reset_index(drop=True)
    return cycle_df, steps_df


def compute_gcd_rate_aggregate(
    cycle_df: pd.DataFrame,
    exclude_first_cycle: bool = True,
) -> pd.DataFrame:
    """
    Mean +/- std per rate stage, over complete, non-OCV-flagged cycles.

    Parameters
    ----------
    cycle_df            : output of compute_gcd_cycle_table()
    exclude_first_cycle : if True, additionally skip the first cycle of
                          each rate stage (formation cycle)
    """
    if cycle_df.empty:
        return cycle_df.copy()

    df = cycle_df[~cycle_df["ocv_start"]].copy()

    if exclude_first_cycle:
        first_cycle = df.groupby("rate_group")["cycle"].transform("min")
        df = df[df["cycle"] > first_cycle]

    numeric_cols = [
        "i_nominal_mA", "current_density_Ag",
        "C_spec_slope", "C_spec_integral", "capacitance_linearity_rel_dev",
        "esr_Ohm", "ir_drop_V",
        "E_discharge_Whkg", "P_mean_Wkg", "P_max_Wkg",
        "coulomb_eff_time", "coulomb_eff_charge", "energy_eff",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    agg = df.groupby("rate_group")[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_cycles"] = df.groupby("rate_group").size()
    agg = agg.reset_index().sort_values("rate_group").reset_index(drop=True)

    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)

    return agg


def compute_capacity_retention(rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Capacity retention (%) per rate stage, relative to the LOWEST-current
    rate stage (not necessarily rate_group==0, though assign_rate_stages()
    numbers groups ascending by current so it normally is):

        retention_pct(stage) = 100 * C_spec_mean(stage) / C_spec_mean(lowest stage)

    computed separately for the slope and integral methods.

    Parameters
    ----------
    rate_df : output of compute_gcd_rate_aggregate()

    Returns
    -------
    DataFrame: rate_group, current_density_Ag, i_nominal_mA,
               retention_slope_pct, retention_integral_pct
    """
    if rate_df.empty:
        return rate_df.copy()

    ref_idx = rate_df["current_density_Ag_mean"].idxmin()
    ref_slope = rate_df.loc[ref_idx, "C_spec_slope_mean"]
    ref_integral = rate_df.loc[ref_idx, "C_spec_integral_mean"]

    out = rate_df[[
        "rate_group", "current_density_Ag_mean", "i_nominal_mA_mean",
        "C_spec_slope_mean", "C_spec_integral_mean",
    ]].rename(columns={
        "current_density_Ag_mean": "current_density_Ag",
        "i_nominal_mA_mean": "i_nominal_mA",
    }).copy()

    out["retention_slope_pct"] = 100.0 * out["C_spec_slope_mean"] / ref_slope
    out["retention_integral_pct"] = 100.0 * out["C_spec_integral_mean"] / ref_integral

    return out.sort_values("rate_group").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# QC report
# ─────────────────────────────────────────────────────────────────────────────

def build_qc_report(
    cycle_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    r2_threshold: float = DEFAULT_R2_WARN_THRESHOLD,
    linearity_threshold: float = DEFAULT_LINEARITY_WARN_THRESHOLD,
    coulomb_eff_range: tuple[float, float] = DEFAULT_COULOMB_EFF_PLAUSIBLE,
) -> str:
    """Plain-text QC summary: dropped steps, over-flags, fit quality, ..."""
    lines: list[str] = ["GCD QC report", "=" * 60]

    dropped = steps_df[~steps_df["complete"]]
    lines.append(f"\nDropped incomplete trailing steps: {len(dropped)}")
    for _, r in dropped.iterrows():
        lines.append(f"  cycle={r['cycle']} step={r['step']} ({r['type']}, c/d={r['c_d']}) "
                      f"-- {r['n_points']} points, {r['duration_s']:.3f}s")

    ocv = cycle_df[cycle_df["ocv_start"]]
    lines.append(f"\nCycles flagged ocv_start (excluded from aggregates): {len(ocv)}")
    for _, r in ocv.iterrows():
        lines.append(f"  cycle={r['cycle']}")

    flagged = cycle_df[cycle_df["over_flags"] != ""]
    lines.append(f"\nCycles with 'over' flags: {len(flagged)}")
    for _, r in flagged.iterrows():
        lines.append(f"  cycle={r['cycle']}: {r['over_flags']}")

    low_r2 = cycle_df[cycle_df["slope_r2"] < r2_threshold]
    lines.append(f"\nCycles with slope-fit R^2 < {r2_threshold}: {len(low_r2)}")
    for _, r in low_r2.iterrows():
        lines.append(f"  cycle={r['cycle']}: R^2={r['slope_r2']:.4f} (n={r['slope_n_points']})")

    low_points = cycle_df[cycle_df["slope_n_points"] < DEFAULT_MIN_SLOPE_POINTS]
    lines.append(f"\nCycles with < {DEFAULT_MIN_SLOPE_POINTS} points in the slope-fit window: {len(low_points)}")
    for _, r in low_points.iterrows():
        lines.append(f"  cycle={r['cycle']}: {r['slope_n_points']} points")

    bad_linearity = cycle_df[cycle_df["capacitance_linearity_rel_dev"] > linearity_threshold]
    lines.append(f"\nCycles with slope-vs-integral deviation > {linearity_threshold*100:.0f}%: {len(bad_linearity)}")
    for _, r in bad_linearity.iterrows():
        lines.append(f"  cycle={r['cycle']}: {r['capacitance_linearity_rel_dev']*100:.1f}%")

    lo, hi = coulomb_eff_range
    non_ocv = cycle_df[~cycle_df["ocv_start"]]
    implausible = non_ocv[
        (non_ocv["coulomb_eff_charge"] < lo) | (non_ocv["coulomb_eff_charge"] > hi)
    ]
    lines.append(f"\nCycles with Coulomb efficiency outside [{lo}, {hi}] (excl. ocv_start): {len(implausible)}")
    for _, r in implausible.iterrows():
        lines.append(f"  cycle={r['cycle']}: {r['coulomb_eff_charge']:.4f}")

    return "\n".join(lines) + "\n"
