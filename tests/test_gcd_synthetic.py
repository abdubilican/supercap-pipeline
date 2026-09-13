"""
tests/test_gcd_synthetic.py
----------------------------
Correctness proof for gcd_analysis.py: an ideal capacitor + ESR model with
known C and known ESR, analytically closed-form, sampled at IRREGULAR time
points (matching the real tester's coarse, event-triggered sampling). Since
the model is perfectly linear in V(t) on each step, both the slope method
and the trapezoidal integral method must recover the exact ground-truth C
regardless of sampling irregularity (the trapezoidal rule is exact for a
linear integrand, independent of node spacing).

Model (current I constant-magnitude, in mA; R = ESR in Ohm; C in Farad):
  charge   (0 -> +I):  V(t) = I_A*R + (I_A/C)*t                  t in [0, t_ch]
  discharge, direct reversal (+I -> -I):
           V(t) = (V_max - 2*I_A*R) - (I_A/C)*t                  t in [0, t_dis]
  discharge, from rest (0 -> -I), rest held at V_cap_at_switch:
           V(t) = (V_rest_end - I_A*R) - (I_A/C)*t
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supercap_pipeline.gcd_analysis import (
    compute_capacitance_integral,
    compute_capacitance_slope,
    compute_efficiencies,
    compute_energy_power,
    compute_gcd_cycle_table,
    compute_ir_drop,
    compute_relaxation,
    segment_steps,
)

_TOL = 1e-4  # relative tolerance; the model is analytically exact


def _irregular_times(t0: float, t1: float, n: int, seed: int) -> np.ndarray:
    """n monotonic points from t0 to t1 inclusive, irregular spacing."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 1.8, size=n - 1)
    raw = raw / raw.sum() * (t1 - t0)
    return t0 + np.concatenate([[0.0], np.cumsum(raw)])


def _step_df(cycle: int, step: int, type_: str, c_d: str, t_step: np.ndarray, v: np.ndarray, i_mA: float) -> pd.DataFrame:
    n = len(t_step)
    return pd.DataFrame({
        "t_step": t_step, "cycle": cycle, "step": step, "type": type_, "c/d": c_d,
        "V12": v, "V1R": np.nan, "V2R": np.nan, "I12": i_mA, "over": "--",
    })


def _assemble(steps: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate steps into one DataFrame with a monotonic global `t`."""
    out = []
    t_offset = 0.0
    for g in steps:
        g = g.copy()
        g["t"] = t_offset + g["t_step"]
        out.append(g)
        t_offset = g["t"].iloc[-1] + 1.0  # gap between steps, like real data
    return pd.concat(out, ignore_index=True)


_META = dict(
    cell_tag="SYN", exp_name="synthetic", basis="total_active_mass",
    mass_total_g=0.02, mass_per_electrode_g=0.01, tester_cycles=None,
)


# ─────────────────────────────────────────────────────────────────────────────
# Case A: direct current reversal (no Rest between charge and discharge)
# ─────────────────────────────────────────────────────────────────────────────

def _make_case_direct_reversal(seed: int = 0):
    I_mA, R_ohm, C_F = 1.0, 10.0, 0.01
    I_A = I_mA / 1000.0
    V_max = 1.00  # terminal voltage at end of charge

    V_cap_switch = V_max - I_A * R_ohm          # 0.99
    t_ch = C_F * V_cap_switch / I_A              # 9.9 s
    V0_extrap = V_max - 2 * I_A * R_ohm          # 0.98
    t_dis = C_F * V0_extrap / I_A                # 9.8 s

    t_charge = _irregular_times(0.0, t_ch, 25, seed)
    v_charge = I_A * R_ohm + (I_A / C_F) * t_charge

    t_dis_arr = _irregular_times(0.0, t_dis, 25, seed + 1)
    v_dis = V0_extrap - (I_A / C_F) * t_dis_arr

    t_rest = _irregular_times(0.0, 10.0, 8, seed + 2)
    v_rest = 0.01 + (0.05 - 0.01) * (t_rest / 10.0)  # arbitrary linear recovery

    t_dummy = np.array([0.0, 0.001])
    v_dummy = np.array([0.0, 0.0])

    df = _assemble([
        _step_df(1, 20, "CC", "c", t_charge, v_charge, I_mA),
        _step_df(1, 30, "CC", "d", t_dis_arr, v_dis, -I_mA),
        _step_df(1, 40, "Rest", "d", t_rest, v_rest, 0.0),
        _step_df(2, 20, "CC", "c", t_dummy, v_dummy, I_mA),   # sacrificial trailing step
    ])

    truth = dict(
        C_F=C_F, R_ohm=R_ohm, I_mA=I_mA, V_max=V_max,
        V0_extrap=V0_extrap, ir_drop_V=V_max - V0_extrap, t_ch=t_ch, t_dis=t_dis,
    )
    return df, truth


def test_ir_drop_direct_reversal():
    df, truth = _make_case_direct_reversal()
    df2, steps_df = segment_steps(df)
    dis_id = int(steps_df[(steps_df["cycle"] == 1) & (steps_df["kind"] == "discharge")].iloc[0]["step_id"])

    ir = compute_ir_drop(df2, steps_df, dis_id, n_points=4)
    assert ir["esr_variant"] == "direct_reversal"
    assert abs(ir["ir_drop_V"] - truth["ir_drop_V"]) / truth["ir_drop_V"] < _TOL
    assert abs(ir["esr_Ohm"] - truth["R_ohm"]) / truth["R_ohm"] < _TOL
    assert ir["ir_fit_r2"] > 0.999  # perfectly linear model


def test_capacitance_slope_and_integral_direct_reversal():
    df, truth = _make_case_direct_reversal()
    df2, steps_df = segment_steps(df)
    dis_id = int(steps_df[(steps_df["cycle"] == 1) & (steps_df["kind"] == "discharge")].iloc[0]["step_id"])
    ir = compute_ir_drop(df2, steps_df, dis_id, n_points=4)
    u_max = ir["v_prev_end_V"]

    slope = compute_capacitance_slope(df2, dis_id, u_max)
    assert abs(slope["c_slope_F"] - truth["C_F"]) / truth["C_F"] < _TOL
    assert slope["slope_r2"] > 0.999

    integ = compute_capacitance_integral(df2, dis_id, u_max, ir["ir_drop_V"])
    assert abs(integ["c_integral_F"] - truth["C_F"]) / truth["C_F"] < _TOL


def test_energy_power_direct_reversal():
    df, truth = _make_case_direct_reversal()
    df2, steps_df = segment_steps(df)
    dis_id = int(steps_df[(steps_df["cycle"] == 1) & (steps_df["kind"] == "discharge")].iloc[0]["step_id"])
    ir = compute_ir_drop(df2, steps_df, dis_id, n_points=4)
    integ = compute_capacitance_integral(df2, dis_id, ir["v_prev_end_V"], ir["ir_drop_V"])

    energy = compute_energy_power(df2, dis_id, ir["esr_Ohm"], integ["u_eff_V"])
    # E = I * integral(V dt); integral of a triangle 0..U_eff over t_dis = 0.5*U_eff*t_dis
    expected_integral_Vs = 0.5 * integ["u_eff_V"] * truth["t_dis"]
    expected_e_mWh = truth["I_mA"] * expected_integral_Vs / 3600.0
    assert abs(energy["e_mWh"] - expected_e_mWh) / expected_e_mWh < _TOL

    expected_p_max_mW = (integ["u_eff_V"] ** 2) / (4 * truth["R_ohm"]) * 1000.0
    assert abs(energy["p_max_mW"] - expected_p_max_mW) / expected_p_max_mW < _TOL


def test_efficiencies_and_relaxation_direct_reversal():
    df, truth = _make_case_direct_reversal()
    df2, steps_df = segment_steps(df)
    cyc1 = steps_df[steps_df["cycle"] == 1]
    charge_id = int(cyc1[cyc1["kind"] == "charge"].iloc[0]["step_id"])
    dis_id    = int(cyc1[cyc1["kind"] == "discharge"].iloc[0]["step_id"])

    eff = compute_efficiencies(df2, charge_id, dis_id)
    expected_coulomb = truth["t_dis"] / truth["t_ch"]
    assert abs(eff["coulomb_eff_time"] - expected_coulomb) / expected_coulomb < _TOL
    assert abs(eff["coulomb_eff_charge"] - expected_coulomb) / expected_coulomb < _TOL
    assert 0.0 < eff["energy_eff"] < 1.0  # ESR losses must reduce energy efficiency below 1

    relax = compute_relaxation(df2, steps_df, dis_id)
    assert relax is not None
    assert abs(relax["relax_duration_s"] - 10.0) < 1e-6
    assert abs(relax["relax_v_end_V"] - 0.05) < 1e-6
    assert abs(relax["relax_dV_V"] - (0.05 - 0.01)) < 1e-6


def test_full_cycle_table_direct_reversal():
    """End-to-end: compute_gcd_cycle_table() recovers the same ground truth."""
    df, truth = _make_case_direct_reversal()
    cycle_df, steps_df = compute_gcd_cycle_table(df, _META)

    assert len(cycle_df) == 1  # cycle 2's dummy charge has no discharge pair -> skipped
    row = cycle_df.iloc[0]
    assert row["esr_variant"] == "direct_reversal"
    assert abs(row["esr_Ohm"] - truth["R_ohm"]) / truth["R_ohm"] < _TOL
    assert abs(row["c_cell_slope_F"] - truth["C_F"]) / truth["C_F"] < _TOL
    assert abs(row["c_cell_integral_F"] - truth["C_F"]) / truth["C_F"] < _TOL
    # total_active_mass basis: C_spec = 4*C_cell/m_total
    expected_spec = 4.0 * truth["C_F"] / _META["mass_total_g"]
    assert abs(row["C_spec_slope"] - expected_spec) / expected_spec < _TOL
    assert row["capacitance_linearity_rel_dev"] < 0.01
    assert not row["ocv_start"]  # charge starts at V=I*R, well within default 0.05V tolerance...

    # Dropped trailing step must be logged as incomplete.
    dropped = steps_df[~steps_df["complete"]]
    assert len(dropped) == 1
    assert dropped.iloc[0]["cycle"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Case B: Rest between charge and discharge (single current reversal)
# ─────────────────────────────────────────────────────────────────────────────

def _make_case_from_rest(seed: int = 10):
    I_mA, R_ohm, C_F = 1.0, 5.0, 0.02
    I_A = I_mA / 1000.0
    V_max = 1.00

    V_cap_switch = V_max - I_A * R_ohm          # 0.995
    t_ch = C_F * V_cap_switch / I_A

    V_rest_end = V_cap_switch                    # ideal capacitor: no self-discharge
    V0_extrap = V_rest_end - I_A * R_ohm          # single IR step (0 -> -I)
    t_dis = C_F * V0_extrap / I_A

    t_charge = _irregular_times(0.0, t_ch, 22, seed)
    v_charge = I_A * R_ohm + (I_A / C_F) * t_charge

    t_rest = _irregular_times(0.0, 5.0, 6, seed + 1)
    v_rest = np.full_like(t_rest, V_rest_end)

    t_dis_arr = _irregular_times(0.0, t_dis, 26, seed + 2)
    v_dis = V0_extrap - (I_A / C_F) * t_dis_arr

    t_dummy = np.array([0.0, 0.001])
    v_dummy = np.array([0.0, 0.0])

    df = _assemble([
        _step_df(1, 20, "CC", "c", t_charge, v_charge, I_mA),
        _step_df(1, 25, "Rest", "-", t_rest, v_rest, 0.0),
        _step_df(1, 30, "CC", "d", t_dis_arr, v_dis, -I_mA),
        _step_df(2, 20, "CC", "c", t_dummy, v_dummy, I_mA),
    ])

    truth = dict(C_F=C_F, R_ohm=R_ohm, I_mA=I_mA, ir_drop_V=V_rest_end - V0_extrap)
    return df, truth


def test_ir_drop_from_rest():
    df, truth = _make_case_from_rest()
    df2, steps_df = segment_steps(df)
    dis_id = int(steps_df[(steps_df["cycle"] == 1) & (steps_df["kind"] == "discharge")].iloc[0]["step_id"])

    ir = compute_ir_drop(df2, steps_df, dis_id, n_points=4)
    assert ir["esr_variant"] == "from_rest"
    assert abs(ir["ir_drop_V"] - truth["ir_drop_V"]) / truth["ir_drop_V"] < _TOL
    assert abs(ir["esr_Ohm"] - truth["R_ohm"]) / truth["R_ohm"] < _TOL


def test_capacitance_from_rest():
    df, truth = _make_case_from_rest()
    df2, steps_df = segment_steps(df)
    dis_id = int(steps_df[(steps_df["cycle"] == 1) & (steps_df["kind"] == "discharge")].iloc[0]["step_id"])
    ir = compute_ir_drop(df2, steps_df, dis_id, n_points=4)

    slope = compute_capacitance_slope(df2, dis_id, ir["v_prev_end_V"])
    assert abs(slope["c_slope_F"] - truth["C_F"]) / truth["C_F"] < _TOL

    integ = compute_capacitance_integral(df2, dis_id, ir["v_prev_end_V"], ir["ir_drop_V"])
    assert abs(integ["c_integral_F"] - truth["C_F"]) / truth["C_F"] < _TOL


# ─────────────────────────────────────────────────────────────────────────────
# Three-electrode consistency (V12 == V1R - V2R)
# ─────────────────────────────────────────────────────────────────────────────

def test_three_electrode_consistency_and_capacitance():
    from supercap_pipeline.gcd_analysis import compute_three_electrode

    I_mA, R_ohm, C_F = 1.0, 10.0, 0.01
    I_A = I_mA / 1000.0
    V_max = 1.00
    V_cap_switch = V_max - I_A * R_ohm
    V0_extrap = V_max - 2 * I_A * R_ohm
    t_dis = C_F * V0_extrap / I_A

    t_dis_arr = _irregular_times(0.0, t_dis, 25, 42)
    v12 = V0_extrap - (I_A / C_F) * t_dis_arr
    # split cell voltage arbitrarily between working/counter electrode vs reference
    v1r = 0.6 * v12 + 0.1
    v2r = v1r - v12

    g = pd.DataFrame({"t_step": t_dis_arr, "V12": v12, "V1R": v1r, "V2R": v2r, "I12": -I_mA})
    g["step_id"] = 0

    three = compute_three_electrode(g, 0, V_cap_switch)
    assert three is not None
    assert three["v12_consistency_max_abs_err_V"] < 1e-9
    # C1 is computed on V1R = 0.6*V12 + const -> dV1R/dt = 0.6 * dV12/dt -> C1 = C_cell/0.6
    assert abs(three["c1_F"] - C_F / 0.6) / (C_F / 0.6) < _TOL
