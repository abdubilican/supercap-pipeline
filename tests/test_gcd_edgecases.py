"""
tests/test_gcd_edgecases.py
-----------------------------
Edge cases called out explicitly in the task: a truncated last step (of
any kind, not just a charge step), fully-NaN auxiliary columns (no
reference electrode / no environmental sensors), and a Rest step sitting
between the charge and discharge of one cycle (handled through the full
compute_gcd_cycle_table() orchestrator, not just the low-level function).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supercap_pipeline.gcd_analysis import compute_gcd_cycle_table, segment_steps
from test_gcd_synthetic import _META, _assemble, _irregular_times, _make_case_from_rest, _step_df


def test_truncated_last_step_is_a_discharge_step():
    """
    If the file ends mid-DISCHARGE (not just mid-charge as in the example
    fixture), that discharge must also be dropped -- leaving zero complete
    cycles, not a crash and not a half-computed row.
    """
    t_charge = np.linspace(0.0, 10.0, 20)
    v_charge = 0.01 + 0.1 * t_charge
    t_dis = np.linspace(0.0, 3.0, 5)  # truncated: real discharge would run ~10s
    v_dis = 1.0 - 0.05 * t_dis

    df = _assemble([
        _step_df(1, 20, "CC", "c", t_charge, v_charge, 1.0),
        _step_df(1, 30, "CC", "d", t_dis, v_dis, -1.0),
    ])

    cycle_df, steps_df = compute_gcd_cycle_table(df, _META)

    dropped = steps_df[~steps_df["complete"]]
    assert len(dropped) == 1
    assert dropped.iloc[0]["kind"] == "discharge"
    assert len(cycle_df) == 0  # no complete charge+discharge pair remains


def test_all_nan_auxiliary_columns_do_not_crash():
    """
    T_cell/Pressure/Dilation/Force can be entirely NaN (sensors not
    installed) and V1R/V2R can be entirely NaN (no reference electrode,
    two-electrode-only cell) -- both are normal, not errors.
    """
    I_mA, R_ohm, C_F = 1.0, 8.0, 0.015
    I_A = I_mA / 1000.0
    V_max = 1.0
    V_cap_switch = V_max - I_A * R_ohm
    t_ch = C_F * V_cap_switch / I_A
    V0 = V_max - 2 * I_A * R_ohm
    t_dis = C_F * V0 / I_A

    t_charge = _irregular_times(0.0, t_ch, 20, 100)
    v_charge = I_A * R_ohm + (I_A / C_F) * t_charge
    t_dis_arr = _irregular_times(0.0, t_dis, 20, 101)
    v_dis = V0 - (I_A / C_F) * t_dis_arr
    t_dummy = np.array([0.0, 0.001])

    df = _assemble([
        _step_df(1, 20, "CC", "c", t_charge, v_charge, I_mA),
        _step_df(1, 30, "CC", "d", t_dis_arr, v_dis, -I_mA),
        _step_df(2, 20, "CC", "c", t_dummy, np.array([0.0, 0.0]), I_mA),
    ])
    # V1R/V2R already NaN (see _step_df); add the fully-NaN sensor columns too.
    for col in ["T_cell", "Pressure", "Dilation", "Force"]:
        df[col] = np.nan

    cycle_df, steps_df = compute_gcd_cycle_table(df, _META)

    assert len(cycle_df) == 1
    row = cycle_df.iloc[0]
    assert "C1_spec" not in cycle_df.columns  # no three-electrode data -> columns absent
    assert np.isfinite(row["c_cell_slope_F"])
    assert np.isfinite(row["esr_Ohm"])


def test_rest_between_charge_and_discharge_via_full_orchestrator():
    """Case B (from_rest ESR variant) through compute_gcd_cycle_table()."""
    df, truth = _make_case_from_rest()
    cycle_df, steps_df = compute_gcd_cycle_table(df, _META)

    assert len(cycle_df) == 1
    row = cycle_df.iloc[0]
    assert row["esr_variant"] == "from_rest"
    assert abs(row["esr_Ohm"] - truth["R_ohm"]) / truth["R_ohm"] < 1e-3
    assert abs(row["c_cell_slope_F"] - truth["C_F"]) / truth["C_F"] < 1e-3
    assert abs(row["c_cell_integral_F"] - truth["C_F"]) / truth["C_F"] < 1e-3
