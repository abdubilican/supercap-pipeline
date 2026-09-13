"""
tests/test_gcd_parser.py
-------------------------
Parser-level tests against the real (truncated) tester export
tests/fixtures/Output_Claude.txt, and the full loader against
tests/fixtures/gcd_experiment/ (same data, named Output.txt + info.json
as a real measurement folder would have it).

The fixture is an exact byte-for-byte prefix of a real 2.5V GCD
measurement (BIA-BA-244-01 / A7029), cut off mid charge-step of cycle 13
-- exactly the "file ends mid-step" scenario the task describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from supercap_pipeline.loader import read_tester_txt
from supercap_pipeline.gcd_analysis import segment_steps, flag_ocv_start, assign_rate_stages
from supercap_pipeline.gcd_loader import load_gcd_experiment
from supercap_pipeline.gcd_analysis import compute_gcd_cycle_table

FIXTURES = Path(__file__).parent / "fixtures"


def test_raw_parser_against_output_claude_txt():
    df = read_tester_txt(FIXTURES / "Output_Claude.txt")
    df = df.dropna(subset=["t", "V12", "I12"]).sort_values("t").reset_index(drop=True)
    df["cycle"] = df["cycle"].fillna(0).astype(int)
    df["step"] = df["step"].fillna(0).astype(int)

    df2, steps_df = segment_steps(df)
    steps_df = flag_ocv_start(df2, steps_df)
    steps_df = assign_rate_stages(steps_df)

    # File ends mid charge-step of cycle 13 -> exactly one dropped trailing step.
    dropped = steps_df[~steps_df["complete"]]
    assert len(dropped) == 1
    assert dropped.iloc[0]["cycle"] == 13
    assert dropped.iloc[0]["type"] == "CC"
    assert dropped.iloc[0]["c_d"] == "c"

    # 3 complete rate stages (cycle 13's 2.7123 mA stage never completes a pair).
    complete_charge = steps_df[(steps_df["kind"] == "charge") & steps_df["complete"]]
    rate_groups = sorted(complete_charge["rate_group"].unique())
    assert rate_groups == [0, 1, 2]

    nominal_currents = sorted(complete_charge.groupby("rate_group")["i_nominal_mA"].first())
    expected = [0.2712, 0.6781, 1.3561]
    for got, exp in zip(nominal_currents, expected):
        assert abs(got - exp) / exp < 0.01

    # 4 cycles per rate stage (cycles 1-4, 5-8, 9-12).
    cycles_per_group = complete_charge.groupby("rate_group")["cycle"].nunique()
    assert list(cycles_per_group) == [4, 4, 4]

    # First charge step does not start near 0V (OCV ~1.369V) -> flagged.
    first_charge = steps_df[steps_df["kind"] == "charge"].sort_values("step_id").iloc[0]
    assert bool(first_charge["ocv_start"]) is True

    # Voltage window: charged up to ~2.5V, discharged down to ~0V.
    v_max = df2["V12"].max()
    assert 2.4 < v_max < 2.6


def test_load_gcd_experiment_fixture_folder():
    df, meta = load_gcd_experiment(FIXTURES / "gcd_experiment")

    assert meta["basis"] == "total_active_mass"
    assert meta["mass_total_g"] == pytest.approx(0.0276)
    assert meta["tester_cycles"] is None  # Output-Cycles.txt not present in this fixture

    cycle_df, steps_df = compute_gcd_cycle_table(df, meta)

    assert len(cycle_df) == 12  # cycles 1-12 complete; cycle 13 truncated & dropped
    assert sorted(cycle_df["rate_group"].unique()) == [0, 1, 2]
    assert cycle_df["ocv_start"].sum() == 1
    assert cycle_df.loc[cycle_df["ocv_start"], "cycle"].iloc[0] == 1

    v_max = df["V12"].max()
    v_min = df["V12"].min()
    assert 2.4 < v_max < 2.6
    assert v_min > -0.1  # discharge cutoff near 0V, no significant overshoot
