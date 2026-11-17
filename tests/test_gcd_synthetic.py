"""
tests/test_gcd_synthetic.py
---------------------------
Universality of the GCD evaluation (lab guide GCD section 9) on synthetic
exports of an ideal capacitor C = 1 F with ESR = 1 Ohm and R_leak = 5 kOhm
(see gcd_synth.py), m1 = m2 = 0.4 g, so 0.25 A/g = 100 mA:
  (a) 3 current densities x 3 cycles, 0 -> 2.5 V
  (b) 6 current densities x 10 cycles, non-monotonic, repeated block, U_min = 1.25 V
  (c) rate series + 500-cycle block, C falling linearly to 75 % (end of life)
  (d) floating: 10 holds of 10 h at 2.5 V, check-ups of 3 cycles at two j,
      C falling and ESR rising
  (e) 10 s rest between charge and discharge (ESR with dI = I)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gcd_synth import GcdCell, rate_program, write_masses, write_txt
from supercap_pipeline.export_io import GCD_COLUMNS, read_export
from supercap_pipeline.pipeline import evaluate_gcd, run_gcd_export

M = 0.4
MASSES = dict(m1_g=M, m2_g=M, m_mean_g=M, source="test")


def mA(j: float) -> float:
    return j * 1000.0 * M


def _evaluate(cell: GcdCell, tmp_path) -> dict:
    df = read_export(write_txt(cell.frame(), tmp_path / "Output.txt"), GCD_COLUMNS)
    return evaluate_gcd(df, MASSES)


def _blocks(res) -> list[tuple]:
    g = res["ev"].groupby("block").agg(test=("test_type", "first"), j=("j_block_Ag", "first"),
                                       n=("cycle", "size"), occ=("occurrence", "first"),
                                       umin=("U_min", "first"))
    return [(r.test, round(r.j, 3), r.n, r.occ, r.umin) for r in g.itertuples()]


def _last_cycles(res) -> dict[int, int]:
    ev = res["ev"]
    return {b: int(g["cycle"].max()) for b, g in ev.groupby("block")}


def test_a_three_rates_three_cycles(tmp_path):
    cell = GcdCell()
    cell.rest(10)
    rate_program(cell, [mA(0.25), mA(0.5), mA(1.0)], 3)
    res = _evaluate(cell, tmp_path)

    assert _blocks(res) == [("rate", 0.25, 3, 1, 0.0), ("rate", 0.5, 3, 1, 0.0),
                            ("rate", 1.0, 3, 1, 0.0)]
    rate = res["rate"].set_index("block")
    assert rate["selected_cycle"].to_dict() == _last_cycles(res)
    assert rate["prev_cycle"].to_dict() == {b: c - 1 for b, c in _last_cycles(res).items()}
    assert res["ev"]["used"].all()
    low = rate.loc[1]
    assert low["C"] == pytest.approx(1.0, rel=0.01)
    assert low["C_el"] == pytest.approx(2 * low["C"] / M)
    for r in rate.itertuples():
        assert r.ESR == pytest.approx(1.0, rel=0.02)
        assert r.dI_A == pytest.approx(2 * r.I_abs_mA / 1000, rel=0.001)  # no rest: dI = 2|I|
    assert rate.loc[3, "retention_pct"] == pytest.approx(100 * rate.loc[3, "C"] / low["C"])


def test_b_non_monotonic_repeated_block_u_min_half(tmp_path):
    cell = GcdCell(Vc=1.25)
    cell.rest(10)
    js = [0.5, 0.25, 0.375, 1.0, 0.75, 0.625, 0.375]
    rate_program(cell, [mA(j) for j in js], 10, u_min=1.25)
    res = _evaluate(cell, tmp_path)

    occ = [1, 1, 1, 1, 1, 1, 2]
    assert _blocks(res) == [("rate", j, 10, o, 1.25) for j, o in zip(js, occ)]
    rate = res["rate"].set_index("block")
    assert rate["selected_cycle"].to_dict() == _last_cycles(res)
    low = rate.loc[2]                                   # 0.25 A/g
    assert low["C"] == pytest.approx(1.0, rel=0.01)
    assert (rate["ESR"] - 1.0).abs().max() < 0.02
    assert (rate["U_min"] == 1.25).all()
    assert rate.loc[1, "retention_ref_block"] == 2      # reference = lowest j
    # non-standard current densities are flagged, standard ones are not
    flagged = {round(r.j_block_Ag, 3) for r in rate.itertuples()
               if any(t.startswith("non-standard") for _, t in r.block_flags)}
    assert flagged == {0.375, 0.75, 0.625}


def test_c_rate_series_then_cycling_end_of_life(tmp_path):
    cell = GcdCell(pts=25)
    cell.rest(10)
    rate_program(cell, [mA(0.25), mA(1.0)], 3)
    n = 500
    for k in range(n):
        cell.C = 1.0 - 0.25 * k / (n - 1)
        cell.cycle_(mA(0.5), 2.5, 0.0)
    res = _evaluate(cell, tmp_path)

    assert _blocks(res) == [("rate", 0.25, 3, 1, 0.0), ("rate", 1.0, 3, 1, 0.0),
                            ("cycling", 0.5, 500, 1, 0.0)]
    (cyc,) = res["cycling"]
    assert cyc["reference"] == 1
    # true retention C_k / C_1 = 1 - 0.25 (k-1)/499 crosses 80 % at k = 400.2
    assert cyc["eol_criterion"] == "retention < 80 %"
    assert cyc["eol_eol_x"] == pytest.approx(400.2, abs=1.0)
    t = res["cyc_t"]
    assert len(t) == 500
    assert t["retention_pct"].iloc[-1] == pytest.approx(75.0, abs=0.5)
    assert (t["ESR_rel"] - 1).abs().max() < 0.02
    assert res["rate"]["C"].iloc[0] == pytest.approx(1.0, rel=0.01)


def test_d_floating(tmp_path):
    cell = GcdCell()
    cell.rest(10)
    for k in range(11):
        cell.C, cell.R = 1.0 - 0.03 * k, 1.0 + 0.08 * k
        if k:
            cell.cc(mA(0.25), 2.5)          # charge to the hold voltage
            cell.hold(2.5, 10 * 3600.0)     # 10 h float
            cell.cc(-mA(0.25), 0.0)         # back to U_min
        rate_program(cell, [mA(0.25), mA(1.0)], 3)
    res = _evaluate(cell, tmp_path)

    assert res["n_holds"] == 10
    assert set(res["ev"]["test_type"]) == {"floating"}
    holds = res["holds"]
    assert holds["duration_h"].to_numpy() == pytest.approx(np.full(10, 10.0), abs=0.01)
    assert holds["V_hold"].to_numpy() == pytest.approx(np.full(10, 2.5))
    # steady state: I = U / (R + R_leak) [A] -> mA
    leak_mA = 1000 * 2.5 / (5000 + 1 + 0.08 * np.arange(1, 11))
    assert holds["leak_mA"].to_numpy() == pytest.approx(leak_mA, rel=0.01)
    assert holds["leak_mA_per_g"].to_numpy() == pytest.approx(leak_mA / M, rel=0.01)
    fl = res["fl_t"]
    assert len(fl) == 22                         # 11 check-ups x 2 current densities
    assert sorted(fl["hold_time_h"].unique()) == pytest.approx(list(np.arange(0, 101, 10.0)), abs=0.01)
    low = fl[fl["j_block_Ag"].round(3) == 0.25].sort_values("checkup")
    assert low["C"].iloc[0] == pytest.approx(1.0, rel=0.01)
    assert low["ESR_rel"].iloc[-1] == pytest.approx(1.8, rel=0.02)
    assert low["retention_pct"].to_numpy() == pytest.approx(100 * (1 - 0.03 * np.arange(11)), abs=0.5)
    # each check-up uses its last complete cycle per current density
    last = res["ev"].groupby(["checkup", "block"])["cycle"].max()
    assert set(fl["cycle"]) == set(last.to_numpy())
    for f in res["floating"]:
        # retention 1 - 0.03 k crosses 80 % at k = 6.67 -> 66.7 h
        assert f["eol_criterion"] == "retention < 80 %"
        assert f["eol_eol_x"] == pytest.approx(66.7, abs=10.0)


def test_e_rest_between_charge_and_discharge(tmp_path):
    cell = GcdCell()
    cell.rest(10)
    rate_program(cell, [mA(0.25), mA(0.5), mA(1.0)], 3, rest_s=10.0)
    res = _evaluate(cell, tmp_path)

    assert [b[:3] for b in _blocks(res)] == [("rate", 0.25, 3), ("rate", 0.5, 3), ("rate", 1.0, 3)]
    ev = res["ev"]
    assert ev["has_rest"].all()
    rate = res["rate"]
    for r in rate.itertuples():
        assert r.I_before_mA == pytest.approx(0.0, abs=1e-9)
        assert r.dI_A == pytest.approx(r.I_abs_mA / 1000, rel=0.001)      # after a rest dI = |I|
        assert r.ESR == pytest.approx(1.0, rel=0.02)
    assert rate["C"].iloc[0] == pytest.approx(1.0, rel=0.01)


def test_end_to_end_outputs(tmp_path):
    cell = GcdCell()
    cell.rest(10)
    rate_program(cell, [mA(0.25), mA(0.5)], 3)
    folder = tmp_path / "02_GCD"
    write_txt(cell.frame(), folder / "Output.txt")
    write_masses(folder, M)
    r = run_gcd_export(folder / "Output.txt", output_dir=tmp_path / "out")

    out = r["out"]
    assert out == tmp_path / "out" / "02_GCD" / "gcd"     # no info.json -> folder name as sample id
    sheets = pd.read_excel(out / "results.xlsx", sheet_name=None)
    assert {"Cycles", "Rate", "Flags", "Meta", "Warnings"} <= set(sheets)
    assert "Cycling" not in sheets and "Floating" not in sheets
    assert len(sheets["Cycles"]) == 6 and len(sheets["Rate"]) == 2
    for name in ("Cycles", "Rate", "Flags", "Meta"):
        assert (out / f"results_{name}.csv").exists()
    assert (out / "gcd.log").exists()
    plots = {p.stem for p in (out / "plots").glob("*.png")}
    assert {"gcd_profiles_time", "gcd_profiles_charge", "gcd_rate_capability", "gcd_esr",
            "gcd_ragone", "gcd_qc_0.25Ag", "gcd_qc_0.5Ag"} == plots
    assert {p.stem for p in (out / "plots").glob("*.svg")} == plots
    assert {p.stem for p in (out / "plots").glob("*.xlsx")} == plots


def test_missing_masses_stop(tmp_path):
    cell = GcdCell()
    cell.rest(10)
    rate_program(cell, [mA(0.25)], 2)
    folder = tmp_path / "02_GCD"
    write_txt(cell.frame(), folder / "Output.txt")
    with pytest.raises(FileNotFoundError, match="Input-Variables"):
        run_gcd_export(folder / "Output.txt", output_dir=tmp_path / "out")
