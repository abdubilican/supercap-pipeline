"""
tests/test_cv_synthetic.py
--------------------------
Universality of the CV segmentation and evaluation (lab guide 2-5) on
synthetic exports of an ideal capacitor (C = 1 F, R = 1 Ohm), see cv_synth.py:
  (a) symmetric program, 3 windows x 2 scan rates x 3 cycles
  (b) one-sided 0 -> 2.5 V, 4 scan rates x 7 cycles
  (c) blocks in non-monotonic order with a repeated block
plus the end-to-end run (txt and xlsx input, output files).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cv_synth import one_sided_program, symmetric_program, write_masses, write_txt, write_xlsx
from supercap_pipeline.cv_analysis import evaluate_blocks
from supercap_pipeline.cv_io import read_export
from supercap_pipeline.cv_segmentation import (
    detect_blocks, drop_transition_sweeps, find_sweeps, pair_cycles,
)
from supercap_pipeline.pipeline import run_cv_export

C_TRUE = 1.0
MASSES = dict(m1_g=0.002, m2_g=0.002, m_mean_g=0.002, source="test")


def _evaluate(cell, tmp_path):
    df = read_export(write_txt(cell.frame(), tmp_path / "Output.txt"))
    sweeps = drop_transition_sweeps(find_sweeps(df))
    cycles = detect_blocks(pair_cycles(sweeps))
    return df, cycles, evaluate_blocks(df, sweeps, cycles, MASSES)


def _check_blocks(blocks, expected):
    got = [(b["window_type"], b["U_max"], b["nu_nom_mVs"], b["occurrence"]) for b in blocks]
    assert got == expected


def test_symmetric_program(tmp_path):
    program = [(u, nu) for u in (1.0, 1.5, 2.5) for nu in (0.001, 0.010)]
    cell = symmetric_program(program, n_cycles=3)
    df, cycles, blocks = _evaluate(cell, tmp_path)

    _check_blocks(blocks, [("symmetric", u, nu * 1000, 1) for u, nu in program])
    for b in blocks:
        assert b["n_complete"] == 3
        # last cycle used: the highest instrument cycle carrying this block's sweeps
        block_cycles = cycles[cycles["block"] == b["block"]]
        last = block_cycles.iloc[-1]
        assert min(b["up"]["row_start"], b["down"]["row_start"]) == last["row_start"]
        in_block = df["excel_row"].between(block_cycles["row_start"].min(), last["row_end"])
        assert b["up"]["cycle"] == df.loc[in_block, "cycle"].max()
        if b["nu_nom_mVs"] == 1.0:
            for p in ("+", "-"):
                assert b["pol"][p]["C"] == pytest.approx(C_TRUE, rel=0.01)
            assert b["C"] == pytest.approx(C_TRUE, rel=0.01)
            assert b["C_el"] == pytest.approx(4 * b["C"] / (2 * MASSES["m_mean_g"]))
            assert b["stability_text"] != "n/a"


def test_one_sided_program(tmp_path):
    rates = [0.001, 0.005, 0.020, 0.050]
    cell = one_sided_program(2.5, rates, n_cycles=7)
    df, cycles, blocks = _evaluate(cell, tmp_path)

    _check_blocks(blocks, [("one-sided", 2.5, nu * 1000, 1) for nu in rates])
    for b in blocks:
        assert list(b["pol"]) == ["+"]
        assert b["n_complete"] == 7
        assert b["up"]["cycle"] == df.loc[df["excel_row"] <= b["down"]["row_end"], "cycle"].max()
    slow = blocks[0]
    assert slow["C"] == pytest.approx(C_TRUE, rel=0.01)
    assert slow["pol"]["+"]["eta"] == pytest.approx(1.0, abs=0.01)
    # ideal capacitor: C falls (slightly) with nu, never a trend flag
    assert not any("scan-rate trend" in f for b in blocks for f in b["flags"])


def test_non_monotonic_order_and_repeated_block(tmp_path):
    program = [(2.0, 0.010), (1.0, 0.001), (2.0, 0.001), (1.0, 0.010), (2.0, 0.010)]
    cell = symmetric_program(program, n_cycles=3)
    _, _, blocks = _evaluate(cell, tmp_path)

    _check_blocks(blocks, [
        ("symmetric", 2.0, 10.0, 1),
        ("symmetric", 1.0, 1.0, 1),
        ("symmetric", 2.0, 1.0, 1),
        ("symmetric", 1.0, 10.0, 1),
        ("symmetric", 2.0, 10.0, 2),
    ])
    assert [b["block"] for b in blocks] == [1, 2, 3, 4, 5]
    for b in blocks:
        if b["nu_nom_mVs"] == 1.0:
            assert b["C"] == pytest.approx(C_TRUE, rel=0.01)
    # the slowest-rate block is the reported one per window
    reported = {(b["U_max"], b["nu_nom_mVs"]) for b in blocks if b["reported_for_window"]}
    assert reported == {(1.0, 1.0), (2.0, 1.0)}


def test_ideal_capacitor_has_no_spurious_flags(tmp_path):
    cell = symmetric_program([(1.0, 0.001)], n_cycles=3)
    _, _, blocks = _evaluate(cell, tmp_path)
    b = blocks[0]
    assert b["flags"] == []
    for d in b["pol"].values():
        assert d["flags"] == []
        assert d["eta"] == pytest.approx(1.0, abs=0.01)


def test_single_cycle_block_is_flagged(tmp_path):
    cell = symmetric_program([(1.0, 0.001)], n_cycles=1)
    _, _, blocks = _evaluate(cell, tmp_path)
    assert "single cycle, may not be stabilised" in blocks[0]["flags"]
    assert blocks[0]["stability_text"] == "n/a"


def test_end_to_end_txt_and_xlsx_agree(tmp_path):
    cell = symmetric_program([(1.0, 0.001), (1.0, 0.010)], n_cycles=2)
    frame = cell.frame()

    txt_dir = tmp_path / "txt" / "01_CV"
    txt_dir.mkdir(parents=True)
    write_txt(frame, txt_dir / "Output.txt")
    write_masses(txt_dir)
    r_txt = run_cv_export(txt_dir / "Output.txt", output_dir=tmp_path / "out_txt")

    xl_dir = tmp_path / "xl"
    xl_dir.mkdir()
    write_xlsx(frame, xl_dir / "SYN-001.xlsx")
    write_masses(xl_dir)
    r_xl = run_cv_export(xl_dir / "SYN-001.xlsx", output_dir=tmp_path / "out_xl")

    cols = ["up_row_a", "up_row_b", "down_row_a", "down_row_b", "Q_dis_mAh", "C_F", "C_el_Fg"]
    pd.testing.assert_frame_equal(r_txt["results"][cols], r_xl["results"][cols])

    out = r_xl["out"]
    assert out == tmp_path / "out_xl" / "SYN-001" / "cv"
    assert (out / "results.xlsx").exists() and (out / "results.csv").exists()
    assert (out / "cv.log").read_text(encoding="utf-8").count("Dropped") == 1
    sheets = pd.read_excel(out / "results.xlsx", sheet_name=None)
    assert set(sheets) == {"Results", "Summary", "Meta", "Warnings"}
    assert len(sheets["Results"]) == 4 and len(sheets["Summary"]) == 2
    pngs = {p.name for p in (out / "plots").glob("*.png")}
    assert {"cv_1mVs.png", "cv_10mVs.png", "rate_capability.png", "window_plot.png",
            "normalised_pm1.00V.png"} <= pngs
    assert len([p for p in pngs if p.startswith("control_block")]) == 2
    assert len(list((out / "plots").glob("*.svg"))) == len(pngs)
    assert {p.name[:-4] for p in (out / "plots").glob("*.csv")} == {p[:-4] for p in pngs}
    csv_text = (out / "plots" / "cv_1mVs.csv").read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0] == "V12 (V);I12/m_mean (A/g);cycle;Qcd (mAh)"
    assert "e-" not in csv_text  # no exponent notation
    cv1 = pd.read_csv(out / "plots" / "cv_1mVs.csv", sep=";", encoding="utf-8-sig")
    m_mean = r_xl["masses"]["m_mean_g"]
    b = r_xl["blocks"][0]
    src = pd.read_excel(xl_dir / "SYN-001.xlsx", sheet_name="Output", skiprows=[1])
    src.columns = [c.strip() for c in src.columns]
    first = src.iloc[b["up"]["row_start"] - 3]  # first plotted row = start of the up sweep
    assert cv1["V12 (V)"].iloc[0] == pytest.approx(first.V12)
    assert cv1["I12/m_mean (A/g)"].iloc[0] == pytest.approx(first.I12 / (1000 * m_mean))
    assert cv1["Qcd (mAh)"].iloc[0] == pytest.approx(first.Qcd)
    assert cv1["cycle"].iloc[0] == first.cycle
    ctrl = pd.read_csv(next((out / "plots").glob("control_block01_*.csv")),
                       sep=";", encoding="utf-8-sig")
    assert (ctrl["point"] == "0 V crossing (interpolated)").sum() == 2
    assert set(ctrl["discharge_area"].dropna()) == {"+", "-"}

    # Excel rows point at the right raw rows: re-interpolate from the raw sheet
    raw = pd.read_excel(xl_dir / "SYN-001.xlsx", sheet_name="Output", header=0, skiprows=[1])
    raw.columns = [c.strip() for c in raw.columns]
    row = r_xl["results"].iloc[0]
    a, b = raw.iloc[int(row.down_row_a) - 3], raw.iloc[int(row.down_row_b) - 3]
    q0 = a.Qcd - a.V12 * (b.Qcd - a.Qcd) / (b.V12 - a.V12)
    assert row.down_V_a == pytest.approx(a.V12) and row.down_Qcd_a_mAh == pytest.approx(a.Qcd)
    assert row.down_Q0_mAh == pytest.approx(q0 - row.down_Qcd_first_mAh)


def test_missing_mass_file_stops(tmp_path):
    cell = symmetric_program([(1.0, 0.010)], n_cycles=2)
    folder = tmp_path / "m"
    folder.mkdir()
    write_txt(cell.frame(), folder / "Output.txt")
    with pytest.raises(FileNotFoundError, match="Input-Variables"):
        run_cv_export(folder / "Output.txt", output_dir=tmp_path / "out")


def test_asymmetric_masses_report_exact_value(tmp_path):
    cell = symmetric_program([(1.0, 0.001)], n_cycles=2)
    folder = tmp_path / "m"
    folder.mkdir()
    write_txt(cell.frame(), folder / "Output.txt")
    write_masses(folder, m1="0.0010", m2="0.0020")
    r = run_cv_export(folder / "Output.txt", output_dir=tmp_path / "out")
    b = r["blocks"][0]
    assert b["C_el_exact"] == pytest.approx(b["C"] * (1 / 0.001 + 1 / 0.002))
    assert any("m1 - m2" in w for w in
               pd.read_excel(r["out"] / "results.xlsx", sheet_name="Warnings")["warning"])
