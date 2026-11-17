"""
tests/test_gcd_reference.py
---------------------------
Regression against the expected values of lab guide GCD section 10 for the
GCD export of BIA-VR-009-01-05 (Output.txt, 887 data rows), with
m1 = m2 = 0.032164 g as in the guide.

The raw file is not in the repository. Point SUPERCAP_REF_GCD at it (the
Output.txt), or put it at tests/fixtures/BIA-VR-009-01-05/gcd/Output.txt;
otherwise the tests are skipped. The CV comparison test additionally needs
SUPERCAP_REF_XLSX (the CV export of the same cell, see test_cv_reference).

Tolerance: 0.5 %; values printed in the guide with few digits additionally
allow half a unit of the last printed digit.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from supercap_pipeline.export_io import GCD_COLUMNS, read_export
from supercap_pipeline.pipeline import evaluate_gcd, run_cv_export, run_gcd_export

REF = Path(os.environ.get(
    "SUPERCAP_REF_GCD",
    Path(__file__).parent / "fixtures" / "BIA-VR-009-01-05" / "gcd" / "Output.txt"))
REF_CV = Path(os.environ.get("SUPERCAP_REF_XLSX", "__missing__"))
pytestmark = pytest.mark.skipif(not REF.exists(), reason=f"reference file not found: {REF}")

M = 0.032164
MASSES = dict(m1_g=M, m2_g=M, m_mean_g=M, source="section 10")

# j -> (I mA, selected cycle, charge rows, discharge rows, crossing rows a/b)
PROGRAM = {
    0.1: (3.217, 4, (157, 184), (185, 201), (199, 200)),
    0.5: (16.09, 8, (330, 355), (356, 373), (371, 372)),
    1.0: (32.18, 12, (504, 528), (529, 548), (546, 547)),
    2.5: (80.44, 16, (678, 701), (702, 720), (718, 719)),
    3.11: (100.04, 20, (847, 869), (870, 889), (886, 887)),
}
# j -> U1, dU_IR, ESR, Q_ch, Q_dis, t_dis, E_ch, E_dis  (value, last printed digit)
TABLE2 = {
    0.1: [(2.4820, 1e-4), (0.0180, 1e-4), (2.80, 1e-2), (0.8161, 1e-4), (0.7046, 1e-4),
          (788.6, 0.1), (1.1892, 1e-4), (0.8447, 1e-4)],
    0.5: [(2.4099, 1e-4), (0.0902, 1e-4), (2.80, 1e-2), (0.6723, 1e-4), (0.6538, 1e-4),
          (146.3, 0.1), (0.9777, 1e-4), (0.7812, 1e-4)],
    1.0: [(2.3171, 1e-4), (0.1830, 1e-4), (2.84, 1e-2), (0.6039, 1e-4), (0.5966, 1e-4),
          (66.75, 1e-2), (0.9278, 1e-4), (0.6803, 1e-4)],
    2.5: [(2.0177, 1e-4), (0.4826, 1e-4), (3.00, 1e-2), (0.4474, 1e-4), (0.4457, 1e-4),
          (19.94, 1e-2), (0.7965, 1e-4), (0.4206, 1e-4)],
    3.11: [(1.8871, 1e-4), (0.6131, 1e-4), (3.06, 1e-2), (0.3903, 1e-4), (0.3893, 1e-4),
           (14.01, 1e-2), (0.7284, 1e-4), (0.3353, 1e-4)],
}
T2_COLS = ["U1", "dU_IR", "ESR", "Q_ch", "Q_dis", "t_dis", "E_ch", "E_dis"]
# j -> C, C_E, C_cell_sp, C_el, retention %, eta_C, eta_E, E_sp, P_sp
TABLE3 = {
    0.1: [(1.022, 1e-3), (0.987, 1e-3), (15.9, 0.1), (63.5, 0.1), (100, 1), (0.863, 1e-3),
          (0.710, 1e-3), (13.1, 0.1), (60, 1)],
    0.5: [(0.977, 1e-3), (0.968, 1e-3), (15.2, 0.1), (60.7, 0.1), (95.6, 0.1), (0.972, 1e-3),
          (0.799, 1e-3), (12.1, 0.1), (299, 1)],
    1.0: [(0.927, 1e-3), (0.912, 1e-3), (14.4, 0.1), (57.6, 0.1), (90.7, 0.1), (0.988, 1e-3),
          (0.733, 1e-3), (10.6, 0.1), (570, 1)],
    2.5: [(0.795, 1e-3), (0.744, 1e-3), (12.4, 0.1), (49.4, 0.1), (77.8, 0.1), (0.996, 1e-3),
          (0.528, 1e-3), (6.54, 1e-2), (1180, 10)],
    3.11: [(0.743, 1e-3), (0.678, 1e-3), (11.5, 0.1), (46.2, 0.1), (72.7, 0.1), (0.998, 1e-3),
           (0.460, 1e-3), (5.21, 1e-2), (1340, 10)],
}
T3_COLS = ["C", "C_E", "C_cell_sp", "C_el", "retention_pct", "eta_C", "eta_E", "E_sp", "P_sp"]
PREV_C = {0.1: (3, 1.0250), 0.5: (7, 0.9781), 1.0: (11, 0.9274), 2.5: (15, 0.7954),
          3.11: (19, 0.7428)}


def _close(got, expected, last_digit):
    return abs(got - expected) <= max(0.005 * abs(expected), 0.5 * last_digit + 1e-12)


@pytest.fixture(scope="module")
def result():
    df = read_export(REF, GCD_COLUMNS)
    assert len(df) == 887
    return evaluate_gcd(df, MASSES)


@pytest.fixture(scope="module")
def by_j(result):
    rate = result["rate"]
    return {min(PROGRAM, key=lambda j: abs(j - r["j_block_Ag"])): r
            for r in rate.to_dict("records")}


def test_program_detected(result, by_j):
    ev = result["ev"]
    assert result["n_holds"] == 0
    assert ev["block"].nunique() == 5
    assert (ev.groupby("block").size() == 4).all()
    assert set(ev["test_type"]) == {"rate"}
    assert (ev["U_min"] == 0).all()
    assert not ev["has_rest"].any()
    c1 = ev[ev["cycle"] == 1].iloc[0]
    assert not c1["used"] and c1["incomplete_reason"] == "starts outside window"
    assert c1["V_pre"] == pytest.approx(-2.1636, abs=1e-3)
    assert sorted(by_j) == sorted(PROGRAM)


@pytest.mark.parametrize("j", sorted(PROGRAM))
def test_selected_cycles_and_rows(by_j, j):
    I, cyc, ch, dis, ab = PROGRAM[j]
    r = by_j[j]
    assert r["j_block_Ag"] == pytest.approx(j, rel=0.005)
    assert r["I_block_mA"] == pytest.approx(I, rel=0.005)
    assert r["selected_cycle"] == cyc
    assert (r["row_ch_start"], r["row_ch_end"]) == ch
    assert (r["row_dis_start"], r["row_dis_end"]) == dis
    assert (r["row_a"], r["row_b"]) == ab


def test_worked_example_cycle_4(result):
    c = result["ev"].set_index("cycle").loc[4]
    assert c["U_max"] == pytest.approx(2.500003, abs=1e-6)
    assert c["U1"] == pytest.approx(2.482009, abs=1e-6)
    assert c["t_U1_ms"] == pytest.approx(1.43, abs=0.01)
    assert c["dU_IR"] == pytest.approx(0.017994, abs=1e-6)
    assert c["ESR"] == pytest.approx(2.797, abs=0.001)
    assert c["Q_dis"] == pytest.approx(0.704620, abs=1e-5)
    assert c["t_dis"] == pytest.approx(788.58, abs=0.01)
    assert c["Q_ch"] == pytest.approx(0.816108, abs=1e-6)
    assert c["E_ch"] == pytest.approx(1.189242, abs=1e-6)
    assert c["E_dis"] == pytest.approx(0.844705, abs=1e-5)
    assert c["C"] == pytest.approx(1.022, abs=5e-4)
    assert c["C_E"] == pytest.approx(0.987, abs=5e-4)
    assert c["I_leak_mA"] == pytest.approx(0.24, abs=0.005)


@pytest.mark.parametrize("j", sorted(TABLE2))
def test_table_ir_charges_energies(by_j, j):
    r = by_j[j]
    for col, (exp, d) in zip(T2_COLS, TABLE2[j]):
        assert _close(r[col], exp, d), (j, col, r[col], exp)


@pytest.mark.parametrize("j", sorted(TABLE3))
def test_table_capacitance_energy_power(by_j, j):
    r = by_j[j]
    for col, (exp, d) in zip(T3_COLS, TABLE3[j]):
        assert _close(r[col], exp, d), (j, col, r[col], exp)


@pytest.mark.parametrize("j", sorted(PREV_C))
def test_second_to_last_cycle_and_stability(by_j, j):
    cyc, c_prev = PREV_C[j]
    r = by_j[j]
    assert r["prev_cycle"] == cyc
    assert _close(r["C_prev"], c_prev, 1e-4)
    assert r["stability"] < 0.003
    assert not any(t.startswith("not stable") for _, t in r["block_flags"])


def test_expected_flags(result, by_j):
    ev = result["ev"].set_index("cycle")

    def texts(cycle):
        return [t for _, t in ev.loc[cycle, "cycle_flags"]]

    assert texts(1) == ["starts outside window - not used"]
    assert any(t.startswith("non-standard current density") for _, t in by_j[3.11]["block_flags"])
    assert not any(t.startswith("non-standard") for j in (0.1, 0.5, 1.0, 2.5)
                   for _, t in by_j[j]["block_flags"])
    f4 = texts(4)
    assert any(t.startswith("eta_C = 0.863") for t in f4)
    assert any(t.startswith("low efficiency") for t in f4)
    assert not any(t.startswith("eta_C") for t in texts(8))   # 0.972 >= 0.97
    for cyc, should in ((4, False), (8, False), (12, False), (16, True), (20, True)):
        assert any(t.startswith("curve shape") for t in texts(cyc)) == should, cyc
        assert any(t.startswith("resistance-limited") for t in texts(cyc)) == (cyc == 20), cyc
        assert any(t.startswith("logging too coarse") for t in texts(cyc)), cyc
    assert not any(s == "flag" and t.startswith("ESR not constant") for s, t in result["series_flags"])
    assert result["rate"].attrs["esr_spread"] == pytest.approx(0.096, abs=0.002)


@pytest.mark.skipif(not REF_CV.exists(), reason="CV reference (SUPERCAP_REF_XLSX) not available")
def test_comparison_with_cv(tmp_path):
    # copy both measurements so their Input-Variables.txt / info.json travel along
    for src, name in ((REF_CV.parent, "01_CV"), (REF.parent, "02_GCD")):
        shutil.copytree(src, tmp_path / "cell" / name,
                        ignore=shutil.ignore_patterns("Output-*.txt", "log.txt", "T-Controller*"))
    run_cv_export(tmp_path / "cell" / "01_CV" / REF_CV.name, output_dir=tmp_path / "out")
    r = run_gcd_export(tmp_path / "cell" / "02_GCD" / REF.name, output_dir=tmp_path / "out")
    cmp = r["cv_comparison"]
    assert cmp is not None
    assert cmp["cv_window"].startswith("+-U_max")
    assert cmp["cv_nu_mVs"] == 1
    assert cmp["cv_C_el"] == pytest.approx(56.9, abs=0.05)
    assert cmp["gcd_C_el"] == pytest.approx(63.5, abs=0.06)
    assert cmp["deviation"] == pytest.approx(0.117, abs=0.003)
    assert cmp["severity"] == "info"
    assert "CV window +-U_max, GCD window 0->U_max" in cmp["text"]
