"""
tests/test_peis_synthetic.py
----------------------------
PEIS evaluation on spectra generated from the lab-guide model itself
(peis_synth.py): parameter recovery, KK residuals, and universality
(lab guide PEIS section 9):
  (a) 6 spectra at 0, 0.5, 1, 1.5, 2, 2.5 V
  (b) repeated voltages in non-monotonic order (0, 2.5, 0, 1, 0 V)
  (c) a spectrum without arc (R_c = 0) -> R0-TLMQ0, knee NaN
  (d) 5 points per decade, 1 MHz - 1 mHz
  (e) inductive points at the highest frequencies (L = 300 nH): excluded,
      or fitted with L0 when FIT_INCLUDE_L
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peis_synth import TRUE, ImpedanceFile
from supercap_pipeline.peis_spectra import read_impedance_export
from supercap_pipeline.pipeline import evaluate_peis, run_peis_export

M = 0.032164
MASSES = dict(m1_g=M, m2_g=M, m_mean_g=M, source="test")
AREA = dict(area_cm2=2.5447, diameter_mm=18.0, source="test")


def _evaluate(imp: ImpedanceFile, tmp_path, include_L=None) -> dict:
    df = read_impedance_export(imp.write(tmp_path / "03_PEIS"))
    return evaluate_peis(df, MASSES, AREA, include_L=include_L)


def _params(row) -> dict:
    return {n: row[f"fit_{n}"] for n in TRUE if f"fit_{n}" in row}


def test_model_recovery_and_kk(tmp_path):
    imp = ImpedanceFile()
    imp.spectrum(0.0)
    r = _evaluate(imp, tmp_path)
    s = r["spectra"].iloc[0]
    assert s["fit_circuit"] == "R0-p(R1,CPE1)-TLMQ0" and s["fit_success"]
    for n, true in TRUE.items():
        assert s[f"fit_{n}"] == pytest.approx(true, rel=0.03), n
    assert s["kk_max_res_pct"] < 1.0
    assert s["n_excluded"] == 1                      # the deliberately bad first point
    pts = r["points"]
    first = pts[pts["excel_row"] == 3].iloc[0]
    assert first["excluded"] and "amplitude" in first["exclude_reason"] \
        and "SFDR" in first["exclude_reason"]
    assert np.isnan(first["kk_res_re"]) and np.isnan(first["fit_res_re"])
    assert s["points_per_decade"] == pytest.approx(10, abs=0.01)
    assert s["arc"] and np.isfinite(s["f_knee"])
    # graphical estimates close to the model
    assert s["R_s_circle"] == pytest.approx(TRUE["R_s"], abs=0.1)
    assert s["R_c_circle"] == pytest.approx(TRUE["R_c"], rel=0.1)
    assert s["EDR"] == pytest.approx(s["fit_R_s"] + s["fit_R_c"] + s["fit_R_ion"] / 3)
    assert not any(t.startswith("fit:") for t in r["flags"]["flag"])


def test_a_six_voltages(tmp_path):
    imp = ImpedanceFile()
    volts = [0, 0.5, 1, 1.5, 2, 2.5]
    for v in volts:
        imp.spectrum(v, params={"Q_dl": 1.0 + 0.1 * v})
    r = _evaluate(imp, tmp_path)
    s = r["spectra"]
    assert s["label"].tolist() == ["0 V", "0.5 V", "1 V", "1.5 V", "2 V", "2.5 V"]
    assert s["slug"].tolist() == ["0V", "0p5V", "1V", "1p5V", "2V", "2p5V"]
    assert s["fit_success"].all()
    assert s["fit_Q_dl"].to_numpy() == pytest.approx(1.0 + 0.1 * np.array(volts), rel=0.03)
    assert r["reversibility"].empty
    # colours run from light to dark with voltage, one per label
    assert len(set(map(tuple, s["color"]))) == 6


def test_b_repeated_voltages(tmp_path):
    imp = ImpedanceFile()
    for v, qdl in ((0, 1.0), (2.5, 1.3), (0, 0.99), (1, 1.1), (0, 0.80)):
        imp.spectrum(v, params={"Q_dl": qdl})
    r = _evaluate(imp, tmp_path)
    s = r["spectra"]
    assert s["label"].tolist() == ["0 V (1)", "2.5 V", "0 V (2)", "1 V", "0 V (3)"]
    assert s["slug"].tolist() == ["0V_1", "2p5V", "0V_2", "1V", "0V_3"]
    assert s["order"].tolist() == [1, 2, 3, 4, 5]
    assert s.loc[s["V_dc"] == 0, "marker"].tolist() == ["o", "s", "^"]
    rev = r["reversibility"]
    assert set(rev["first"]) == {"0 V (1)"} and set(rev["last"]) == {"0 V (3)"}
    cs = rev[rev["quantity"] == "C_s(f_min)"].iloc[0]
    assert cs["flag"]                                 # Q_dl 1.0 -> 0.8: about -20 %
    assert any(t.startswith("not reversible") for t in r["flags"]["flag"])


def test_c_no_arc(tmp_path):
    imp = ImpedanceFile()
    imp.spectrum(0.0, params={"R_c": 0.0})
    r = _evaluate(imp, tmp_path)
    s = r["spectra"].iloc[0]
    assert not s["arc"] and s["R_c_circle"] == 0
    assert s["fit_circuit"] == "R0-TLMQ0" and s["fit_success"]
    assert np.isnan(s["f_knee"]) and np.isnan(s["row_knee"])
    for n in ("R_s", "R_ion", "Q_dl", "alpha_dl"):
        assert s[f"fit_{n}"] == pytest.approx(TRUE[n], rel=0.03), n
    flags = r["flags"]["flag"].tolist()
    assert any(t.startswith("no high-frequency arc") for t in flags)


def test_d_five_points_per_decade_wide_range(tmp_path):
    imp = ImpedanceFile()
    imp.spectrum(0.0, f_max=1e6, f_min=1e-3, ppd=5)
    r = _evaluate(imp, tmp_path)
    s = r["spectra"].iloc[0]
    assert s["n_points"] == 46
    assert s["points_per_decade"] == pytest.approx(5, abs=0.01)
    assert s["f_max"] == pytest.approx(1e6) and s["f_min"] == pytest.approx(1e-3)
    for n, true in TRUE.items():
        assert s[f"fit_{n}"] == pytest.approx(true, rel=0.03), n


def test_e_inductive_points(tmp_path):
    L = 300e-9
    imp = ImpedanceFile()
    imp.spectrum(0.0, f_max=1e6, f_min=1e-2, L=L, bad_first=False)
    r = _evaluate(imp, tmp_path)
    pts, s = r["points"], r["spectra"].iloc[0]
    ind = pts["inductive"].to_numpy()
    assert ind[0] and ind.sum() >= 2
    assert np.all(ind[:ind.sum()]) and not ind[ind.sum():].any()     # contiguous from f_max
    assert pts.loc[pts["inductive"], "excluded"].all()
    assert pts.loc[pts["inductive"], "exclude_reason"].str.contains("inductive").all()
    assert s["fit_circuit"] == "R0-p(R1,CPE1)-TLMQ0"

    r2 = _evaluate(imp, tmp_path / "L", include_L=True)
    s2 = r2["spectra"].iloc[0]
    assert s2["fit_circuit"] == "L0-R0-p(R1,CPE1)-TLMQ0" and s2["fit_success"]
    assert not r2["points"]["excluded"].any()
    assert s2["fit_L"] == pytest.approx(L, rel=0.05)
    for n, true in TRUE.items():
        assert s2[f"fit_{n}"] == pytest.approx(true, rel=0.03), n


def test_end_to_end_outputs(tmp_path):
    imp = ImpedanceFile()
    for v in (0, 2.5, 0):
        imp.spectrum(v)
    folder = tmp_path / "03_PEIS"
    imp.write(folder)
    r = run_peis_export(folder / "Output-Impedance.txt", output_dir=tmp_path / "out")
    out = r["out"]
    assert out == tmp_path / "out" / "03_PEIS" / "peis"
    sheets = pd.read_excel(out / "results.xlsx", sheet_name=None)
    assert {"Spectra", "Points", "Reversibility", "Comparison", "Flags", "Meta"} <= set(sheets)
    assert len(sheets["Spectra"]) == 3 and len(sheets["Points"]) == 3 * 71
    meta = dict(zip(sheets["Meta"]["key"], sheets["Meta"]["value"]))
    assert "default ELECTRODE_DIAMETER_MM = 18 mm" in meta["area_source"]
    assert meta["impedance.py"] == "1.7.1"
    assert (out / "peis.log").exists()
    names = {p.stem for p in (out / "plots").glob("*.png")}
    overlays = {"peis_nyquist", "peis_nyquist_hf", "peis_bode", "peis_complex_capacitance",
                "peis_kk_residuals", "peis_cs_vs_f", "peis_parameters_vs_voltage",
                "peis_nyquist_area", "peis_complex_capacitance_specific"}
    per = {f"peis_{k}_{slug}" for k in ("fit_nyquist", "fit_bode", "kk")
           for slug in ("0V_1", "2p5V", "0V_2")}
    assert names == overlays | per
    assert {p.stem for p in (out / "plots").glob("*.xlsx")} == names
    assert {p.stem for p in (out / "plots").glob("*.svg")} == names
