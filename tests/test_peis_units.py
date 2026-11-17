"""
tests/test_peis_units.py
------------------------
Unit tests of the PEIS building blocks: circle fit of the arc, knee, Lin-KK,
labels, capacitance / tau0 rules, electrode area from Input-Variables.txt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supercap_pipeline.export_io import read_electrode_area
from supercap_pipeline.peis_analysis import capacitance_values, circle_fit_arc, find_knee
from supercap_pipeline.peis_kk import lin_kk, linkk_impedance_py_works
from supercap_pipeline.peis_spectra import label_spectra


def test_circle_fit_exact_semicircle():
    # R_s = 0.3, R_c = 2: semicircle centre (1.3, 0), points at f >= 1 kHz, then a tail
    th = np.linspace(0.15, np.pi - 0.15, 15)
    x, y = 1.3 - np.cos(th), np.sin(th)
    f = np.logspace(5, 3, 15)
    g = circle_fit_arc(np.r_[f, 100.0], np.r_[x, 2.4], np.r_[y, 0.5])
    assert g["arc"]
    assert g["R_s"] == pytest.approx(0.3, abs=1e-9) and g["R_c"] == pytest.approx(2.0, abs=1e-9)
    assert g["b"] == pytest.approx(0.0, abs=1e-9)
    assert g["f_apex"] == f[7]
    assert g["C_c"] == pytest.approx(1 / (2 * np.pi * f[7] * 2.0))


def test_no_arc_r_s_from_zero_crossing_or_upper_bound():
    f = np.logspace(5, 2, 6)
    g = circle_fit_arc(f, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60], [-0.2, -0.1, 0.1, 0.3, 0.6, 1.0])
    assert not g["arc"] and g["R_c"] == 0
    assert g["R_s"] == pytest.approx(0.25)                   # -Z'' crosses 0 between 0.2 and 0.3
    g2 = circle_fit_arc(f, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert g2["R_s"] == 0.2 and g2["R_s_upper_bound"]
    assert any(t.startswith("R_s upper bound") for _, t in g2["flags"])


def test_knee_first_local_minimum_below_apex():
    f = np.array([1e4, 3e3, 1e3, 300, 100, 30, 10, 3])
    y = np.array([0.5, 1.0, 0.8, 0.4, 0.3, 0.35, 0.3, 2.0])   # minima at 100 Hz and 10 Hz
    k = find_knee(f, np.arange(8.0), y, np.arange(10, 18), f_apex=3e3)
    assert (k["f_knee"], k["row_knee"]) == (100.0, 14)
    assert np.isnan(find_knee(f, np.arange(8.0), y, np.arange(8), f_apex=np.nan)["f_knee"])


def test_lin_kk_on_valid_and_invalid_data():
    from peis_synth import TRUE, model_z
    f = np.logspace(5, -2, 71)
    Z = model_z(f, TRUE)                       # the guide's model: KK-consistent by construction
    r = lin_kk(f, Z)
    assert r["converged"] and r["mu"] < 0.85 and r["M"] > 3
    assert r["max_res"] < 0.5
    # a non-causal distortion at low frequency shows up as residual
    Zbad = Z * (1 + 0.05 * (f < 0.1))
    assert lin_kk(f, Zbad)["max_res"] > 1.0


def test_impedance_py_linkk_status_is_reported():
    ok, note = linkk_impedance_py_works()
    assert isinstance(ok, bool) and "linKK" in note


def test_labels_slugs_styles():
    spec = pd.DataFrame({"spectrum": [1, 2, 3, 4, 5], "order": [1, 2, 3, 4, 5],
                         "V_dc": [0.0, 2.5, 0.0, -0.5, 0.0]})
    s = label_spectra(spec)
    assert s["label"].tolist() == ["0 V (1)", "2.5 V", "0 V (2)", "-0.5 V", "0 V (3)"]
    assert s["slug"].tolist() == ["0V_1", "2p5V", "0V_2", "m0p5V", "0V_3"]
    zero = s[s["V_dc"] == 0]
    assert len(set(map(tuple, zero["color"]))) == 1
    assert zero["marker"].tolist() == ["o", "s", "^"] and zero["filled"].tolist() == [True, False, True]


def test_tau0_rules():
    f = np.logspace(1, -2, 31)
    C_s = np.ones(31)
    C_re = np.ones(31)
    phi = np.full(31, -85.0)
    peak = np.exp(-((np.log10(f) + 1.0) ** 2) / 0.1)           # maximum at 0.1 Hz: 1 decade above f_min
    c = capacitance_values(f, C_s, C_re, peak, phi, 0.01)
    assert c["f0"] == pytest.approx(0.1) and not c["flags"]
    rising = 1 / f                                              # still rising at f_min
    c2 = capacitance_values(f, C_s, C_re, rising, phi, 0.01)
    assert np.isnan(c2["tau0"]) and c2["flags"][0][1].startswith("tau0 not determined")
    near = np.exp(-((np.log10(f) + 1.7) ** 2) / 0.01)          # f0 = 0.02 Hz: 0.3 decades
    c3 = capacitance_values(f, C_s, C_re, near, np.full(31, -70.0), 0.01)
    texts = [t for _, t in c3["flags"]]
    assert any(t.startswith("tau0 uncertain") for t in texts)
    assert any(t.startswith("capacitive line tilted") for t in texts)


@pytest.mark.parametrize("line, area", [
    ("d [mm]\t18", np.pi * 0.9 ** 2),
    ("diameter\t1.2 cm", np.pi * 0.6 ** 2),
    ("A\t2.0", 2.0),
    ("area [mm2]\t150", 1.5),
])
def test_electrode_area_from_input_variables(tmp_path, line, area):
    p = tmp_path / "Input-Variables.txt"
    key, val = line.split("\t")
    p.write_text(f"t\tKey\tValue\ns\t\t\n0\tm1\t0.03\n0\tm2\t0.03\n0\t{key}\t{val}\n", encoding="utf-8")
    a = read_electrode_area(p)
    assert a["area_cm2"] == pytest.approx(area)


def test_electrode_area_absent(tmp_path):
    p = tmp_path / "Input-Variables.txt"
    p.write_text("t\tKey\tValue\ns\t\t\n0\tm1\t0.03\n0\tmR\t1\n", encoding="utf-8")
    assert read_electrode_area(p) is None
