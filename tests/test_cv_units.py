"""
tests/test_cv_units.py
----------------------
Unit tests: 0 V interpolation (lab guide 3.1), the section-9 worked example
(charges, capacitance, specific values), quality checks and the mass file.
"""

from __future__ import annotations

import numpy as np
import pytest

from cv_synth import write_masses
from supercap_pipeline.cv_analysis import (
    capacitance, charges_symmetric, evaluability, interpolate_zero_crossing,
)
from supercap_pipeline.cv_io import read_masses
from supercap_pipeline.cv_segmentation import classify_window, round_sig


# ── interpolation ────────────────────────────────────────────────────────────

def test_crossing_between_points():
    x = interpolate_zero_crossing([-0.3, -0.1, 0.1, 0.3], [0.0, 1.0, 3.0, 5.0])
    assert (x["a"], x["b"], x["n_crossings"]) == (1, 2, 1)
    assert x["Q0"] == pytest.approx(2.0)


def test_crossing_exactly_at_data_point():
    x = interpolate_zero_crossing([-0.2, -0.1, 0.0, 0.1], [0.0, 1.0, 2.0, 3.0])
    assert (x["a"], x["b"]) == (2, 2)
    assert x["Q0"] == 2.0
    assert x["n_crossings"] == 1  # the exact zero is not counted twice


def test_several_crossings_use_first():
    x = interpolate_zero_crossing([-0.1, 0.1, -0.1, 0.1], [0.0, 2.0, 10.0, 20.0])
    assert x["n_crossings"] == 3
    assert (x["a"], x["b"]) == (0, 1)
    assert x["Q0"] == pytest.approx(1.0)


def test_no_crossing():
    assert interpolate_zero_crossing([-0.3, -0.2, -0.1], [0.0, 1.0, 2.0]) is None


def test_worked_example_section_9():
    """+-1.0 V, 1 mV/s, Excel rows 315/316 and 349/350 of BIA-VR-009-01-05."""
    up = interpolate_zero_crossing([-0.12991, 0.01225], [0.17511, 0.21383])
    down = interpolate_zero_crossing([0.08220, -0.11704], [-0.19479, -0.25379])
    assert up["Q0"] == pytest.approx(0.21049, abs=5e-6)
    assert down["Q0"] == pytest.approx(-0.21913, abs=5e-6)

    ch = charges_symmetric(dict(Q0=up["Q0"], Q_end=0.57687, I0=np.nan),
                           dict(Q0=down["Q0"], Q_end=-0.58476, I0=np.nan))
    assert ch["+"]["Q_dis"] == pytest.approx(0.2191, abs=5e-5)
    assert ch["+"]["Q_ch"] == pytest.approx(0.3664, abs=5e-5)
    assert ch["-"]["Q_dis"] == pytest.approx(0.2105, abs=5e-5)
    assert ch["-"]["Q_ch"] == pytest.approx(0.3656, abs=5e-5)

    m = 0.032164
    cp = capacitance(ch["+"]["Q_dis"], ch["+"]["Q_ch"], 1.0, m, m)
    cm = capacitance(ch["-"]["Q_dis"], ch["-"]["Q_ch"], 1.0, m, m)
    assert cp["C"] == pytest.approx(0.789, abs=5e-4)
    assert cm["C"] == pytest.approx(0.758, abs=5e-4)
    C = (cp["C"] + cm["C"]) / 2
    assert C == pytest.approx(0.773, abs=5e-4)
    assert C / (2 * m) == pytest.approx(12.0, abs=0.05)       # C_cell,sp
    assert 2 * C / m == pytest.approx(48.1, abs=0.05)         # C_el
    assert cp["eta"] == pytest.approx(0.60, abs=0.005)
    assert cm["eta"] == pytest.approx(0.58, abs=0.005)
    assert np.isnan(cp["C_el_exact"])  # equal masses


def test_negative_discharge_is_not_evaluable_and_keeps_sign():
    assert evaluability(-0.0146, 0.2) == "not evaluable (Q_dis <= 0)"
    assert evaluability(0.0007, 0.2).startswith("not evaluable (Q_dis < 5")
    assert evaluability(0.2, 0.3) is None
    cap = capacitance(-0.0146, 0.2, 1.0, 0.03, 0.03, evaluable=False)
    assert np.isnan(cap["C"]) and np.isnan(cap["C_el"])
    assert cap["eta"] == pytest.approx(-0.0146 / 0.2)


def test_classify_window_and_rounding():
    assert classify_window(-1.499, 1.501) == ("symmetric", 1.5)
    assert classify_window(0.004, 2.5) == ("one-sided", 2.5)
    assert classify_window(0.5, 2.5)[0] == "other"
    assert round_sig(0.998) == 1.0 and round_sig(49.7) == 50 and round_sig(10.04) == 10


# ── masses ───────────────────────────────────────────────────────────────────

def test_read_masses_el_cell_format(tmp_path):
    m = read_masses(write_masses(tmp_path))
    assert m["m1_g"] == pytest.approx(0.00138)
    assert m["m2_g"] == pytest.approx(0.00138)
    assert m["m_mean_g"] == pytest.approx(0.00138)


def test_read_masses_mg_unit(tmp_path):
    p = tmp_path / "Input-Variables.txt"
    p.write_text("m1 [mg] = 1.5\nm2 = 1.7 mg\n", encoding="ascii")
    m = read_masses(p)
    assert m["m1_g"] == pytest.approx(0.0015)
    assert m["m2_g"] == pytest.approx(0.0017)


def test_read_masses_missing_mass(tmp_path):
    p = tmp_path / "Input-Variables.txt"
    p.write_text("t\tKey\tValue\ns\t\t\n0\tm1\t0.002\n", encoding="ascii")
    with pytest.raises(ValueError, match="m2"):
        read_masses(p)


def test_read_masses_implausible_unit(tmp_path):
    p = tmp_path / "Input-Variables.txt"
    p.write_text("t\tKey\tValue\ns\t\t\n0\tm1\t1.38\n0\tm2\t1.38\n", encoding="ascii")
    with pytest.raises(ValueError, match="check the unit"):
        read_masses(p)
