"""
tests/test_gcd_units.py
-----------------------
Unit tests of the GCD building blocks (lab guide GCD 2-5): U_min crossing
and interpolation, IR drop / ESR with and without rest, U_min from the end
voltage, block detection with a repeated block, end of life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supercap_pipeline.gcd_analysis import end_of_life, find_crossing, interpolate_at, ir_drop
from supercap_pipeline.gcd_segmentation import cluster_values, detect_blocks, snap


# ── U_min crossing / interpolation (3.2) ─────────────────────────────────────

def test_crossing_between_points():
    V = np.array([2.0, 1.0, 0.10, -0.02])
    a, b = find_crossing(V, 0.0)
    assert (a, b) == (2, 3)
    q = interpolate_at(0.0, V[a], V[b], -0.6, -0.7)
    assert q == pytest.approx(-0.6 + (0 - 0.1) * (-0.1) / (-0.12))


def test_crossing_exactly_on_a_point():
    V = np.array([2.0, 1.0, 0.0, -0.01])
    a, b = find_crossing(V, 0.0)
    assert (a, b) == (1, 2)
    assert interpolate_at(0.0, V[a], V[b], -0.5, -0.7) == pytest.approx(-0.7)  # = X_b


def test_overshoot_below_u_min_is_ignored():
    # points after b (the tester logs a few more below the limit) do not matter
    V = np.array([1.0, 0.1, -0.000001, -0.0087, -0.014])
    assert find_crossing(V, 0.0) == (1, 2)


def test_no_crossing():
    assert find_crossing(np.array([2.0, 1.5, 1.3]), 1.25) is None
    assert find_crossing(np.array([1.2, 1.0]), 1.25) is None   # starts below U_min


def test_worked_example_interpolation():
    """Rows 199/200 of the reference: Q_dis = 0.704620 mAh, t_dis = 788.58 s."""
    V_a, V_b = 0.100698, -0.000001
    assert -interpolate_at(0.0, V_a, V_b, -0.671999, -0.704620) == pytest.approx(0.704620, abs=1e-6)
    assert interpolate_at(0.0, V_a, V_b, 752.07, 788.58) == pytest.approx(788.58, abs=0.01)
    assert interpolate_at(0.0, V_a, V_b, 1.414257, 1.412623) == pytest.approx(1.412623, abs=1e-6)


# ── IR drop / ESR (3.1) ──────────────────────────────────────────────────────

def test_esr_direct_switching_uses_two_times_current():
    du, esr = ir_drop(2.500003, 2.482009, 3.216614, -3.2167)
    assert du == pytest.approx(0.017994, abs=1e-6)
    assert esr == pytest.approx(0.017994 / (2 * 0.0032166), rel=1e-4)   # 2.797 Ohm


def test_esr_after_rest_uses_current_once():
    du, esr = ir_drop(2.49, 2.39, 0.0, -100.0)
    assert esr == pytest.approx(0.10 / 0.100)


# ── U_min, clustering, blocks (2) ────────────────────────────────────────────

def test_u_min_from_end_voltage():
    assert snap(-0.000004) == 0.0
    assert snap(-0.014) == 0.0
    assert snap(-0.035) == 0.0
    assert snap(1.238) == pytest.approx(1.25)
    assert snap(1.2495) == pytest.approx(1.25)


def test_cluster_values():
    assert list(cluster_values([0.1, 0.5, 0.1005, 0.499])) == [0, 1, 0, 1]


def _cycles(spec):
    """spec: list of (|I| mA, U_max, U_end, checkup, n)."""
    rows = []
    for I, umax, uend, cu, n in spec:
        rows += [dict(I_abs_mA=I, U_max=umax, U_end=uend, checkup=cu,
                      dV_last_ch=0.001, dV_last_dis=0.001)] * n
    c = pd.DataFrame(rows)
    c["cycle"] = np.arange(1, len(c) + 1)
    return c


def test_blocks_with_repeated_block():
    c = detect_blocks(_cycles([(10, 2.5, -0.001, 0, 3), (50, 2.5, -0.010, 0, 3),
                               (10.05, 2.5003, -0.004, 0, 3), (10, 2.5, -0.001, 0, 2)]))
    blocks = c.groupby("block").agg(n=("cycle", "size"), occ=("occurrence", "first"),
                                    I=("I_block_mA", "first"))
    # 10 mA appears three times; 10.05 is within 1 % of 10 -> same combination,
    # but consecutive with the following 10 mA block -> merged with it
    assert blocks["n"].tolist() == [3, 3, 5]
    assert blocks["occ"].tolist() == [1, 1, 2]
    assert (c["U_min"] == 0).all()


def test_coarse_logging_jitter_does_not_split_blocks():
    # 110 mA, 0.34 s half-cycles (long-term file A7029): the logged end points
    # overshoot the limits by 30-57 mV and jitter by ~25 mV between cycles
    rows = []
    rng = np.random.default_rng(1)
    for k in range(200):
        rows.append(dict(I_abs_mA=110.0, U_max=2.53 + 0.025 * rng.random(),
                         U_end=-0.029 - 0.023 * rng.random(), checkup=0,
                         dV_last_ch=0.0296, dV_last_dis=0.0058))
    c = pd.DataFrame(rows)
    c["cycle"] = np.arange(1, len(c) + 1)
    b = detect_blocks(c)
    assert b["block"].nunique() == 1
    assert (b["U_min"] == 0).all()


def test_blocks_split_by_u_min_and_holds():
    c = detect_blocks(_cycles([(10, 2.5, -0.01, 0, 2), (10, 2.5, 1.24, 0, 2),
                               (10, 2.5, 1.24, 1, 2)]))
    assert c.groupby("block").size().tolist() == [2, 2, 2]
    assert c.groupby("block")["U_min"].first().tolist() == pytest.approx([0.0, 1.25, 1.25])
    assert c.groupby("block")["occurrence"].first().tolist() == [1, 1, 2]


# ── End of life (5.1) ────────────────────────────────────────────────────────

def test_end_of_life_interpolated():
    x = np.arange(1, 11, dtype=float)
    ret = np.linspace(1.0, 0.7, 10)          # crosses 0.8 between x = 7 and 8
    eol = end_of_life(x, ret, np.ones(10), 0)
    assert eol["criterion"] == "retention < 80 %"
    assert eol["eol_x"] == pytest.approx(1 + (0.2 / 0.3) * 9)


def test_end_of_life_by_esr_and_not_reached():
    x = np.arange(5, dtype=float)
    eol = end_of_life(x, np.ones(5), np.array([1.0, 1.5, 1.9, 2.3, 2.5]), 0)
    assert eol["criterion"] == "ESR/ESR0 > 2" and eol["eol_x"] == pytest.approx(2.25)
    assert end_of_life(x, np.ones(5), np.ones(5), 0)["criterion"] == "not reached"
