"""
tests/test_capacitance.py
--------------------------
Unit tests for compute_specific_capacitance() using synthetic ideal CVs.

For a perfectly rectangular CV:
  I_anodic  = +C_cell * v   [mA]
  I_cathodic = -C_cell * v  [mA]
  => C_sp_expected = C_cell / m_lightest  [F/g]  (independent of v and ΔV)

Sweep pattern tested: 0 → +V_max → -V_max → 0 (matches real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supercap_pipeline.analysis import compute_specific_capacitance


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic CV builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_ideal_cv(
    v_mVs: float,
    window_V: float,
    C_F: float,
    cycle: int = 1,
    n_pts: int = 4000,
) -> pd.DataFrame:
    """
    Build a synthetic ideal rectangular CV DataFrame for one cycle.

    Sweep: 0 -> +window_V -> -window_V -> 0  (same as the real measurements).
    I_anodic  = +C_F * v_Vs * 1e3  mA
    I_cathodic = -C_F * v_Vs * 1e3  mA

    Parameters
    ----------
    v_mVs    : scan rate [mV/s]
    window_V : half-window voltage [V]
    C_F      : cell capacitance [F]
    cycle    : cycle number label
    n_pts    : number of time-series points (per segment; total ~4×)
    """
    v_Vs = v_mVs * 1e-3            # V/s
    I0   = C_F * v_Vs * 1e3        # mA (ideal rectangular amplitude)
    t_seg = window_V / v_Vs         # s for 0 → V_max

    n1 = n_pts                      # 0 → V_max
    n2 = 2 * n_pts                  # V_max → -V_max
    n3 = n_pts                      # -V_max → 0

    t1 = np.linspace(0,           t_seg,     n1)
    t2 = np.linspace(t_seg,       3 * t_seg, n2)
    t3 = np.linspace(3 * t_seg,   4 * t_seg, n3)

    V1 = v_Vs * t1                                    # 0 → V_max
    V2 = window_V - v_Vs * (t2 - t_seg)              # V_max → -V_max
    V3 = -window_V + v_Vs * (t3 - 3 * t_seg)         # -V_max → 0

    I1 = np.full(n1, +I0)
    I2 = np.full(n2, -I0)
    I3 = np.full(n3, +I0)

    # Concatenate, dropping the duplicate boundary points
    t_all = np.concatenate([t1, t2[1:], t3[1:]])
    V_all = np.concatenate([V1, V2[1:], V3[1:]])
    I_all = np.concatenate([I1, I2[1:], I3[1:]])

    return pd.DataFrame({
        "t":                t_all,
        "V12":              V_all,
        "I12":              I_all,
        "I12_spec_mAg":     I_all,  # placeholder; not used by analysis
        "cycle":            cycle,
        "type":             "VS",
        "scan_speed_mVs":   float(v_mVs),
        "voltage_window_V": float(window_V),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

_TOL = 0.002  # 0.2 % tolerance (numerical integration error from finite n_pts)


def test_basic_rectangular_cv():
    """C_sp == C_cell / m_lightest for an ideal rectangular CV."""
    C_F  = 1e-3   # 1 mF
    m_g  = 1e-3   # 1 mg  => expected C_sp = 1.0 F/g
    meta = {"m_lightest": m_g}

    df      = _make_ideal_cv(v_mVs=1.0, window_V=1.0, C_F=C_F)
    cap_df  = compute_specific_capacitance(df, meta)

    assert len(cap_df) == 1, f"Expected 1 row, got {len(cap_df)}"
    got = cap_df["C_sp_Fg"].iloc[0]
    expected = C_F / m_g  # 1.0 F/g
    err = abs(got - expected) / expected
    assert err < _TOL, f"C_sp={got:.5f} F/g, expected={expected:.5f} F/g (err={err*100:.3f}%)"


@pytest.mark.parametrize("v_mVs", [1.0, 10.0, 100.0])
def test_scan_rate_independence(v_mVs):
    """C_sp must be independent of scan rate for an ideal capacitor."""
    C_F  = 1e-3
    m_g  = 1.38e-3  # realistic electrode mass
    meta = {"m_lightest": m_g}

    df      = _make_ideal_cv(v_mVs=v_mVs, window_V=1.0, C_F=C_F)
    cap_df  = compute_specific_capacitance(df, meta)

    got      = cap_df["C_sp_Fg"].iloc[0]
    expected = C_F / m_g
    err = abs(got - expected) / expected
    assert err < _TOL, (
        f"v={v_mVs} mV/s: C_sp={got:.5f} F/g, expected={expected:.5f} F/g "
        f"(err={err*100:.3f}%)"
    )


@pytest.mark.parametrize("window_V", [1.0, 1.5, 2.0, 2.5])
def test_window_independence(window_V):
    """C_sp must be independent of voltage window for an ideal capacitor."""
    C_F  = 1e-3
    m_g  = 1.38e-3
    meta = {"m_lightest": m_g}

    df      = _make_ideal_cv(v_mVs=10.0, window_V=window_V, C_F=C_F)
    cap_df  = compute_specific_capacitance(df, meta)

    got      = cap_df["C_sp_Fg"].iloc[0]
    expected = C_F / m_g
    err = abs(got - expected) / expected
    assert err < _TOL, (
        f"window=+/-{window_V} V: C_sp={got:.5f} F/g, expected={expected:.5f} F/g "
        f"(err={err*100:.3f}%)"
    )


def test_units_mA_to_F_g():
    """
    Explicit unit trace for a hand-calculable case:
      C_cell = 0.01 F, m = 0.01 g, v = 0.01 V/s (10 mV/s), window = 2.0 V
      I0 = C * v = 0.01 * 0.01 = 1e-4 A = 0.1 mA
      A  = 4 * I0[mA] * window_V = 4 * 0.1 * 2.0 = 0.8 mA*V
      C_sp = 0.8e-3 / (2 * 0.01 * 4.0 * 0.01) = 0.0008 / 0.0008 = 1.0 F/g
    """
    C_F   = 0.01
    m_g   = 0.01
    v_mVs = 10.0
    w_V   = 2.0
    meta  = {"m_lightest": m_g}

    df     = _make_ideal_cv(v_mVs=v_mVs, window_V=w_V, C_F=C_F)
    cap_df = compute_specific_capacitance(df, meta)

    got      = cap_df["C_sp_Fg"].iloc[0]
    expected = C_F / m_g  # = 1.0 F/g
    err = abs(got - expected) / expected
    assert err < _TOL, f"C_sp={got:.5f} F/g, expected={expected:.5f} F/g (err={err*100:.3f}%)"


def test_multiple_conditions_in_one_dataframe():
    """Multiple (speed, window) conditions in one df all yield correct C_sp."""
    C_F  = 5e-4
    m_g  = 1.38e-3
    meta = {"m_lightest": m_g}
    expected = C_F / m_g

    dfs = []
    cyc = 1
    for v_mVs in [1.0, 10.0, 100.0]:
        for w_V in [1.0, 2.0]:
            dfs.append(_make_ideal_cv(v_mVs=v_mVs, window_V=w_V, C_F=C_F, cycle=cyc))
            cyc += 1

    combined = pd.concat(dfs, ignore_index=True)
    cap_df   = compute_specific_capacitance(combined, meta)

    assert len(cap_df) == 6, f"Expected 6 conditions, got {len(cap_df)}"
    for _, row in cap_df.iterrows():
        err = abs(row["C_sp_Fg"] - expected) / expected
        assert err < _TOL, (
            f"v={row['scan_speed_mVs']} mV/s, w={row['voltage_window_V']} V: "
            f"C_sp={row['C_sp_Fg']:.5f} F/g, expected={expected:.5f} F/g "
            f"(err={err*100:.3f}%)"
        )
