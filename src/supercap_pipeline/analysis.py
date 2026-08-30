"""
analysis.py
-----------
Compute specific capacitance from CV curves.

Formula (area-integral, standard literature definition):

    C_sp [F/g] = A × 1e-3
                 ─────────────────────────────────
                 2 · v [V/s] · ΔV_total [V] · m [g]

where
  A = ∮|I|·|dV|   unsigned closed-loop area integral  [mA·V]
                  implemented as  sum(|I_avg[i]| · |ΔV[i]|)  over all segments
  v               scan rate in V/s  (= scan_speed_mVs / 1000)
  ΔV_total        total voltage window = 2 × voltage_window_V
  m               mass of the lightest electrode in grams
  1e-3            unit conversion mA·V → A·V = C·V = J  →  /V = C  →  F

Note: using |I|·|dV| (unsigned magnitudes) is equivalent to |∮I·dV| for our
sweep pattern (0 → +Vmax → −Vmax → 0) because every segment contributes
positively to the signed integral. The unsigned form is stated here explicitly
to match the literature convention and remain correct for any sweep shape.

The first cycle at each (scan_speed, voltage_window) condition is excluded
from the mean/std calculation (it starts from a different voltage and has an
asymmetric CV shape), but it IS included in the per-cycle table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_specific_capacitance(
    df: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    """
    Calculate specific capacitance C_sp for every usable cycle.

    Parameters
    ----------
    df   : time-series DataFrame from loader.load_experiment()
    meta : metadata dict from loader.load_experiment()

    Returns
    -------
    DataFrame with columns:
        cycle, scan_speed_mVs, voltage_window_V, C_sp_Fg
    Sorted by (voltage_window_V, scan_speed_mVs, cycle).
    """
    m     = meta["m_lightest"]   # g
    rows  = []
    vs_df = df[(df["type"] == "VS") & (df["cycle"] > 0)].copy()

    groups = vs_df.groupby(
        ["cycle", "scan_speed_mVs", "voltage_window_V"], sort=True
    )

    for (cyc, speed, window), grp in groups:
        if np.isnan(speed) or np.isnan(window) or len(grp) < 10:
            continue

        grp  = grp.sort_values("t")
        I    = grp["I12"].to_numpy(dtype=float)   # mA
        V    = grp["V12"].to_numpy(dtype=float)   # V

        # ∮|I|·|dV| : sum of unsigned-current × unsigned-voltage-step
        dV_seg    = np.diff(V)
        I_abs_mid = (np.abs(I[:-1]) + np.abs(I[1:])) / 2.0
        area = float(np.sum(I_abs_mid * np.abs(dV_seg)))   # [mA·V]
        if area == 0.0:
            continue

        v_Vs    = speed * 1e-3                     # V/s
        delta_V = 2.0 * window                     # total window [V]

        C_sp = area * 1e-3 / (2.0 * v_Vs * delta_V * m)   # F/g

        rows.append({
            "cycle":             int(cyc),
            "scan_speed_mVs":   float(speed),
            "voltage_window_V": float(window),
            "C_sp_Fg":          float(C_sp),
        })
        log.debug(
            "cyc=%d  speed=%g mV/s  window=+/-%g V  area=%.4f mA*V  C_sp=%.2f F/g",
            cyc, speed, window, area, C_sp,
        )

    cap_df = pd.DataFrame(rows).sort_values(
        ["voltage_window_V", "scan_speed_mVs", "cycle"]
    ).reset_index(drop=True)

    return cap_df


def compute_mean_capacitance(
    cap_df: pd.DataFrame,
    exclude_first_cycle: bool = True,
) -> pd.DataFrame:
    """
    Aggregate per-cycle C_sp into mean ± std per (scan_speed, voltage_window).

    Parameters
    ----------
    cap_df              : output of compute_specific_capacitance()
    exclude_first_cycle : if True, skip the lowest cycle number in each
                          (scan_speed, voltage_window) group (formation cycle)

    Returns
    -------
    DataFrame with columns:
        scan_speed_mVs, voltage_window_V,
        C_sp_mean_Fg, C_sp_std_Fg, n_cycles
    Sorted by (voltage_window_V, scan_speed_mVs).
    """
    df = cap_df.copy()

    if exclude_first_cycle:
        # For each (speed, window) group, find the minimum cycle number and exclude it
        first_cycle_idx = (
            df.groupby(["scan_speed_mVs", "voltage_window_V"])["cycle"]
              .transform("min")
        )
        df = df[df["cycle"] > first_cycle_idx]

    agg = (
        df.groupby(["scan_speed_mVs", "voltage_window_V"])["C_sp_Fg"]
          .agg(
              C_sp_mean_Fg = "mean",
              C_sp_std_Fg  = "std",
              n_cycles     = "count",
          )
          .reset_index()
          .sort_values(["voltage_window_V", "scan_speed_mVs"])
          .reset_index(drop=True)
    )

    # Replace NaN std (single-cycle groups) with 0
    agg["C_sp_std_Fg"] = agg["C_sp_std_Fg"].fillna(0.0)

    return agg
