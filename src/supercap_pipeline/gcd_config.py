"""
gcd_config.py
-------------
Every threshold of the GCD evaluation (lab guide "Specific Capacitance from
GCD Data (PAT-Tester)") in one place. Section numbers refer to that guide.
Nothing here describes a particular measurement program: no cycle numbers,
step numbers, currents, voltages or masses.
"""

from __future__ import annotations

# ── 1. Input data (shared with CV, defined in export_io) ─────────────────────
from .export_io import (  # noqa: E402  (re-exported so the Meta sheet lists them)
    OUTPUT_SHEET, EXCEL_FIRST_DATA_ROW, MASS_FILE_PATTERN, MASS_MIN_G, MASS_MAX_G,
)

# ── 2. Segmentation ──────────────────────────────────────────────────────────
N_CYCLING = 100                    # a block with >= this many cycles is a cycling test
VOLTAGE_TOL_V = 0.005              # U_max / end-of-discharge voltage: within 5 mV of the block value
CURRENT_REL_TOL = 0.01             # |I| of consecutive cycles in one block: within 1 %
START_TOL_V = 0.050                # a complete cycle starts within 50 mV of U_min
U_LIMIT_GRID_V = 0.05              # U_min = block's median end-of-discharge voltage on this grid,
U_LIMIT_SNAP_SHIFT = 0.25          # rounding window shifted up by this fraction of the grid: the
                                   # tester stops at or below the limit (0 V: -37.5 ... +12.5 mV)
HOLD_V_TOL_V = 0.005               # voltage hold: V12 within +-5 mV of its median ...
HOLD_MIN_DURATION_S = 1800.0       # ... for at least 30 min, while |I| decays

# ── 3. Evaluation per cycle ──────────────────────────────────────────────────
MAH_TO_COULOMB = 3.6               # 1 mAh = 3.6 C  (and mWh * 3.6 = J)
T_U1_MAX_MS = 10.0                 # U1 read later than this after the switch -> "IR drop read late"
SLOPE_FIT_UPPER = 0.80             # optional info column C_slope: fit between 80 % ...
SLOPE_FIT_LOWER = 0.40             # ... and 40 % of (U1 - U_min) above U_min
SLOPE_FIT_MIN_POINTS = 10          # ... only with at least 10 points in that range

# ── 4. Quality checks ────────────────────────────────────────────────────────
STANDARD_J = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20]   # A/g (per m_mean)
STANDARD_J_REL_TOL = 0.02          # j within 2 % of a standard value, else flag
STABLE_CYCLE_TOL = 0.01            # |C_last - C_prev| / C_last > 1 % -> flag
ETA_C_WARN = 0.97                  # eta_C below -> warning
ETA_C_FLAG = 0.90                  # eta_C below -> flag "low efficiency"
C_VS_CE_TOL = 0.05                 # |C - C_E| / C > 5 % -> flag (curve shape)
IR_DROP_MAX_FRACTION = 0.20        # dU_IR / U_max > 20 % -> flag "resistance-limited"
ESR_SPREAD_TOL = 0.10              # (max ESR - min ESR) / min ESR in a rate series > 10 % -> flag
MIN_DISCHARGE_POINTS = 50          # fewer points in the discharge -> warning "logging too coarse"
CV_COMPARE_TOL = 0.15              # |C_el(GCD) - C_el(CV)| / C_el(CV) > 15 % -> flag

# ── 5. Long-term tests ───────────────────────────────────────────────────────
STABILISED_TOL = 0.01              # reference cycle: first with |C - C_next| / C < 1 %
EOL_RETENTION = 0.8                # end of life: retention below 80 % ...
EOL_ESR = 2.0                      # ... or ESR / ESR0 above 2
HOLD_LEAK_WINDOW_S = 600.0         # floating: mean leakage current over the last 10 min of a hold

# ── 6. Outputs (shared plot style) ───────────────────────────────────────────
from .plot_style import PLOT_DPI  # noqa: E402


def thresholds() -> dict[str, object]:
    """All constants above as {name: value}, for the Meta sheet."""
    return {k: v for k, v in globals().items() if k.isupper() and not callable(v)}
