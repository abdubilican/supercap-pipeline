"""
cv_config.py
------------
Every tolerance and threshold of the CV evaluation (lab guide "Specific
Capacitance from CV Data (PAT-Tester)") in one place. Section numbers refer
to that guide. Nothing here describes a particular measurement program:
no cycle numbers, step numbers, scan rates, windows or masses.
"""

from __future__ import annotations

# ── 1. Input data ────────────────────────────────────────────────────────────
OUTPUT_SHEET = "Output"            # sheet holding the raw data in the xlsx export
EXCEL_FIRST_DATA_ROW = 3           # row 1 = names, row 2 = units -> data from row 3
MASS_FILE_PATTERN = "*Input-Variables*.txt"
MASS_MIN_G = 1e-4                  # plausibility range of one electrode mass [g];
MASS_MAX_G = 1.0                   # outside it the unit is probably not g -> error

# ── 2. Segmentation ──────────────────────────────────────────────────────────
TRANSITION_MIN_FRACTION = 0.50     # drop sweep if |dV| < 50 % of the neighbours' window
MIN_SWEEP_POINTS = 5               # ... or if it has fewer points than this
SCAN_RATE_FIT_FRACTION = 0.80      # linear fit V12(t_step) over the middle 80 %
SCAN_RATE_SIG_FIGS = 2             # nominal scan rate = measured, 2 significant figures
WINDOW_REL_TOL = 0.02              # vertex tolerance: 2 % of V_max ...
WINDOW_ABS_TOL_V = 0.020           # ... or 20 mV, whichever is larger
U_MAX_ROUND_V = 0.01               # U_max rounded to 0.01 V for grouping
DURATION_REL_TOL = 0.05            # sweep lasts 2*U/nu (sym.) or U/nu (one-sided) +-5 %

# ── 3. Charges ───────────────────────────────────────────────────────────────
QCD_START_REL_TOL = 0.01           # warn if |Qcd[first row]| > 1 % of the sweep's end value

# ── 4. Capacitance ───────────────────────────────────────────────────────────
MAH_TO_COULOMB = 3.6               # 1 mAh = 3.6 C
MASS_ASYMMETRY_TOL = 0.10          # |m1-m2|/m_mean > 10 % -> also report C_el,exact

# ── 5. Quality checks ────────────────────────────────────────────────────────
MIN_DISCHARGE_FRACTION = 0.05      # Q_dis < 5 % of Q_ch -> not evaluable
POLARITY_DEV_TOL = 0.05            # |C+ - C-| / C > 5 % -> flag
EFFICIENCY_MIN = 0.95              # eta below this -> flag (warning only)
STABILITY_TOL = 0.02               # |C_last - C_prev| / C_last > 2 % -> flag
SCAN_RATE_TREND_TOL = 0.02         # C rising with nu by > 2 % -> flag the faster block
MAGNITUDE_RATIO_MIN = 0.3          # C / C_est outside [0.3, 3] -> "possible unit error"
MAGNITUDE_RATIO_MAX = 3.0

# ── 6. Outputs ───────────────────────────────────────────────────────────────
PLOT_DPI = 300


def tolerances() -> dict[str, float | int | str]:
    """All constants above as {name: value}, for the Meta sheet."""
    return {k: v for k, v in globals().items() if k.isupper()}
