"""
peis_config.py
--------------
Every threshold and setting of the PEIS evaluation (lab guide "Impedance
Spectroscopy (PEIS) of Supercapacitors (PAT-Tester)") in one place.
Section numbers refer to that guide. Nothing here describes a particular
measurement: no steps, voltages, frequencies or row numbers.
"""

from __future__ import annotations

# ── 1. Input data (shared with CV/GCD, defined in export_io) ─────────────────
from .export_io import (  # noqa: E402  (re-exported so the Meta sheet lists them)
    IMPEDANCE_FILE, MASS_FILE_PATTERN, MASS_MIN_G, MASS_MAX_G,
)

ELECTRODE_DIAMETER_MM = 18.0       # default electrode diameter if Input-Variables.txt has none

# ── 2. Spectra ───────────────────────────────────────────────────────────────
DC_VOLTAGE_ROUND_V = 0.01          # DC voltage of a spectrum = median V12 rounded to this

# ── 3. Data quality (guide section 7) ────────────────────────────────────────
AMP_TOL = 0.10                     # |dV12 - A_set| / A_set > 10 % -> point excluded
SFDR_MIN = 30.0                    # SFDR(Z12) below this [dB] -> point excluded
FIT_INCLUDE_L = False              # True: keep inductive HF points and fit with L0- in front
REVERSIBILITY_TOL = 0.05           # repeated DC voltage: relative change > 5 % -> flag
CS_CHECK_TOL = 0.001               # C_s vs instrument Cs: > 0.1 % -> flag
KK_MU = 0.85                       # Lin-KK: increase M until mu < 0.85 (Schönleber)
KK_M_START = 3                     # first number of Voigt elements tried
KK_WARN = 1.0                      # KK residual per point > 1 % -> warning
KK_FLAG = 2.0                      # KK residual per point > 2 % -> flag

# ── 4. Graphical evaluation (guide sections 8, 9) ────────────────────────────
ARC_FMIN = 1000.0                  # high-frequency arc: points with f >= 1 kHz
F_ESR = 1000.0                     # ESR: Z' at the measured point nearest 1 kHz
F_CS_REF = 0.1                     # C_s also reported at the point nearest 0.1 Hz
TAU0_MIN_DECADES = 0.5             # log10(f0 / f_min) below this -> "tau0 uncertain"
PHI_CAPACITIVE_DEG = -80.0         # phi(f_min) above this -> info "capacitive line tilted"

# ── 5. Equivalent-circuit fit (guide section 10) ─────────────────────────────
CIRCUIT_ARC = "R0-p(R1,CPE1)-TLMQ0"
CIRCUIT_NO_ARC = "R0-TLMQ0"
CIRCUIT_L_PREFIX = "L0-"
FIT_BOUNDS = {                     # parameter: (lower, upper)
    "R_s": (0.0, 5.0),
    "R_c": (0.0, 20.0),
    "Q_c": (1e-9, 1e-3),
    "alpha_c": (0.5, 1.0),
    "R_ion": (0.0, 50.0),
    "Q_dl": (0.01, 10.0),
    "alpha_dl": (0.5, 1.0),
    "L": (0.0, 1e-5),
}
FIT_START_ALPHA_C = 0.85
FIT_START_R_ION = 1.0
FIT_START_ALPHA_DL = 0.9
FIT_REL_UNCERTAINTY_MAX = 0.10     # relative parameter uncertainty > 10 % -> flag
FIT_BOUND_PROXIMITY = 0.01         # parameter within 1 % of a bound -> flag
FIT_MAX_RESIDUAL = 5.0             # max fit residual > 5 % -> flag
FIT_RS_AGREE_OHM = 0.1             # |R_s(fit) - R_s(circle)| > 0.1 Ohm -> flag
FIT_RC_AGREE_REL = 0.10            # |R_c(fit) - R_c(circle)| / R_c > 10 % -> flag
FIT_CURVE_POINTS_PER_DECADE = 20   # density of the plotted fit curve

# ── 6. Comparison with CV and GCD (guide section 11) ─────────────────────────
COMPARE_TOL = 0.15                 # relative deviation / range tolerance
COMPARE_U_MAX_TOL_V = 0.1          # spectrum DC voltage within 0.1 V of the GCD U_max

# ── 7. Plots ─────────────────────────────────────────────────────────────────
NYQUIST_MARK_FREQS = (1e4, 1e3, 1e2, 1.0, 0.1, 0.01)   # Hz, labelled where measured
NYQUIST_MARK_TOL_DECADES = 0.1     # "measured" = a point within 0.1 decade of the target
NYQUIST_HF_ZOOM = 1.2              # HF zoom: up to 1.2 x the largest Z'_knee (or Z'(1 kHz))
from .plot_style import PLOT_DPI  # noqa: E402


def settings() -> dict[str, object]:
    """All constants above as {name: value}, for the Meta sheet."""
    return {k: v for k, v in globals().items() if k.isupper() and not callable(v)}
