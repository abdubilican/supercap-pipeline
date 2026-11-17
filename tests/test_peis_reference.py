"""
tests/test_peis_reference.py
----------------------------
Regression against the expected values of lab guide PEIS section 10 for the
impedance export of BIA-VR-009-01-05 (Output-Impedance.txt, 4 spectra),
with m1 = m2 = 0.032164 g.

The raw file is not in the repository. Point SUPERCAP_REF_PEIS at it, or put
it at tests/fixtures/BIA-VR-009-01-05/peis/Output-Impedance.txt; otherwise
the tests are skipped. The comparison test additionally needs SUPERCAP_REF_GCD.

Tolerances: graphical values 0.5 % (frequencies exact), fit 1 %, Lin-KK
residuals +-0.2 percentage points and M +-2.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from supercap_pipeline.peis_spectra import read_impedance_export
from supercap_pipeline.pipeline import evaluate_peis, run_gcd_export, run_peis_export

REF = Path(os.environ.get(
    "SUPERCAP_REF_PEIS",
    Path(__file__).parent / "fixtures" / "BIA-VR-009-01-05" / "peis" / "Output-Impedance.txt"))
REF_GCD = Path(os.environ.get("SUPERCAP_REF_GCD", "__missing__"))
pytestmark = pytest.mark.skipif(not REF.exists(), reason=f"reference file not found: {REF}")

M = 0.032164
MASSES = dict(m1_g=M, m2_g=M, m_mean_g=M, source="section 10")
AREA = dict(area_cm2=np.pi * 0.9 ** 2, diameter_mm=18.0, source="default")
LABELS = ["0 V (1)", "0.5 V", "2.5 V", "0 V (2)"]

DETECTED = {  # label: (order, step, rows, V_dc, excluded rows)
    "0 V (1)": (1, 50, (3, 73), 0.0, [3, 4]),
    "0.5 V": (2, 100, (75, 145), 0.5, [75, 76]),
    "2.5 V": (3, 150, (147, 217), 2.5, [147, 148]),
    "0 V (2)": (4, 190, (219, 289), 0.0, [219]),
}
GRAPHICAL = {  # column: values in LABELS order
    "R_s_circle": [0.234, 0.254, 0.254, 0.261],
    "R_c_circle": [4.998, 4.819, 2.571, 5.088],
    "Zre_knee": [5.501, 5.315, 2.905, 5.622],
    "Zre_1k": [5.060, 4.924, 2.803, 5.174],
    "C_s_01": [0.6795, 0.6974, 0.9367, 0.6642],
    "C_s_fmin": [0.8527, 0.8754, 1.0874, 0.8346],
    "C_re_fmin": [0.7307, 0.7540, 0.9848, 0.7141],
    "C_el": [53.02, 54.44, 67.62, 51.90],
    "tau0": [6.415, 6.415, 4.068, 6.415],
    "phi_fmin": [-67.77, -68.13, -72.11, -67.67],
    "I_last_mA": [-0.0439, -0.0092, 0.5823, -0.0499],
}
FREQS = {  # exact frequencies
    "f_apex": [12451.171, 12451.171, 31494.14, 12451.171],
    "f_knee": [31.844429, 31.844429, 78.125, 31.844429],
    "f0": [0.024811, 0.024811, 0.039125, 0.024811],
}
KNEE_ROWS = [38, 110, 178, 254]
KK = [(25, 0.28), (26, 0.79), (18, 1.79), (25, 0.33)]
FIT = {
    "fit_R_s": [0.2314, 0.2494, 0.2558, 0.2716],
    "fit_R_c": [5.115, 4.920, 2.584, 5.187],
    "fit_Q_c": [1.533e-5, 1.417e-5, 1.151e-5, 1.421e-5],
    "fit_alpha_c": [0.8424, 0.8498, 0.8635, 0.8500],
    "fit_R_ion": [1.920, 1.902, 1.737, 1.908],
    "fit_Q_dl": [0.6830, 0.7049, 0.9359, 0.6652],
    "fit_alpha_dl": [0.9216, 0.9240, 0.9399, 0.9198],
    "EDR": [5.986, 5.803, 3.419, 6.095],
    "fit_max_res_pct": [3.98, 3.46, 1.94, 3.82],
}
REL_UNC = {"fit_R_s_rel_unc": [0.08, 0.07, 0.06, 0.05], "fit_R_ion_rel_unc": [0.05, 0.04, 0.02, 0.05]}
C_EFF_UF = [2.62, 2.61, 2.22, 2.65]


def _close(got, exp, rel, last_digit=0.0):
    return abs(got - exp) <= max(rel * abs(exp), 0.5 * last_digit + 1e-12)


@pytest.fixture(scope="module")
def result():
    return evaluate_peis(read_impedance_export(REF), MASSES, AREA)


@pytest.fixture(scope="module")
def spectra(result):
    return result["spectra"].set_index("label").loc[LABELS]


def test_detected_spectra(result, spectra):
    assert spectra.index.tolist() == LABELS
    assert (spectra["n_points"] == 71).all()
    assert spectra["f_max"].to_numpy() == pytest.approx(99121.093)
    assert spectra["f_min"].to_numpy() == pytest.approx(0.009973)
    assert spectra["A_set_mV"].to_numpy() == pytest.approx(10.0, abs=0.01)
    assert (spectra["n_inductive"] == 0).all()
    pts = result["points"]
    for lab, (order, step, rows, v, excl) in DETECTED.items():
        s = spectra.loc[lab]
        assert (s["order"], s["step"], (s["row_start"], s["row_end"]), s["V_dc"]) == (order, step, rows, v)
        got = pts[(pts["label"] == lab) & pts["excluded"]]["excel_row"].tolist()
        assert got == excl, lab
    reasons = pts.set_index("excel_row")["exclude_reason"]
    assert reasons[3].startswith("amplitude 3.2 mV") and "SFDR 14.9 dB" in reasons[3]
    assert reasons[4] == "SFDR 28.6 dB < 30 dB"


@pytest.mark.parametrize("col", sorted(GRAPHICAL))
def test_graphical_values(spectra, col):
    for lab, exp in zip(LABELS, GRAPHICAL[col]):
        digits = {"I_last_mA": 1e-4, "C_el": 0.01, "phi_fmin": 0.01, "tau0": 0.001}.get(col, 1e-3)
        assert _close(spectra.loc[lab, col], exp, 0.005, digits), (col, lab, spectra.loc[lab, col])


def test_frequencies_exact(spectra):
    for col, vals in FREQS.items():
        assert spectra[col].to_numpy() == pytest.approx(vals, rel=1e-6), col
    assert spectra["row_knee"].tolist() == KNEE_ROWS
    assert spectra["f_1k"].to_numpy() == pytest.approx(1007.08, rel=1e-6)
    assert spectra["f_Cs_01"].to_numpy() == pytest.approx(0.101725, rel=1e-5)


def test_lin_kk(spectra):
    for lab, (m, mx) in zip(LABELS, KK):
        assert abs(spectra.loc[lab, "kk_M"] - m) <= 2, lab
        assert abs(spectra.loc[lab, "kk_max_res_pct"] - mx) <= 0.2, lab
    assert spectra.loc["2.5 V", "kk_f_max_res"] == pytest.approx(0.009973)


@pytest.mark.parametrize("col", sorted(FIT))
def test_fit(spectra, col):
    for lab, exp in zip(LABELS, FIT[col]):
        assert _close(spectra.loc[lab, col], exp, 0.01), (col, lab, spectra.loc[lab, col])


def test_fit_uncertainties_and_derived(spectra):
    for col, vals in REL_UNC.items():
        for lab, exp in zip(LABELS, vals):
            assert abs(spectra.loc[lab, col] - exp) <= 0.006, (col, lab)
    others = [c for c in spectra.columns if c.endswith("_rel_unc") and c not in REL_UNC]
    assert (spectra[others] <= 0.06).all().all()
    assert (spectra["C_c_eff"] * 1e6).to_numpy() == pytest.approx(C_EFF_UF, abs=0.01)


def test_reversibility(result):
    rev = result["reversibility"].set_index("quantity")
    assert rev.loc["C_s(f_min)", "change_pct"] == pytest.approx(-2.1, abs=0.05)
    assert rev.loc["Z'(1 kHz)", "change_pct"] == pytest.approx(2.2, abs=0.05)
    assert not rev["flag"].any()


def test_expected_flags(result):
    f = result["flags"]
    flags = f[f["severity"] == "flag"]
    tau = flags[flags["flag"].str.startswith("tau0 uncertain")]["spectrum"].tolist()
    assert sorted(tau) == sorted(["0 V (1)", "0.5 V", "0 V (2)"])
    assert len(flags) == 3                                    # nothing else, in particular no fit flag
    tilted = f[f["flag"].str.startswith("capacitive line tilted")]["spectrum"].tolist()
    assert sorted(tilted) == sorted(LABELS)
    kk = f[f["flag"].str.startswith("KK residual")]
    assert len(kk) == 1 and kk.iloc[0]["severity"] == "warning" and kk.iloc[0]["spectrum"] == "2.5 V"
    assert kk.iloc[0]["f_Hz"] == pytest.approx(0.009973)
    excluded = f[f["flag"].str.startswith("point excluded")]["row"].tolist()
    assert sorted(excluded) == [3, 4, 75, 76, 147, 148, 219]


@pytest.mark.skipif(not REF_GCD.exists(), reason="GCD reference (SUPERCAP_REF_GCD) not available")
def test_comparison_with_gcd(tmp_path):
    for src, name in ((REF_GCD.parent, "02_GCD"), (REF.parent, "03_PEIS")):
        shutil.copytree(src, tmp_path / "cell" / name,
                        ignore=shutil.ignore_patterns("Output-C*.txt", "Output-S*.txt", "log.txt",
                                                      "T-Controller*"))
    run_gcd_export(tmp_path / "cell" / "02_GCD" / REF_GCD.name, output_dir=tmp_path / "out")
    r = run_peis_export(tmp_path / "cell" / "03_PEIS" / REF.name, output_dir=tmp_path / "out")
    cmp = r["comparison"].set_index("check")
    esr = cmp.loc["GCD ESR vs Z'(1 kHz)"]
    assert esr["peis_spectrum"] == "2.5 V"
    assert esr["peis"] == pytest.approx(2.803, abs=0.001)
    assert esr["other"] == pytest.approx(2.80, abs=0.005)
    assert abs(esr["deviation_pct"]) < 0.5 and esr["severity"] == "info"
    c = cmp.loc["GCD C within C_s(f_min) range"]
    assert c["other"] == pytest.approx(1.022, abs=0.001) and c["severity"] == "info"
