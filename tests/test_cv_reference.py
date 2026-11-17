"""
tests/test_cv_reference.py
--------------------------
Regression against the expected values of lab guide section 9 for
BIA-VR-009-01-05.xlsx (mass-independent part; m1 = m2 = 0.032164 g for the
specific values).

The raw file is not in the repository. Point SUPERCAP_REF_XLSX at it, or put
it at tests/fixtures/BIA-VR-009-01-05/BIA-VR-009-01-05.xlsx; otherwise the
test is skipped.

Tolerance: 0.5 % on charges and capacitances; values printed in the guide
with 4 (charges) / 3 (C) / 2 (eta) decimals additionally allow half a unit
of the last printed digit.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from supercap_pipeline.cv_analysis import evaluate_blocks
from supercap_pipeline.export_io import read_export
from supercap_pipeline.cv_segmentation import (
    detect_blocks, drop_transition_sweeps, find_sweeps, pair_cycles,
)

REF = Path(os.environ.get(
    "SUPERCAP_REF_XLSX",
    Path(__file__).parent / "fixtures" / "BIA-VR-009-01-05" / "BIA-VR-009-01-05.xlsx"))
pytestmark = pytest.mark.skipif(not REF.exists(), reason=f"reference file not found: {REF}")

M = 0.032164
MASSES = dict(m1_g=M, m2_g=M, m_mean_g=M, source="section 9")

# (U_max, nu) -> (Q(+U->0), Q(-U->0)) [mAh]
Q_DIS = {
    (1.0, 1): (0.2191, 0.2105), (1.0, 10): (0.1173, 0.1375), (1.0, 50): (-0.0146, -0.0074),
    (1.5, 1): (0.3669, 0.3572), (1.5, 10): (0.2313, 0.2484), (1.5, 50): (0.0007, 0.0314),
    (2.0, 1): (0.5421, 0.5199), (2.0, 10): (0.4976, 0.4861), (2.0, 50): (0.3768, 0.3749),
    (2.5, 1): (0.6333, 0.6371), (2.5, 10): (0.6613, 0.6605), (2.5, 50): (0.5306, 0.5364),
}
# (U_max, nu) -> (C+, C-, eta+, eta-)
C_ETA = {
    (1.0, 1): (0.789, 0.758, 0.60, 0.58), (1.0, 10): (0.422, 0.495, 0.47, 0.51),
    (1.5, 1): (0.880, 0.857, 0.56, 0.57), (1.5, 10): (0.555, 0.596, 0.51, 0.53),
    (2.0, 1): (0.976, 0.936, 0.53, 0.52), (2.0, 10): (0.896, 0.875, 0.69, 0.70),
    (2.0, 50): (0.678, 0.675, 0.62, 0.62),
    (2.5, 1): (0.912, 0.917, 0.40, 0.42), (2.5, 10): (0.952, 0.951, 0.67, 0.67),
    (2.5, 50): (0.764, 0.772, 0.63, 0.64),
}


def _close(got, expected, last_digit):
    return abs(got - expected) <= max(0.005 * abs(expected), 0.5 * last_digit + 1e-12)


@pytest.fixture(scope="module")
def evaluation():
    df = read_export(REF)
    sweeps = drop_transition_sweeps(find_sweeps(df))
    cycles = detect_blocks(pair_cycles(sweeps))
    blocks = evaluate_blocks(df, sweeps, cycles, MASSES)
    return {(b["U_max"], b["nu_nom_mVs"]): b for b in blocks}, blocks


def test_twelve_blocks(evaluation):
    by_key, blocks = evaluation
    assert len(blocks) == 12
    assert set(by_key) == set(Q_DIS)
    assert all(b["window_type"] == "symmetric" and b["occurrence"] == 1 for b in blocks)


def test_worked_example_rows(evaluation):
    b = evaluation[0][(1.0, 1)]
    assert b["up"]["cycle"] == 5
    assert (b["up"]["step"], b["up"]["row_start"], b["up"]["row_end"]) == (20, 291, 325)
    assert (b["down"]["step"], b["down"]["row_start"], b["down"]["row_end"]) == (30, 326, 359)
    assert (b["up"]["row_a"], b["up"]["row_b"]) == (315, 316)
    assert (b["down"]["row_a"], b["down"]["row_b"]) == (349, 350)
    assert b["up"]["Q0"] == pytest.approx(0.21049, rel=0.005)
    assert b["down"]["Q0"] == pytest.approx(-0.21913, rel=0.005)
    assert b["C"] == pytest.approx(0.773, rel=0.005)
    assert b["C_cell_sp"] == pytest.approx(12.0, abs=0.05)
    assert b["C_el"] == pytest.approx(48.1, abs=0.05)


@pytest.mark.parametrize("key", sorted(Q_DIS))
def test_discharge_charges(evaluation, key):
    b = evaluation[0][key]
    qp, qm = Q_DIS[key]
    assert _close(b["pol"]["+"]["Q_dis"], qp, 1e-4), (key, b["pol"]["+"]["Q_dis"], qp)
    assert _close(b["pol"]["-"]["Q_dis"], qm, 1e-4), (key, b["pol"]["-"]["Q_dis"], qm)


@pytest.mark.parametrize("key", sorted(C_ETA))
def test_capacitance_and_efficiency(evaluation, key):
    b = evaluation[0][key]
    cp, cm, ep, em = C_ETA[key]
    assert _close(b["pol"]["+"]["C"], cp, 1e-3)
    assert _close(b["pol"]["-"]["C"], cm, 1e-3)
    assert abs(b["pol"]["+"]["eta"] - ep) <= 0.005 + 1e-9
    assert abs(b["pol"]["-"]["eta"] - em) <= 0.005 + 1e-9


def test_expected_flags(evaluation):
    by_key = evaluation[0]
    for key, b in by_key.items():
        ne = any(f.startswith("not evaluable") for d in b["pol"].values() for f in d["flags"])
        assert ne == (key in {(1.0, 50), (1.5, 50)}), key
        assert all(any(f.startswith("eta") for f in d["flags"]) for d in b["pol"].values()), key
    for key in [(1.0, 10), (1.5, 10)]:
        assert any(f.startswith("polarity deviation") for f in by_key[key]["flags"])
    assert any(f.startswith("scan-rate trend") for f in by_key[(2.5, 10)]["flags"])
    trend = {k for k, b in by_key.items() if any(f.startswith("scan-rate trend") for f in b["flags"])}
    assert trend == {(2.5, 10)}
