"""
tests/test_gcd_loader.py
--------------------------
The task is explicit: no fallback values for missing masses or a missing
normalization convention -- a hard error with a clear message instead.
These tests pin that behaviour down so it can't regress silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supercap_pipeline.gcd_loader import load_gcd_experiment

_MINIMAL_OUTPUT_TXT = (
    "t\tcycle\tstep\ttype\tc/d\tV12\tI12\r\n"
    "s\t\t\t\t\tV\tmA\r\n"
    "0.0\t1\t20\tCC\tc\t0.0\t1.0\r\n"
    "1.0\t1\t20\tCC\tc\t0.5\t1.0\r\n"
)


def _write_experiment(folder: Path, composition_fields: list[dict], gcd_config: dict | None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Output.txt").write_text(_MINIMAL_OUTPUT_TXT, encoding="ascii")
    info = {
        "experiment": {"experimentName": "test-exp"},
        "testCell": {"cellTag": "X1", "composition": {"fields": composition_fields}},
    }
    if gcd_config is not None:
        info["gcdConfig"] = gcd_config
    (folder / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _weight_field(key: str, value) -> dict:
    return {"definition": {"key": key}, "value": value}


_GOOD_FIELDS = [_weight_field("Weight1", 0.01), _weight_field("Weight2", 0.01)]


def test_missing_gcd_config_raises(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config=None)
    with pytest.raises(ValueError, match="gcdConfig"):
        load_gcd_experiment(tmp_path)


def test_invalid_basis_raises(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config={"basis": "not_a_real_basis"})
    with pytest.raises(ValueError, match="basis"):
        load_gcd_experiment(tmp_path)


def test_missing_weight1_raises(tmp_path):
    fields = [_weight_field("Weight2", 0.01)]
    _write_experiment(tmp_path, fields, gcd_config={"basis": "cell"})
    with pytest.raises(ValueError, match="Weight1"):
        load_gcd_experiment(tmp_path)


def test_zero_weight_raises(tmp_path):
    fields = [_weight_field("Weight1", 0.0), _weight_field("Weight2", 0.01)]
    _write_experiment(tmp_path, fields, gcd_config={"basis": "cell"})
    with pytest.raises(ValueError, match="Weight1"):
        load_gcd_experiment(tmp_path)


def test_valid_config_loads(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config={"basis": "total_active_mass"})
    df, meta = load_gcd_experiment(tmp_path)
    assert meta["basis"] == "total_active_mass"
    assert meta["mass_total_g"] == pytest.approx(0.02)
    assert len(df) == 2


def test_explicit_basis_replaces_missing_gcd_config(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config=None)
    _, meta = load_gcd_experiment(tmp_path, basis="single_electrode")
    assert meta["basis"] == "single_electrode"


def test_explicit_basis_overrides_info_json(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config={"basis": "cell"})
    _, meta = load_gcd_experiment(tmp_path, basis="total_active_mass")
    assert meta["basis"] == "total_active_mass"


def test_invalid_explicit_basis_raises(tmp_path):
    _write_experiment(tmp_path, _GOOD_FIELDS, gcd_config=None)
    with pytest.raises(ValueError, match="basis"):
        load_gcd_experiment(tmp_path, basis="per_banana")
