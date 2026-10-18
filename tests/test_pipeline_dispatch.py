"""
tests/test_pipeline_dispatch.py
---------------------------------
detect_technique() classifies a measurement folder from its Output.txt
CONTENT (not its folder name -- real GCD step folders are named e.g.
"01_Galvanostatic Polarization Method 2.5V(3)", which contains neither
"CV" nor "GCD"). run_experiment() uses that to dispatch a whole cell
folder (possibly containing both CV and GCD step subfolders) to the
right pipeline, without touching the existing single-folder run()/
run_gcd() signatures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from supercap_pipeline.pipeline import detect_technique, run_experiment


def _write_output_txt(folder: Path, rows: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    header = "t\tcycle\tstep\ttype\tc/d\tV12\tI12\r\n"
    units  = "s\t\t\t\t\tV\tmA\r\n"
    (folder / "Output.txt").write_text(header + units + "\r\n".join(rows) + "\r\n", encoding="ascii")


def test_detect_technique_cv(tmp_path):
    folder = tmp_path / "01_Cyclic Voltametry"
    _write_output_txt(folder, [
        "0.0\t1\t10\tVS\t-\t0.0\t0.1",
        "0.1\t1\t10\tVS\t-\t0.1\t0.1",
    ])
    assert detect_technique(folder) == "cv"


def test_detect_technique_gcd(tmp_path):
    folder = tmp_path / "01_Galvanostatic Polarization Method 2.5V(3)"
    _write_output_txt(folder, [
        "0.0\t1\t20\tCC\tc\t0.0\t1.0",
        "0.1\t1\t20\tCC\tc\t0.1\t1.0",
        "10.0\t1\t30\tCC\td\t2.5\t-1.0",
    ])
    assert detect_technique(folder) == "gcd"


def test_detect_technique_unknown(tmp_path):
    folder = tmp_path / "weird_folder"
    _write_output_txt(folder, [
        "0.0\t0\t10\tRest\t-\t1.0\t0.0",
    ])
    assert detect_technique(folder) == "unknown"


def test_detect_technique_missing_output_txt(tmp_path):
    folder = tmp_path / "empty_folder"
    folder.mkdir()
    assert detect_technique(folder) == "unknown"


def test_run_experiment_dispatches_per_subfolder(tmp_path, monkeypatch):
    cell_folder = tmp_path / "A9999"
    _write_output_txt(cell_folder / "01_Cyclic Voltametry", [
        "0.0\t1\t10\tVS\t-\t0.0\t0.1",
    ])
    _write_output_txt(cell_folder / "02_Galvanostatic Polarization Method 2.5V(3)", [
        "0.0\t1\t20\tCC\tc\t0.0\t1.0",
    ])
    (cell_folder / "03_no_output_here").mkdir(parents=True)

    calls = {"cv": [], "gcd": []}
    monkeypatch.setattr(
        "supercap_pipeline.pipeline.run",
        lambda folder, output_dir=None: calls["cv"].append(Path(folder).name),
    )
    monkeypatch.setattr(
        "supercap_pipeline.pipeline.run_gcd",
        lambda folder, output_dir=None: calls["gcd"].append(Path(folder).name),
    )

    run_experiment(cell_folder)

    assert calls["cv"] == ["01_Cyclic Voltametry"]
    assert calls["gcd"] == ["02_Galvanostatic Polarization Method 2.5V(3)"]


def test_run_experiment_finds_nested_measurements_from_experiment_root(tmp_path, monkeypatch):
    # <exp>/<cell_group>/<cell_tag>/<step>/Output.txt, run from <exp>
    exp = tmp_path / "2026-10-13-BIA-VR-017-MC-Supercap-FullCell"
    _write_output_txt(exp / "BIA-VR-017-01" / "A0001" / "01_Cyclic Voltametry", [
        "0.0\t1\t10\tVS\t-\t0.0\t0.1",
    ])
    _write_output_txt(exp / "BIA-VR-017-02" / "A0002" / "01_Cyclic Voltametry", [
        "0.0\t1\t10\tVS\t-\t0.0\t0.1",
    ])

    calls = []
    monkeypatch.setattr(
        "supercap_pipeline.pipeline.run",
        lambda folder, output_dir=None: calls.append((Path(folder), output_dir)),
    )

    out = tmp_path / "out"
    run_experiment(exp, output_dir=out)

    assert [f.relative_to(exp) for f, _ in calls] == [
        Path("BIA-VR-017-01/A0001/01_Cyclic Voltametry"),
        Path("BIA-VR-017-02/A0002/01_Cyclic Voltametry"),
    ]
    # several measurements + explicit -o -> separate subdirectories, no overwriting
    assert [o for _, o in calls] == [
        out / "BIA-VR-017-01/A0001/01_Cyclic Voltametry",
        out / "BIA-VR-017-02/A0002/01_Cyclic Voltametry",
    ]


def test_run_experiment_on_single_measurement_folder(tmp_path, monkeypatch):
    folder = tmp_path / "A0001" / "01_Cyclic Voltametry"
    _write_output_txt(folder, ["0.0\t1\t10\tVS\t-\t0.0\t0.1"])

    calls = []
    monkeypatch.setattr(
        "supercap_pipeline.pipeline.run",
        lambda folder, output_dir=None: calls.append((Path(folder), output_dir)),
    )

    run_experiment(folder, output_dir=tmp_path / "out")
    assert calls == [(folder, tmp_path / "out")]


def test_run_experiment_continues_after_failure(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    _write_output_txt(exp / "A0001" / "01_CV", ["0.0\t1\t10\tVS\t-\t0.0\t0.1"])
    _write_output_txt(exp / "A0002" / "01_CV", ["0.0\t1\t10\tVS\t-\t0.0\t0.1"])

    calls = []

    def fake_run(folder, output_dir=None):
        calls.append(Path(folder).parent.name)
        if Path(folder).parent.name == "A0001":
            raise FileNotFoundError("No info.json found")

    monkeypatch.setattr("supercap_pipeline.pipeline.run", fake_run)

    run_experiment(exp)
    assert calls == ["A0001", "A0002"]
