"""
pipeline.py
-----------
Orchestrates the full analysis for one experiment folder -- CV (cyclic
voltammetry, run()) or GCD (galvanostatic charge/discharge, run_gcd()).

run_experiment() dispatches across a whole cell folder that may contain
both kinds of measurement as step subfolders: it looks at each
subfolder's Output.txt content (not its name) to decide which pipeline
applies, and runs both where present.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from . import cv_config
from .loader       import read_tester_txt
from .cv_io        import find_mass_file, is_cv_export, read_export, read_masses, sample_id_for
from .cv_segmentation import detect_blocks, drop_transition_sweeps, find_sweeps, pair_cycles
from .cv_analysis  import evaluate_blocks
from .cv_output    import meta_table, results_table, summary_table, write_results
from .plots        import plot_all
from .gcd_loader   import load_gcd_experiment
from .gcd_analysis import (
    segment_steps,
    compute_gcd_cycle_table,
    compute_gcd_rate_aggregate,
    compute_capacity_retention,
    build_qc_report,
)
from .gcd_plots import (
    plot_gcd_curves,
    plot_capacitance_vs_current_density,
    plot_capacitance_vs_cycle,
    plot_capacity_retention,
    plot_ragone,
    plot_efficiency_esr_vs_cycle,
)

log = logging.getLogger(__name__)


def _default_output_dir(folder: Path) -> Path:
    """
    Derive  <experiment_root>/output/<cell_group>/<cell_tag>/  from a
    measurement folder typically shaped  <exp>/<cell_group>/<cell_tag>/<step>/.
    """
    parts = folder.parts
    try:
        cell_tag   = parts[-2]          # e.g. "A7029"
        cell_group = parts[-3]          # e.g. "BIA-BA-244-01"
        exp_root   = Path(*parts[:-3])  # e.g. "C:/…/bat-pipeline/2026-…"
    except IndexError:
        exp_root   = folder.parent.parent
        cell_group = folder.parent.name
        cell_tag   = "output"
    return exp_root / "output" / cell_group / cell_tag


def detect_technique(folder: str | Path) -> str:
    """
    Classify a measurement folder's Output.txt by content, not by folder
    name (step-folder names are free text and not a reliable signal, e.g.
    "01_Galvanostatic Polarization Method 2.5V(3)").

    Returns "eis" for impedance measurements (any step type mentioning EIS
    / impedance, or -- as a negative signal only -- "EIS" in the folder
    name; PEIS steps often carry a few CC rows that would otherwise look
    like GCD), "cv" if any row has type == "VS", "gcd" if there are CC rows
    with both c/d == "c" and c/d == "d", else "unknown".
    """
    folder = Path(folder)
    out_file = folder / "Output.txt"
    if not out_file.exists():
        # xlsx export (sheet "Output"): only the CV pipeline reads these
        for xlsx in xlsx_exports(folder):
            if (read_export(xlsx)["type"] == "VS").any():
                return "cv"
        return "unknown"

    df = read_tester_txt(out_file)
    types = df["type"].str.upper()
    if types.str.contains("EIS|IMP|FRA", regex=True).any() or "EIS" in folder.name.upper():
        return "eis"
    if (types == "VS").any():
        return "cv"
    cc_dirs = set(df.loc[types == "CC", "c/d"]) if "c/d" in df.columns else set()
    if {"c", "d"} <= cc_dirs:
        return "gcd"
    return "unknown"


def run(
    cv_folder: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """
    CV evaluation (lab guide method) for one measurement: a folder with
    Output.txt and/or xlsx exports (sheet "Output"), or one export file.
    Each xlsx export in a folder is evaluated; Output.txt only if the folder
    has no xlsx export. Mass file: Input-Variables.txt next to the export.

    Results go to <output root>/<sample-id>/cv/ where the output root is
    `output_dir` or, by default, <parent of the measurement>/output/.
    """
    cv_path = Path(cv_folder)
    if cv_path.is_file():
        exports = [cv_path]
    else:
        exports = xlsx_exports(cv_path) or [cv_path / "Output.txt"]
    for export in exports:
        run_cv_export(export, output_dir)


def xlsx_exports(folder: Path) -> list[Path]:
    """xlsx files in `folder` (not below) that have a sheet "Output"."""
    return [p for p in sorted(folder.glob("*.xls[xm]")) if is_cv_export(p)]


def run_cv_export(export: str | Path, output_dir: str | Path | None = None) -> dict:
    """
    Evaluate one export file:
      read_masses -> read_export -> find_sweeps -> drop_transition_sweeps
      -> pair_cycles -> detect_blocks -> evaluate_blocks (charges,
      capacitance, quality checks) -> write_results + plots + cv.log.
    Stops with an error if the masses are missing. Returns a dict with the
    blocks, tables, output folder and written paths.
    """
    export = Path(export)
    sample_id = sample_id_for(export)
    measurement = export if export.suffix.lower() in (".xlsx", ".xlsm") else export.parent
    root = Path(output_dir) if output_dir is not None else measurement.parent / "output"
    out = root / sample_id / "cv"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print("  Supercap CV Pipeline (lab-guide method)")
    print(f"  Export : {export}")
    print(f"  Output : {out}")
    print(f"{'-'*60}")

    with _cv_log(out / "cv.log") as warnings:
        try:
            masses = read_masses(find_mass_file(export))
            if abs(masses["m1_g"] - masses["m2_g"]) / masses["m_mean_g"] > cv_config.MASS_ASYMMETRY_TOL:
                log_cv.warning("|m1 - m2| / m_mean > %.0f %%: C_el,exact = C*(1/m1 + 1/m2) "
                               "is reported in addition to C_el",
                               cv_config.MASS_ASYMMETRY_TOL * 100)
            df = read_export(export)
            sweeps = drop_transition_sweeps(find_sweeps(df))
            cycles = detect_blocks(pair_cycles(sweeps))
            blocks = evaluate_blocks(df, sweeps, cycles, masses)
            if not blocks:
                log_cv.warning("No evaluable CV block found in %s", export.name)

            results = results_table(blocks, sample_id, export.name)
            summary = summary_table(blocks, sample_id, export.name)
            meta = meta_table(masses, sample_id, str(export), sweeps.attrs.get("n_dropped", 0))
            paths = plot_all(df, blocks, masses, sample_id, out / "plots")
            paths = write_results(out, results, summary, meta, list(warnings)) + paths
        except Exception as exc:
            log_cv.error("Evaluation of %s stopped: %s", export.name, exc)
            raise

    print(f"\n  m1 = {masses['m1_g'] * 1000:.4g} mg, m2 = {masses['m2_g'] * 1000:.4g} mg, "
          f"m_mean = {masses['m_mean_g'] * 1000:.4g} mg  ({Path(masses['source']).name})")
    print(f"  Sweeps: {len(sweeps)} kept, {sweeps.attrs.get('n_dropped', 0)} transition sweeps "
          f"dropped; {len(cycles)} cycles; {len(blocks)} blocks\n")
    if blocks:
        cols = ["block", "occurrence", "window_type", "U_max_V", "nu_nom_mVs", "up_cycle",
                "n_complete_cycles", "C_plus_F", "C_minus_F", "C_F", "C_el_Fg",
                "eta_plus", "eta_minus", "flags"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}",
                                      max_colwidth=80))
    print(f"\n  Saved {len(paths)} files to {out}  (log: cv.log)")
    print(f"{'-'*60}\n")
    return dict(blocks=blocks, results=results, summary=summary, meta=meta,
                out=out, paths=paths, sweeps=sweeps, cycles=cycles, masses=masses)


log_cv = logging.getLogger("supercap_pipeline.cv")


class _WarningCollector(logging.Handler):
    def __init__(self, sink: list[str]):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage())


@contextmanager
def _cv_log(path: Path):
    """
    Route every message of the CV modules (INFO and up: dropped sweeps,
    unpaired sweeps, block decisions, flags, warnings) to `path`; warnings
    also go to the console and to the yielded list (-> Warnings sheet).
    """
    logger = log_cv
    old = (logger.level, logger.propagate, list(logger.handlers))
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("  %(levelname)s: %(message)s"))
    sink: list[str] = []
    logger.handlers = [fh, console, _WarningCollector(sink)]
    logger.setLevel(logging.DEBUG if logging.getLogger().level <= logging.DEBUG else logging.INFO)
    logger.propagate = False
    try:
        yield sink
    finally:
        fh.close()
        logger.setLevel(old[0])
        logger.propagate = old[1]
        logger.handlers = old[2]


def run_gcd(
    gcd_folder: str | Path,
    output_dir: str | Path | None = None,
    basis: str | None = None,
) -> None:
    """
    Full analysis pipeline for one galvanostatic charge/discharge (GCD)
    measurement folder.

    Parameters
    ----------
    gcd_folder : path to the measurement folder (Output.txt + info.json
                 with valid Weight1/Weight2 -- see
                 gcd_loader.load_gcd_experiment())
    output_dir : where to save CSVs, the QC report and PDFs; defaults to
                 <experiment_root>/output/<cell_group>/<cell_tag>/
    basis      : normalization convention; overrides info.json's
                 `gcdConfig.basis`, one of the two is required
    """
    gcd_folder = Path(gcd_folder)
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(gcd_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print(f"  Supercap GCD Pipeline")
    print(f"  Folder : {gcd_folder}")
    print(f"  Output : {output_dir}")
    print(f"{'-'*60}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df, meta = load_gcd_experiment(gcd_folder, basis=basis)

    print(f"\n  Cell tag    : {meta['cell_tag']}")
    print(f"  Experiment  : {meta['exp_name']}")
    print(f"  Electrode 1 : {meta['e1_name']}  m1 = {meta['m1_g']*1000:.3f} mg")
    print(f"  Electrode 2 : {meta['e2_name']}  m2 = {meta['m2_g']*1000:.3f} mg")
    print(f"  Electrolyte : {meta['electrolyte']}")
    print(f"  Basis       : {meta['basis']}")

    # ── 2. Analysis ───────────────────────────────────────────────────────────
    print("\n  Segmenting steps and computing per-cycle GCD metrics …")
    cycle_df, steps_df = compute_gcd_cycle_table(df, meta)

    print(f"\n  Per-cycle table ({len(cycle_df)} complete cycles):")
    print(cycle_df[[
        "cycle", "rate_group", "i_nominal_mA", "esr_Ohm", "esr_variant",
        "C_spec_slope", "C_spec_integral", "coulomb_eff_charge", "energy_eff",
        "ocv_start",
    ]].to_string(index=False))

    rate_df = compute_gcd_rate_aggregate(cycle_df, exclude_first_cycle=True)
    print(f"\n  Mean per rate stage (first cycle of each stage excluded):")
    print(rate_df.to_string(index=False))

    retention_df = compute_capacity_retention(rate_df)
    print(f"\n  Capacity retention vs. lowest rate stage:")
    print(retention_df.to_string(index=False))

    qc_text = build_qc_report(cycle_df, steps_df)
    print(f"\n{qc_text}")

    # Save CSVs + QC report
    cycle_df.to_csv(output_dir / "gcd_cycle_table.csv", index=False)
    rate_df.to_csv(output_dir / "gcd_rate_aggregate.csv", index=False)
    (output_dir / "gcd_qc_report.txt").write_text(qc_text, encoding="utf-8")
    # retention_df's own CSV is saved by plot_capacity_retention() below (capacity_retention.csv)
    print(f"\n  Saved CSVs + QC report -> {output_dir}")

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating GCD plots …")
    df_with_steps, _ = segment_steps(df)  # step_id column, needed by plot_gcd_curves
    paths = plot_gcd_curves(df_with_steps, cycle_df, steps_df, meta, output_dir)
    paths += plot_capacitance_vs_current_density(cycle_df, output_dir, meta)
    paths += plot_capacitance_vs_cycle(cycle_df, output_dir, meta)
    paths += plot_capacity_retention(retention_df, output_dir, meta)
    paths += plot_ragone(cycle_df, output_dir, meta)
    paths += plot_efficiency_esr_vs_cycle(cycle_df, output_dir, meta)

    by_ext: dict[str, list[str]] = {}
    for p in paths:
        by_ext.setdefault(p.suffix, []).append(p.name)
    print(f"\n  Saved {len(paths)} files:")
    for ext in sorted(by_ext):
        print(f"    {ext}: {', '.join(sorted(by_ext[ext]))}")

    print(f"\n{'-'*60}")
    print("  Done.")
    print(f"{'-'*60}\n")


def find_measurement_folders(root: str | Path) -> list[Path]:
    """
    Every folder at or below `root` that directly contains an Output.txt,
    sorted by path. Lets the CLI accept an experiment, cell-group, cell or
    single-measurement folder alike.
    """
    root = Path(root)
    found = {p.parent for p in root.rglob("Output.txt") if p.is_file()}
    found |= {p.parent for p in root.rglob("*.xls[xm]") if is_cv_export(p)}
    return sorted(found)


def run_experiment(
    cell_folder: str | Path,
    output_dir: str | Path | None = None,
    gcd_basis: str | None = None,
) -> None:
    """
    Run CV and/or GCD analysis for every measurement folder at or below
    `cell_folder` (an experiment, cell-group or cell folder), dispatching
    per folder based on detect_technique(). Impedance (EIS) folders and
    folders matching neither technique are skipped; a folder that fails
    to analyze is reported and does not stop the remaining ones.
    `gcd_basis` is passed to run_gcd() as the normalization convention.

    If `output_dir` is given and several measurements are found, each one
    writes to `output_dir/<path relative to cell_folder>` so results don't
    overwrite each other. Without it, each measurement uses its own
    default output directory.
    """
    cell_folder = Path(cell_folder)
    if cell_folder.is_file():  # a single CV export (.xlsx or Output.txt)
        run(cell_folder, output_dir=output_dir)
        return
    measurements = find_measurement_folders(cell_folder)

    print(f"\n{'='*60}")
    print(f"  Supercap Experiment Dispatch")
    print(f"  Folder      : {cell_folder}")
    print(f"  Measurements: {len(measurements)} folder(s) with Output.txt")
    print(f"{'='*60}")

    ran = 0
    failed: list[tuple[Path, Exception]] = []
    for sub in measurements:
        rel = sub.relative_to(cell_folder)
        technique = detect_technique(sub)
        print(f"\n  [{rel}] -> detected technique: {technique}")

        if technique == "eis":
            print("    impedance measurement -- not analyzed by this pipeline, skipped.")
            continue
        if technique == "unknown":
            log.warning("Skipping %s: could not determine technique from Output.txt content.", sub)
            continue

        sub_out = None
        if output_dir is not None:
            sub_out = Path(output_dir) if len(measurements) == 1 else Path(output_dir) / rel

        try:
            if technique == "cv":
                run(sub, output_dir=sub_out)
            else:
                run_gcd(sub, output_dir=sub_out, basis=gcd_basis)
            ran += 1
        except Exception as exc:
            log.exception("Analysis failed for %s", sub)
            failed.append((sub, exc))

    if not measurements:
        log.warning("No folders containing Output.txt found in %s.", cell_folder)

    print(f"\n{'='*60}")
    print(f"  Experiment dispatch done: {ran} measurement(s) analyzed.")
    if failed:
        print(f"  {len(failed)} measurement(s) FAILED:")
        for sub, exc in failed:
            print(f"    {sub.relative_to(cell_folder)}: {type(exc).__name__}: {exc}")
    print(f"{'='*60}\n")
