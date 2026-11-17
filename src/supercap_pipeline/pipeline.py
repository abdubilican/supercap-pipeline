"""
pipeline.py
-----------
Orchestrates the evaluation of one measurement -- CV (cyclic voltammetry,
run()) or GCD (galvanostatic charge/discharge, run_gcd()) -- and of whole
experiment folders (run_experiment()).

The technique is decided per file from the step types in the export
(VS = voltage sweep -> CV, CC = constant current -> GCD; impedance steps
-> skipped), never from folder names.

Output layout: <root>/<sample-id>/cv/ and <root>/<sample-id>/gcd/, where
<root> is the -o folder or, by default, <parent of the measurement>/output/.
A second measurement of the same technique for the same sample gets the
measurement folder name as suffix (gcd_<folder>).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import cv_config
from . import gcd_analysis as gana
from . import gcd_output as gout
from . import gcd_plots as gplots
from . import gcd_segmentation as gseg
from .export_io import (
    GCD_COLUMNS, find_mass_file, is_export, measurement_exports, read_export, read_masses,
    sample_id_for, step_types,
)
from .plot_style import PlotContext
from .cv_segmentation import detect_blocks, drop_transition_sweeps, find_sweeps, pair_cycles
from .cv_analysis import evaluate_blocks
from .cv_output import meta_table, results_table, summary_table, write_results
from .cv_plots import plot_all
from .run_log import evaluation_log

log = logging.getLogger(__name__)
log_cv = logging.getLogger("supercap_pipeline.cv")
log_gcd = logging.getLogger("supercap_pipeline.gcd")


def detect_technique(folder: str | Path) -> str:
    """
    Classify a measurement (folder or export file) by the step types in its
    export, not by folder name (step-folder names are free text, e.g.
    "01_Galvanostatic Polarization Method 2.5V(3)").

    Returns "eis" for impedance measurements (a step type mentioning EIS /
    impedance, or -- as a negative signal only -- "EIS" in the folder name;
    PEIS steps often carry a few CC rows that would otherwise look like GCD),
    "cv" if a step has type VS, "gcd" if a step has type CC, else "unknown".
    """
    folder = Path(folder)
    exports = [p for p in measurement_exports(folder) if p.exists()]
    if not exports:
        return "unknown"
    types = {t.upper() for p in exports for t in step_types(p)}
    if any(re.search("EIS|IMP|FRA", t) for t in types) or "EIS" in folder.name.upper():
        return "eis"
    if "VS" in types:
        if "CC" in types:
            log.warning("%s contains VS and CC steps; evaluated as CV", folder)
        return "cv"
    if "CC" in types:
        return "gcd"
    return "unknown"


def output_dir_for(export: Path, output_dir: str | Path | None, name: str) -> Path:
    """<root>/<sample-id>/<name>; root = output_dir or <parent of the measurement>/output."""
    measurement = export if export.suffix.lower() in (".xlsx", ".xlsm") else export.parent
    root = Path(output_dir) if output_dir is not None else measurement.parent / "output"
    return root / sample_id_for(export) / name


# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

def run(cv_folder: str | Path, output_dir: str | Path | None = None,
        out_name: str = "cv") -> None:
    """
    CV evaluation (lab guide method) for one measurement: a folder with
    Output.txt and/or xlsx exports (sheet "Output"), or one export file.
    Each xlsx export in a folder is evaluated; Output.txt only if the folder
    has no xlsx export. Mass file: Input-Variables.txt next to the export.
    Results go to <root>/<sample-id>/cv/ (see output_dir_for).
    """
    for export in measurement_exports(cv_folder):
        run_cv_export(export, output_dir, out_name)


def run_cv_export(export: str | Path, output_dir: str | Path | None = None,
                  out_name: str = "cv") -> dict:
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
    out = output_dir_for(export, output_dir, out_name)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print("  Supercap CV Pipeline (lab-guide method)")
    print(f"  Export : {export}")
    print(f"  Output : {out}")
    print(f"{'-'*60}")

    with evaluation_log(out / "cv.log", log_cv.name) as warnings:
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
            paths = plot_all(df, blocks, masses, sample_id, out / "plots", source_file=export.name)
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


# ─────────────────────────────────────────────────────────────────────────────
# GCD
# ─────────────────────────────────────────────────────────────────────────────

def run_gcd(gcd_folder: str | Path, output_dir: str | Path | None = None,
            basis: str | None = None, out_name: str = "gcd") -> None:
    """
    GCD evaluation (lab guide method) for one measurement folder or export
    file; results go to <root>/<sample-id>/gcd/. `basis` is accepted for
    CLI compatibility but ignored: the guide fixes the mass convention
    (j per m_mean, C_el = 2C/m_mean, E_sp and P_sp per m1 + m2).
    """
    if basis is not None:
        log.warning("--gcd-basis %s ignored: the lab-guide GCD method fixes the mass "
                    "convention (m_mean = (m1 + m2)/2)", basis)
    for export in measurement_exports(gcd_folder):
        run_gcd_export(export, output_dir, out_name)


def evaluate_gcd(df: pd.DataFrame, masses: dict, sample_dir: Path | None = None) -> dict:
    """
    The GCD evaluation chain on an export DataFrame (read with GCD_COLUMNS):
      find_steps / classify_steps (half-cycles, rests, holds) -> build_cycles
      -> detect_blocks -> classify_tests -> mark_complete -> evaluate_cycles
      -> cycle_checks -> select_cycles -> rate_summary -> compare_with_cv
      (if `sample_dir` has cv/results.xlsx) -> longterm_cycling -> holds_table
      -> longterm_floating.
    """
    steps = gseg.classify_steps(df, gseg.find_steps(df))
    cycles = gseg.build_cycles(df, steps)
    n_holds = int((steps["kind"] == gseg.HOLD).sum())
    res = dict(steps=steps, n_holds=n_holds, ev=pd.DataFrame(), rate=pd.DataFrame(),
               series_flags=[], cv_comparison=None, cyc_t=pd.DataFrame(), cycling=[],
               holds=pd.DataFrame(), fl_t=pd.DataFrame(), floating=[])
    if cycles.empty:
        return res
    cycles = gseg.detect_blocks(cycles)
    cycles = gseg.classify_tests(cycles, has_holds=n_holds > 0)
    cycles = gseg.mark_complete(cycles)
    ev = gana.evaluate_cycles(df, cycles, masses)
    ev["cycle_flags"] = gana.cycle_checks(ev)
    selection = gseg.select_cycles(ev[ev["test_type"] == gseg.RATE])
    rate, series_flags = gana.rate_summary(ev, selection)
    cv_cmp = gana.compare_with_cv(sample_dir, rate) if sample_dir is not None else None
    if cv_cmp:
        (log_gcd.warning if cv_cmp["severity"] == gana.FLAG else log_gcd.info)(
            "CV comparison: %s", cv_cmp["text"])
    cyc_t, cyc_s = gana.longterm_cycling(ev)
    holds = gana.holds_table(df, steps, masses)
    fl_t, fl_s = gana.longterm_floating(ev, holds)
    res.update(ev=ev, rate=rate, series_flags=series_flags, cv_comparison=cv_cmp,
               cyc_t=cyc_t, cycling=cyc_s, holds=holds, fl_t=fl_t, floating=fl_s)
    return res


def run_gcd_export(export: str | Path, output_dir: str | Path | None = None,
                   out_name: str = "gcd") -> dict:
    """
    Evaluate one GCD export file (lab guide GCD):
      read_masses -> read_export -> find_steps / classify_steps (half-cycles,
      rests, holds) -> build_cycles -> detect_blocks -> classify_tests ->
      mark_complete -> evaluate_cycles -> cycle_checks -> select_cycles ->
      rate_summary / compare_with_cv / longterm_cycling / longterm_floating
      -> results.xlsx (+ csv per sheet) + plots (+ xlsx per plot) + gcd.log.
    Stops with an error if the masses are missing.
    """
    export = Path(export)
    sample_id = sample_id_for(export)
    out = output_dir_for(export, output_dir, out_name)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print("  Supercap GCD Pipeline (lab-guide method)")
    print(f"  Export : {export}")
    print(f"  Output : {out}")
    print(f"{'-'*60}")

    with evaluation_log(out / "gcd.log", log_gcd.name) as warnings:
        try:
            masses = read_masses(find_mass_file(export))
            df = read_export(export, GCD_COLUMNS)
            res = evaluate_gcd(df, masses, sample_dir=out.parent)
            if res["ev"].empty:
                log_gcd.warning("No GCD cycle (charge followed by discharge) found in %s",
                                export.name)
            ev, rate, n_holds = res["ev"], res["rate"], res["n_holds"]
            series_flags, cv_cmp = res["series_flags"], res["cv_comparison"]
            cyc_t, cyc_s, fl_t, fl_s = res["cyc_t"], res["cycling"], res["fl_t"], res["floating"]
            holds, steps = res["holds"], res["steps"]

            sheets = {
                "Cycles": gout.cycles_table(ev) if len(ev) else pd.DataFrame(),
                "Rate": gout.rate_table(rate, ev),
                "Cycling": cyc_t, "Floating": fl_t, "Holds": holds,
                "Flags": gout.flags_table(ev, rate, series_flags, cv_cmp, cyc_s, fl_s)
                if len(ev) else pd.DataFrame(columns=["severity", "flag"]),
                "Meta": gout.meta_table(masses, sample_id, str(export), ev, n_holds, cv_cmp,
                                        cyc_s, fl_s),
            }
            for f in sheets["Flags"].itertuples():
                log_gcd.info("%s [block %s, cycle %s]: %s", f.severity,
                             "-" if pd.isna(f.block) else int(f.block),
                             "-" if pd.isna(f.cycle) else int(f.cycle), f.flag)
            ctx = PlotContext(sample_id=sample_id, source_file=export.name, masses=masses,
                              out=out / "plots")
            paths = gplots.plot_all(df, ev, rate, cyc_t, fl_t, ctx) if len(ev) else []
            sheets["Warnings"] = pd.DataFrame({"warning": list(warnings) or ["(none)"]})
            paths = gout.write_results(out, sheets) + paths
        except Exception as exc:
            log_gcd.error("Evaluation of %s stopped: %s", export.name, exc)
            raise

    print(f"\n  m1 = {masses['m1_g'] * 1000:.4g} mg, m2 = {masses['m2_g'] * 1000:.4g} mg, "
          f"m_mean = {masses['m_mean_g'] * 1000:.4g} mg  ({Path(masses['source']).name})")
    if len(ev):
        prog = ev.groupby("block").agg(
            test=("test_type", "first"), j_Ag=("j_block_Ag", "first"), cycles=("cycle", "size"),
            used=("used", "sum"), U_min=("U_min", "first"), U_max=("U_max_block", "first"))
        print(f"  {len(ev)} cycles, {len(prog)} blocks, {n_holds} voltage holds\n")
        print(prog.to_string(float_format=lambda x: f"{x:.4g}"))
        if not rate.empty and "C" in rate:
            cols = ["block", "j_block_Ag", "selected_cycle", "ESR", "C", "C_E", "C_el",
                    "retention_pct", "eta_C", "E_sp", "P_sp"]
            print("\n  Rate series (selected cycles):")
            print(rate[[c for c in cols if c in rate]].to_string(
                index=False, float_format=lambda x: f"{x:.4g}"))
        n_flag = int((sheets["Flags"]["severity"] == gana.FLAG).sum())
        print(f"\n  {n_flag} flag(s), see sheet Flags / gcd.log")
    print(f"\n  Saved {len(paths)} files to {out}")
    print(f"{'-'*60}\n")
    return dict(ev=ev, rate=rate, sheets=sheets, cycling=cyc_s, floating=fl_s,
                cv_comparison=cv_cmp, out=out, paths=paths, masses=masses, steps=steps)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment folders
# ─────────────────────────────────────────────────────────────────────────────

def find_measurement_folders(root: str | Path) -> list[Path]:
    """
    Every folder at or below `root` that directly contains an Output.txt or
    an xlsx export (sheet "Output"), sorted by path. Lets the CLI accept an
    experiment, cell-group, cell or single-measurement folder alike.
    """
    root = Path(root)
    found = {p.parent for p in root.rglob("Output.txt") if p.is_file()}
    found |= {p.parent for p in root.rglob("*.xls[xm]") if is_export(p)}
    return sorted(found)


def run_experiment(
    cell_folder: str | Path,
    output_dir: str | Path | None = None,
    gcd_basis: str | None = None,
) -> None:
    """
    Run CV and/or GCD evaluation for every measurement at or below
    `cell_folder` (an experiment, cell-group, cell or measurement folder, or
    a single export file), dispatching per measurement on detect_technique().
    Impedance (EIS) folders and folders matching neither technique are
    skipped; a measurement that fails is reported and does not stop the
    others. Results: <root>/<sample-id>/<technique>/ (see output_dir_for); a
    second measurement of the same technique and sample gets the suffix
    _<measurement folder name>.
    """
    cell_folder = Path(cell_folder)
    if cell_folder.is_file():
        technique = detect_technique(cell_folder)
        if technique == "gcd":
            run_gcd(cell_folder, output_dir=output_dir, basis=gcd_basis)
        else:
            run(cell_folder, output_dir=output_dir)
        return
    measurements = find_measurement_folders(cell_folder)

    print(f"\n{'='*60}")
    print("  Supercap Experiment Dispatch")
    print(f"  Folder      : {cell_folder}")
    print(f"  Measurements: {len(measurements)} folder(s) with an export")
    print(f"{'='*60}")

    ran = 0
    failed: list[tuple[Path, Exception]] = []
    seen: set[tuple[str, str]] = set()
    for sub in measurements:
        rel = sub.relative_to(cell_folder)
        technique = detect_technique(sub)
        print(f"\n  [{rel}] -> detected technique: {technique}")

        if technique == "eis":
            print("    impedance measurement -- not analyzed by this pipeline, skipped.")
            continue
        if technique == "unknown":
            log.warning("Skipping %s: could not determine technique from the export content.", sub)
            continue

        sample = sample_id_for(measurement_exports(sub)[0])
        out_name = technique if (sample, technique) not in seen else f"{technique}_{sub.name}"
        seen.add((sample, technique))

        try:
            if technique == "cv":
                run(sub, output_dir=output_dir, out_name=out_name)
            else:
                run_gcd(sub, output_dir=output_dir, basis=gcd_basis, out_name=out_name)
            ran += 1
        except Exception as exc:
            log.exception("Analysis failed for %s", sub)
            failed.append((sub, exc))

    if not measurements:
        log.warning("No folders with an export (Output.txt / xlsx) found in %s.", cell_folder)

    print(f"\n{'='*60}")
    print(f"  Experiment dispatch done: {ran} measurement(s) analyzed.")
    if failed:
        print(f"  {len(failed)} measurement(s) FAILED:")
        for sub, exc in failed:
            print(f"    {sub.relative_to(cell_folder)}: {type(exc).__name__}: {exc}")
    print(f"{'='*60}\n")
