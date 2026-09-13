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
from pathlib import Path

from .loader       import load_experiment, read_tester_txt
from .analysis     import compute_specific_capacitance, compute_mean_capacitance
from .plots        import plot_cv_curves, plot_capacitance_vs_scanrate
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

    Returns "cv" if any row has type == "VS", "gcd" if any row has
    type == "CC" with c/d in {"c", "d"}, else "unknown".
    """
    folder = Path(folder)
    out_file = folder / "Output.txt"
    if not out_file.exists():
        return "unknown"

    df = read_tester_txt(out_file)
    if (df["type"] == "VS").any():
        return "cv"
    if ((df["type"] == "CC") & df["c/d"].isin(["c", "d"])).any():
        return "gcd"
    return "unknown"


def run(
    cv_folder: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """
    Full analysis pipeline for one cyclic voltammetry measurement folder.

    Parameters
    ----------
    cv_folder  : path to the measurement folder (contains Output.txt + info.json)
    output_dir : where to save CSVs and PDFs; defaults to
                 <experiment_root>/output/<cell_group>/<cell_tag>/
    """
    cv_folder = Path(cv_folder)
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(cv_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  Supercap CV Pipeline")
    print(f"  Folder : {cv_folder}")
    print(f"  Output : {output_dir}")
    print(f"{'-'*60}")

    df, meta = load_experiment(cv_folder)

    print(f"\n  Cell tag    : {meta['cell_tag']}")
    print(f"  Experiment  : {meta['exp_name']}")
    print(f"  Electrode 1 : {meta['e1_name']}  m1 = {meta['m1']*1000:.3f} mg")
    if meta["m2"]:
        print(f"  Electrode 2 : {meta['e2_name']}  m2 = {meta['m2']*1000:.3f} mg")
    print(f"  Lightest m  : {meta['m_lightest']*1000:.3f} mg")
    print(f"  Electrolyte : {meta['electrolyte']}")

    vs_df = df[(df["type"] == "VS") & (df["cycle"] > 0)]
    conditions = (
        vs_df[["scan_speed_mVs", "voltage_window_V"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["voltage_window_V", "scan_speed_mVs"])
    )
    print(f"\n  Detected conditions ({len(conditions)}):")
    for _, row in conditions.iterrows():
        n_cyc = vs_df[
            (vs_df["scan_speed_mVs"] == row["scan_speed_mVs"]) &
            (vs_df["voltage_window_V"] == row["voltage_window_V"])
        ]["cycle"].nunique()
        print(f"    +/-{row['voltage_window_V']:.1f} V | {row['scan_speed_mVs']:.0f} mV/s | {n_cyc} cycles")

    # ── 2. Analysis ───────────────────────────────────────────────────────────
    print("\n  Computing specific capacitance per cycle …")
    cap_df = compute_specific_capacitance(df, meta)

    print(f"\n  Per-cycle C_sp [F/g]:")
    print(cap_df.to_string(index=False))

    mean_df = compute_mean_capacitance(cap_df, exclude_first_cycle=True)
    print(f"\n  Mean C_sp (cycles 2–end) [F/g]:")
    print(mean_df.to_string(index=False))

    # Save CSVs
    cap_df.to_csv(output_dir / "Csp_per_cycle.csv", index=False)
    mean_df.to_csv(output_dir / "Csp_mean.csv", index=False)
    print(f"\n  Saved CSVs -> {output_dir}")

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating CV plots …")
    cv_paths  = plot_cv_curves(df, meta, output_dir)
    print(f"  Generating C_sp vs scan-rate plots …")
    csp_paths = plot_capacitance_vs_scanrate(mean_df, cap_df, meta, output_dir)

    all_paths = cv_paths + csp_paths
    # Group by suffix for a tidy summary
    by_ext: dict[str, list[str]] = {}
    for p in all_paths:
        by_ext.setdefault(p.suffix, []).append(p.name)
    print(f"\n  Saved {len(all_paths)} files:")
    for ext in sorted(by_ext):
        print(f"    {ext}: {', '.join(sorted(by_ext[ext]))}")

    print(f"\n{'-'*60}")
    print("  Done.")
    print(f"{'-'*60}\n")


def run_gcd(
    gcd_folder: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """
    Full analysis pipeline for one galvanostatic charge/discharge (GCD)
    measurement folder.

    Parameters
    ----------
    gcd_folder : path to the measurement folder (Output.txt + info.json,
                 the latter requiring a `gcdConfig.basis` entry and valid
                 Weight1/Weight2 -- see gcd_loader.load_gcd_experiment())
    output_dir : where to save CSVs, the QC report and PDFs; defaults to
                 <experiment_root>/output/<cell_group>/<cell_tag>/
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
    df, meta = load_gcd_experiment(gcd_folder)

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


def run_experiment(
    cell_folder: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """
    Run CV and/or GCD analysis for every step subfolder of one cell folder
    (e.g. `<exp>/<cell_group>/<cell_tag>/`), dispatching per subfolder based
    on detect_technique(). Subfolders with no Output.txt, or whose content
    matches neither technique, are skipped and logged.
    """
    cell_folder = Path(cell_folder)
    subfolders = sorted(p for p in cell_folder.iterdir() if p.is_dir())

    print(f"\n{'='*60}")
    print(f"  Supercap Experiment Dispatch")
    print(f"  Cell folder : {cell_folder}")
    print(f"{'='*60}")

    ran = 0
    for sub in subfolders:
        if not (sub / "Output.txt").exists():
            continue
        technique = detect_technique(sub)
        print(f"\n  [{sub.name}] -> detected technique: {technique}")
        if technique == "cv":
            run(sub, output_dir=output_dir)
            ran += 1
        elif technique == "gcd":
            run_gcd(sub, output_dir=output_dir)
            ran += 1
        else:
            log.warning("Skipping %s: could not determine technique from Output.txt content.", sub)

    if ran == 0:
        log.warning("No CV or GCD subfolders with recognizable Output.txt found in %s.", cell_folder)

    print(f"\n{'='*60}")
    print(f"  Experiment dispatch done: {ran} subfolder(s) analyzed.")
    print(f"{'='*60}\n")
