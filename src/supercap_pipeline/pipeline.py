"""
pipeline.py
-----------
Orchestrates the full CV analysis for one experiment folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .loader   import load_experiment
from .analysis import compute_specific_capacitance, compute_mean_capacitance
from .plots    import plot_cv_curves, plot_capacitance_vs_scanrate

log = logging.getLogger(__name__)


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

    if output_dir is None:
        # Walk up: cv_folder is typically  <exp>/<cell_group>/<cell_tag>/<step>/
        # We resolve output at  <exp>/output/<cell_group>/<cell_tag>/
        parts = cv_folder.parts
        try:
            cell_tag   = parts[-2]          # e.g. "A7029"
            cell_group = parts[-3]          # e.g. "BIA-BA-244-01-1"
            exp_root   = Path(*parts[:-3])  # e.g. "C:/…/bat-pipeline/2026-…"
        except IndexError:
            exp_root   = cv_folder.parent.parent
            cell_group = cv_folder.parent.name
            cell_tag   = "output"
        output_dir = exp_root / "output" / cell_group / cell_tag
    else:
        output_dir = Path(output_dir)

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
