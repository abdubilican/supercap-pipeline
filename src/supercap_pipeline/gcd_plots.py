"""
gcd_plots.py
------------
Generate all figures for a GCD (galvanostatic charge/discharge) analysis.

Figure set
----------
1. V12 vs t_cycle per rate stage (charge+discharge, one continuous curve per
   cycle), slope-fit window and extrapolated IR-drop point marked on one
   representative cycle, for visual QC.
2. C_spec vs current density (both slope- and integral-method values),
   mean +/- std per rate stage, log x-axis (rate studies convention).
3. C_spec vs cycle number (both methods), points coloured by rate stage.
4. Capacity retention per rate stage (both methods), relative to the
   lowest-current rate stage.
5. Ragone plot: specific power vs. specific energy, log-log.
6. Coulomb/energy efficiency and ESR vs. cycle number.

Each figure is saved as both PDF and PNG, with a companion CSV holding the
data it was drawn from -- same convention as plots.py (CV module), whose
_save_fig() helper is reused here directly.

Axis convention: every GCD plot's linear axes are forced to start at 0
(`set_xlim(left=0)` / `set_ylim(bottom=0)`, upper bound autoscaled) -- this
is a GCD-specific convention and is NOT applied to the CV module's plots
(whose curves are centered on 0 by construction). The one exception is the
C_spec-vs-current-density plot, which keeps its log x-axis (rate-study
convention); only its y-axis is floored at 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gcd_analysis import step_frame
from .plots import _save_fig

# ── Consistent colour per rate stage (cycled if more stages than colours) ────
RATE_COLORS: list[str] = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]


def _color_for_rate(rate_group: int) -> str:
    return RATE_COLORS[rate_group % len(RATE_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_gcd_curves(
    df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> list[Path]:
    """
    One figure per rate stage: V12 vs t_cycle (continuous through charge
    THEN discharge, using the tester's own per-cycle clock) for every
    complete cycle of that stage, slope-fit window shaded and IR-drop
    extrapolation marked on the first non-OCV-flagged cycle. Axes start
    at (0, 0).

    Saves per figure: gcd_curves_rate<NN>.pdf/.png/.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    saved: list[Path] = []

    rate_groups = sorted(cycle_df["rate_group"].unique())
    for rg in rate_groups:
        rg_cycles = cycle_df[cycle_df["rate_group"] == rg].sort_values("cycle")
        color = _color_for_rate(int(rg))
        base = output_dir / f"gcd_curves_rate{int(rg):02d}"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        csv_rows = []

        rep_row = None
        for _, row in rg_cycles.iterrows():
            cyc_steps = steps_df[steps_df["cycle"] == row["cycle"]]
            charge_steps = cyc_steps[cyc_steps["kind"] == "charge"]
            dis_steps = cyc_steps[cyc_steps["kind"] == "discharge"]
            if charge_steps.empty or dis_steps.empty:
                continue
            charge_id = int(charge_steps.iloc[0]["step_id"])
            dis_id = int(dis_steps.iloc[0]["step_id"])

            cd_g = df.loc[df["step_id"].isin([charge_id, dis_id])].sort_values("t_cycle")
            ax.plot(cd_g["t_cycle"], cd_g["V12"], color=color, linewidth=0.9, alpha=0.5)

            tmp = cd_g[["t_cycle", "V12"]].copy()
            tmp.insert(0, "cycle", row["cycle"])
            tmp["phase"] = np.where(cd_g["step_id"] == charge_id, "charge", "discharge")
            csv_rows.append(tmp)

            if rep_row is None and not row["ocv_start"]:
                rep_row = row
                rep_dis_g = step_frame(df, dis_id)  # t_step-ordered, for the fit window

        if rep_row is not None:
            t_cycle_dis_start = float(rep_dis_g["t_cycle"].iloc[0])
            lo, hi = rep_row["u_max_V"] * 0.40, rep_row["u_max_V"] * 0.80
            ax.axhspan(lo, hi, color=color, alpha=0.12, label="slope-fit window")
            n = int(rep_row["ir_n_points"])
            first_pts = rep_dis_g.iloc[:n]
            ax.scatter(first_pts["t_cycle"], first_pts["V12"], color="black", s=14, zorder=3,
                       label="IR-drop fit points")
            ax.scatter([t_cycle_dis_start], [rep_row["u_max_V"] - rep_row["ir_drop_V"]],
                       marker="x", color="red", s=40, zorder=4,
                       label=f"V0 extrap. (cycle {int(rep_row['cycle'])})")

        ax.set_xlabel("t_cycle (s) -- charge then discharge")
        ax.set_ylabel("V12 (V)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{cell_tag} - GCD charge/discharge curves, rate stage {int(rg)} "
                     f"(I={rg_cycles['i_nominal_mA'].iloc[0]:.3f} mA, "
                     f"{rg_cycles['current_density_Ag'].iloc[0]:.4f} A/g)")
        ax.legend(loc="best", framealpha=0.8, fontsize=7)
        ax.grid(True, alpha=0.25, linestyle=":")
        fig.tight_layout()

        if csv_rows:
            csv_df = pd.concat(csv_rows, ignore_index=True)
            csv_path = Path(str(base) + ".csv")
            csv_df.to_csv(csv_path, index=False)
            saved.append(csv_path)

        pdf, png = _save_fig(fig, base)
        plt.close(fig)
        saved.extend([pdf, png])

    return saved


def plot_capacitance_vs_current_density(
    cycle_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> list[Path]:
    """
    C_spec (slope + integral method) vs. current density, log x-axis.
    Saves: csp_vs_current_density.pdf/.png/.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    base = output_dir / "csp_vs_current_density"

    df = cycle_df[~cycle_df["ocv_start"]].copy()
    agg = (
        df.groupby("rate_group")
          .agg(current_density_Ag=("current_density_Ag", "mean"),
               C_slope_mean=("C_spec_slope", "mean"), C_slope_std=("C_spec_slope", "std"),
               C_int_mean=("C_spec_integral", "mean"), C_int_std=("C_spec_integral", "std"))
          .reset_index()
          .sort_values("current_density_Ag")
    )
    agg[["C_slope_std", "C_int_std"]] = agg[["C_slope_std", "C_int_std"]].fillna(0.0)

    csv_path = Path(str(base) + ".csv")
    df[["rate_group", "cycle", "current_density_Ag", "C_spec_slope", "C_spec_integral"]].to_csv(
        csv_path, index=False
    )

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for _, row in df.iterrows():
        ax.scatter(row["current_density_Ag"], row["C_spec_slope"], color="#2563EB", alpha=0.3, s=16)
        ax.scatter(row["current_density_Ag"], row["C_spec_integral"], color="#D97706", alpha=0.3, s=16)

    ax.errorbar(agg["current_density_Ag"], agg["C_slope_mean"], yerr=agg["C_slope_std"],
                fmt="o-", color="#2563EB", capsize=4, label="slope method")
    ax.errorbar(agg["current_density_Ag"], agg["C_int_mean"], yerr=agg["C_int_std"],
                fmt="s-", color="#D97706", capsize=4, label="integral method")

    basis = meta["basis"]
    ax.set_xscale("log")  # log x-axis kept deliberately (rate-study convention)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Current density (A/g)")
    ax.set_ylabel(f"Specific capacitance (F/g, basis={basis})")
    ax.set_title(f"{cell_tag} - C_spec vs current density")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.25, linestyle=":", which="both")
    fig.tight_layout()

    pdf, png = _save_fig(fig, base)
    plt.close(fig)
    return [csv_path, pdf, png]


def plot_capacitance_vs_cycle(
    cycle_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> list[Path]:
    """
    C_spec (both methods) vs. cycle number, points coloured by rate stage.
    Saves: csp_vs_cycle.pdf/.png/.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    base = output_dir / "csp_vs_cycle"

    df = cycle_df.sort_values("cycle")
    csv_path = Path(str(base) + ".csv")
    df[["cycle", "rate_group", "i_nominal_mA", "current_density_Ag",
        "ocv_start", "C_spec_slope", "C_spec_integral"]].to_csv(csv_path, index=False)

    colors = [_color_for_rate(int(rg)) for rg in df["rate_group"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["cycle"], df["C_spec_slope"], c=colors, s=32, marker="o", label="slope method")
    ax.scatter(df["cycle"], df["C_spec_integral"], c=colors, s=32, marker="^", label="integral method")

    for rg, grp in df.groupby("rate_group"):
        ax.axvspan(grp["cycle"].min() - 0.5, grp["cycle"].max() + 0.5,
                   color=_color_for_rate(int(rg)), alpha=0.06)

    basis = meta["basis"]
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Specific capacitance (F/g, basis={basis})")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{cell_tag} - C_spec vs cycle (shaded band = rate stage)")
    ax.legend(framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()

    pdf, png = _save_fig(fig, base)
    plt.close(fig)
    return [csv_path, pdf, png]


def plot_capacity_retention(
    retention_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> list[Path]:
    """
    Capacity retention (%) per rate stage relative to the lowest-current
    rate stage, both methods, as a bar chart.
    Saves: capacity_retention.pdf/.png/.csv
    (retention_df: output of gcd_analysis.compute_capacity_retention())
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    base = output_dir / "capacity_retention"

    csv_path = Path(str(base) + ".csv")
    retention_df.to_csv(csv_path, index=False)

    x = np.arange(len(retention_df))
    width = 0.35
    colors = [_color_for_rate(int(rg)) for rg in retention_df["rate_group"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, retention_df["retention_slope_pct"], width,
           label="slope method", color=colors, alpha=0.9)
    ax.bar(x + width / 2, retention_df["retention_integral_pct"], width,
           label="integral method", color=colors, alpha=0.5)

    ax.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    labels = [f"{rg}\n({i:.3f} A/g)" for rg, i in
              zip(retention_df["rate_group"], retention_df["current_density_Ag"])]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Capacity retention (%)")
    ax.set_xlabel("Rate stage (current density)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{cell_tag} - Capacity retention vs. lowest rate stage")
    ax.legend(framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":", axis="y")
    fig.tight_layout()

    pdf, png = _save_fig(fig, base)
    plt.close(fig)
    return [csv_path, pdf, png]


def plot_ragone(
    cycle_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> list[Path]:
    """Specific power vs. specific energy (Ragone plot), log-log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    base = output_dir / "ragone"

    df = cycle_df[~cycle_df["ocv_start"]].copy()
    csv_path = Path(str(base) + ".csv")
    df[["rate_group", "cycle", "E_discharge_Whkg", "P_mean_Wkg", "P_max_Wkg"]].to_csv(
        csv_path, index=False
    )

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for rg, grp in df.groupby("rate_group"):
        ax.scatter(grp["E_discharge_Whkg"], grp["P_mean_Wkg"], color=_color_for_rate(int(rg)),
                   s=30, label=f"rate {int(rg)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Specific energy (Wh/kg)")
    ax.set_ylabel("Mean specific power (W/kg)")
    ax.set_title(f"{cell_tag} - Ragone plot")
    ax.legend(framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":", which="both")
    fig.tight_layout()

    pdf, png = _save_fig(fig, base)
    plt.close(fig)
    return [csv_path, pdf, png]


def plot_efficiency_esr_vs_cycle(
    cycle_df: pd.DataFrame,
    output_dir: Path,
    meta: dict,
) -> list[Path]:
    """Coulomb/energy efficiency and ESR vs. cycle number (two subplots)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = meta["cell_tag"]
    base = output_dir / "efficiency_esr_vs_cycle"

    df = cycle_df.sort_values("cycle")
    csv_path = Path(str(base) + ".csv")
    df[["cycle", "rate_group", "ocv_start", "coulomb_eff_charge", "energy_eff", "esr_Ohm"]].to_csv(
        csv_path, index=False
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    colors = [_color_for_rate(int(rg)) for rg in df["rate_group"]]

    ax1.scatter(df["cycle"], df["coulomb_eff_charge"] * 100, c=colors, s=24, label="Coulomb eff.")
    ax1.scatter(df["cycle"], df["energy_eff"] * 100, c=colors, marker="^", s=24, label="Energy eff.")
    ax1.set_ylabel("Efficiency (%)")
    ax1.legend(framealpha=0.8, fontsize=8)
    ax1.grid(True, alpha=0.25, linestyle=":")

    ax2.scatter(df["cycle"], df["esr_Ohm"] * 1000, c=colors, s=24)
    ax2.set_ylabel("ESR (mOhm)")
    ax2.set_xlabel("Cycle")
    ax2.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(f"{cell_tag} - Efficiency & ESR vs. cycle")
    fig.tight_layout()

    pdf, png = _save_fig(fig, base)
    plt.close(fig)
    return [csv_path, pdf, png]
