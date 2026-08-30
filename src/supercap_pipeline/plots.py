"""
plots.py
--------
Generate all figures for a supercapacitor CV analysis.

Figure set
----------
1. CV curves (I/mass vs V)
   One figure per scan speed (1, 10, 100 mV/s).
   Each figure shows every cycle at every voltage window.
   Colors are consistent across figures: one colour per voltage window.

2. F/g vs scan speed
   One figure per voltage window (+/-1.0, +/-1.5, +/-2.0, +/-2.5 V).
   Mean +/- std error bars, x-axis on log scale.

Each figure is saved as both PDF and PNG.
A companion CSV is saved alongside every figure with the data it was drawn from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Consistent colour per voltage window ─────────────────────────────────────
WINDOW_COLORS: dict[float, str] = {
    1.0: "#2563EB",   # blue
    1.5: "#16A34A",   # green
    2.0: "#D97706",   # amber
    2.5: "#DC2626",   # red
}
_DEFAULT_COLOR = "#6B7280"   # grey for any unlisted window

plt.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize":  8,
    "figure.dpi":     150,
})


def _save_fig(fig: plt.Figure, base: Path) -> tuple[Path, Path]:
    """
    Save figure as both PDF and PNG.
    `base` must be a full path WITHOUT any extension, e.g. /out/cv_1mVs
    Returns (pdf_path, png_path).
    """
    pdf = Path(str(base) + ".pdf")
    png = Path(str(base) + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    return pdf, png


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_curves(
    df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> list[Path]:
    """
    One figure per scan speed showing I/mass vs V for all voltage windows.
    All cycles are plotted.

    Saves per figure:
      cv_<speed>mVs.pdf
      cv_<speed>mVs.png
      cv_<speed>mVs.csv  — columns: cycle, voltage_window_V, V12, I12_spec_mAg

    Returns list of all saved paths (PDFs + PNGs + CSVs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vs_df = df[(df["type"] == "VS") & (df["cycle"] > 0)].copy()
    scan_speeds = sorted(vs_df["scan_speed_mVs"].dropna().unique())
    windows     = sorted(vs_df["voltage_window_V"].dropna().unique())
    cell_tag    = meta["cell_tag"]

    saved: list[Path] = []

    for speed in scan_speeds:
        speed_df = vs_df[vs_df["scan_speed_mVs"] == speed]
        base = output_dir / f"cv_{speed:.0f}mVs"

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_rows = []
        for window in windows:
            win_df = speed_df[speed_df["voltage_window_V"] == window]
            for cyc in sorted(win_df["cycle"].unique()):
                cyc_df = win_df[win_df["cycle"] == cyc].sort_values("t")
                tmp = cyc_df[["V12", "I12_spec_mAg"]].copy()
                tmp.insert(0, "cycle", cyc)
                tmp.insert(1, "voltage_window_V", window)
                csv_rows.append(tmp)

        csv_df = pd.concat(csv_rows, ignore_index=True)
        csv_path = Path(str(base) + ".csv")
        csv_df.to_csv(csv_path, index=False)
        saved.append(csv_path)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        legend_handles: dict[float, plt.Line2D] = {}

        for window in windows:
            color  = WINDOW_COLORS.get(window, _DEFAULT_COLOR)
            win_df = speed_df[speed_df["voltage_window_V"] == window]

            for i, cyc in enumerate(sorted(win_df["cycle"].unique())):
                cyc_df = win_df[win_df["cycle"] == cyc].sort_values("t")
                line,  = ax.plot(
                    cyc_df["V12"],
                    cyc_df["I12_spec_mAg"],
                    color=color,
                    linewidth=0.9,
                    alpha=0.65,
                )
                if window not in legend_handles:
                    legend_handles[window] = line

        handles = [legend_handles[w] for w in windows if w in legend_handles]
        labels  = [f"+/-{w} V" for w in windows if w in legend_handles]
        ax.legend(handles, labels, title="Voltage window", loc="best", framealpha=0.8)

        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current density (mA/g)")
        ax.set_title(f"{cell_tag} - CV at {speed:.0f} mV/s")
        ax.grid(True, alpha=0.25, linestyle=":")

        fig.tight_layout()
        pdf, png = _save_fig(fig, base)
        plt.close(fig)
        saved.extend([pdf, png])

    return saved


def plot_capacitance_vs_scanrate(
    mean_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> list[Path]:
    """
    One figure per voltage window: F/g vs scan speed (log x-axis).
    Shows mean +/- std error bars and individual cycle scatter points.

    Saves per figure:
      csp_vs_scanrate_<window>V.pdf
      csp_vs_scanrate_<window>V.png
      csp_vs_scanrate_<window>V.csv  — columns: scan_speed_mVs, cycle,
                                        C_sp_Fg, C_sp_mean_Fg, C_sp_std_Fg, n_cycles

    Returns list of all saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows  = sorted(mean_df["voltage_window_V"].unique())
    cell_tag = meta["cell_tag"]
    saved: list[Path] = []

    for window in windows:
        color    = WINDOW_COLORS.get(window, _DEFAULT_COLOR)
        win_mean = mean_df[mean_df["voltage_window_V"] == window].sort_values("scan_speed_mVs")
        win_cyc  = cap_df[cap_df["voltage_window_V"] == window].copy()
        # Use integer tenths (e.g. 1.0V → "10", 2.5V → "25") to avoid dot-in-stem issues
        w_tag = f"{int(round(window * 10))}V"
        base  = output_dir / f"csp_vs_scanrate_{w_tag}"

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_df = win_cyc.merge(
            win_mean[["scan_speed_mVs", "C_sp_mean_Fg", "C_sp_std_Fg", "n_cycles"]],
            on="scan_speed_mVs",
            how="left",
        ).sort_values(["scan_speed_mVs", "cycle"]).reset_index(drop=True)
        csv_path = Path(str(base) + ".csv")
        csv_df.to_csv(csv_path, index=False)
        saved.append(csv_path)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.5, 4))

        for speed_val, grp in win_cyc.groupby("scan_speed_mVs"):
            ax.scatter(
                [speed_val] * len(grp),
                grp["C_sp_Fg"],
                color=color,
                alpha=0.3,
                s=18,
                zorder=2,
            )

        n_label = int(win_mean["n_cycles"].iloc[0]) if not win_mean.empty else "?"
        ax.errorbar(
            win_mean["scan_speed_mVs"],
            win_mean["C_sp_mean_Fg"],
            yerr=win_mean["C_sp_std_Fg"],
            fmt="o-",
            color=color,
            capsize=4,
            linewidth=1.5,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=3,
            label=f"mean +/- std (n={n_label} cycles)",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Scan rate (mV/s)")
        ax.set_ylabel("Specific capacitance (F/g)")
        ax.set_title(f"{cell_tag} - Csp vs scan rate [+/-{window} V window]")
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.25, linestyle=":", which="both")

        fig.tight_layout()
        pdf, png = _save_fig(fig, base)
        plt.close(fig)
        saved.extend([pdf, png])

    return saved
