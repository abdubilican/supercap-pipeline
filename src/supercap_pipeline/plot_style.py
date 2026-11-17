"""
plot_style.py
-------------
Plot style shared by the CV and GCD figures: matplotlib defaults, palettes,
axis styling, the mass subtitle, and save_figure() which writes PNG (300 dpi)
+ SVG, optionally a data CSV, and the Origin-ready <plot>.xlsx.

Colours: a fixed categorical order for voltage windows, and one sequential
palette (light -> dark blue) for magnitudes such as scan rate or current
density, so the same value has the same colour in every figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_excel import Column, Curve, write_plot_excel  # noqa: F401  (re-exported)
from .results_io import write_csv

PLOT_DPI = 300

plt.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize":  8,
    "figure.dpi":     150,
    "svg.fonttype":  "none",
})

# Categorical order for windows (fixed, never cycled past the end: windows
# beyond eight reuse the order with a different marker).
WINDOW_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
WINDOW_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
POLARITY_COLORS = {"+": "#eb6834", "-": "#2a78d6"}
INK = "#2b2b2b"
MUTED = "#8a8a86"
NOT_EVALUABLE = "#e34948"
ACCENT = "#eb6834"


@dataclass
class PlotContext:
    """What every figure of one measurement shares (subtitle, Meta sheet, folder)."""
    sample_id: str
    source_file: str
    masses: dict
    out: Path
    extra_meta: dict = field(default_factory=dict)


def sequential_colors(values) -> dict:
    """Sequential (light -> dark blue) colour per value, smallest lightest."""
    values = sorted(set(values))
    cmap = plt.get_cmap("Blues")
    pos = np.linspace(0.45, 0.95, len(values)) if len(values) > 1 else [0.8]
    return {v: cmap(p) for v, p in zip(values, pos)}


def subtitle(fig: plt.Figure, sample_id: str, masses: dict) -> None:
    fig.text(0.01, 0.995,
             f"{sample_id}  ·  m̄ = {masses['m_mean_g'] * 1000:.4g} mg per electrode "
             f"(m1 = {masses['m1_g'] * 1000:.4g} mg, m2 = {masses['m2_g'] * 1000:.4g} mg)",
             ha="left", va="top", fontsize=8, color=MUTED)


def style_axes(ax, zero_line: bool = True) -> None:
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if zero_line:
        ax.axhline(0, color=MUTED, linewidth=0.6)


def save_figure(fig: plt.Figure, ctx: PlotContext, name: str, *,
                curves: list[Curve] | None = None, csv: pd.DataFrame | None = None,
                mass_basis: dict | None = None, cycles_used: str = "") -> list[Path]:
    """
    Save `fig` as <name>.png (PLOT_DPI) and <name>.svg in ctx.out, plus
    <name>.csv if `csv` is given and the Origin-ready <name>.xlsx if `curves`
    is given (see plot_excel). Closes the figure; returns the written paths.
    """
    ctx.out.mkdir(parents=True, exist_ok=True)
    base = ctx.out / name
    png = Path(str(base) + ".png")
    svg = Path(str(base) + ".svg")
    fig.savefig(png, bbox_inches="tight", dpi=PLOT_DPI)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    paths = [png, svg]
    if csv is not None:
        paths.append(write_csv(csv, Path(str(base) + ".csv")))
    if curves is not None:
        m = ctx.masses
        meta = {
            "sample_id": ctx.sample_id,
            "source_file": ctx.source_file,
            "plot_file": png.name,
            "m1_g": m["m1_g"], "m2_g": m["m2_g"], "m_mean_g": m["m_mean_g"],
            **{f"mass_basis_{k}": v for k, v in (mass_basis or {}).items()},
            "cycles_used": cycles_used,
            **ctx.extra_meta,
        }
        paths.append(write_plot_excel(Path(str(base) + ".xlsx"), curves, meta))
    return paths
