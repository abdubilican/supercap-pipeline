"""
plots.py
--------
Figures of the CV evaluation (lab guide 6.2), saved as PNG (300 dpi) + SVG:

1. cv_<nu>mVs           current density j vs V12, last cycle of every window at one nu
2. control_block<NN>_*  integration control plot per block: I12 vs V12 with the
                        discharge areas shaded and the interpolated 0 V points marked
3. rate_capability      C_el vs nu (log x) per window + retention vs slowest nu
4. window_plot          C_el and eta vs U_max at the slowest nu
5. normalised_<window>  2*I12[A] / (nu[V/s] * m_mean[g]) vs V12, all scan rates

One fixed colour per window in every figure; a sequential palette for scan rates.
Every figure also gets a <name>.csv with the data drawn in it (with Excel row,
instrument cycle and Qcd for the curve plots).

_save_fig() (PDF + PNG) and the rcParams below are also used by gcd_plots.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cv_config as cfg
from .cv_output import write_csv
from .cv_segmentation import SYMMETRIC

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


def _save_png_svg(fig: plt.Figure, base: Path, data: pd.DataFrame | None = None) -> list[Path]:
    """Save as PNG (PLOT_DPI) and SVG, plus `data` as <base>.csv; close the figure."""
    png = Path(str(base) + ".png")
    svg = Path(str(base) + ".svg")
    fig.savefig(png, bbox_inches="tight", dpi=cfg.PLOT_DPI)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    paths = [png, svg]
    if data is not None:
        paths.append(write_csv(data, Path(str(base) + ".csv")))
    return paths


def _curve_data(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Rows of the evaluated cycle with block identification, for the plot CSVs."""
    cyc = _cycle_frame(df, b)
    up_rows = set(range(b["up_sweep"].row_start, b["up_sweep"].row_end + 1))
    return pd.DataFrame({
        "window": window_label(window_key(b)),
        "U_max_V": b["U_max"],
        "nu_nom_mVs": b["nu_nom_mVs"],
        "block": b["block"],
        "occurrence": b["occurrence"],
        "excel_row": cyc["excel_row"].to_numpy(),
        "cycle": cyc["cycle"].to_numpy(),
        "step": cyc["step"].to_numpy(),
        "sweep": ["up" if r in up_rows else "down" for r in cyc["excel_row"]],
        "V12_V": cyc["V12"].to_numpy(),
        "I12_mA": cyc["I12"].to_numpy(),
        "Qcd_mAh": cyc["Qcd"].to_numpy(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────────────────────────────────────

def window_key(b: dict) -> tuple[str, float]:
    return (b["window_type"], b["U_max"])


def window_label(key: tuple[str, float]) -> str:
    wtype, u = key
    return f"±{u:g} V" if wtype == SYMMETRIC else f"0–{u:g} V"


def window_slug(key: tuple[str, float]) -> str:
    wtype, u = key
    return f"pm{u:.2f}V" if wtype == SYMMETRIC else f"0-{u:.2f}V"


def window_styles(blocks: list[dict]) -> dict:
    """{window key: (colour, marker)} -- symmetric windows first, then by U_max."""
    keys = sorted({window_key(b) for b in blocks}, key=lambda k: (k[0] != SYMMETRIC, k[1]))
    return {k: (WINDOW_PALETTE[i % len(WINDOW_PALETTE)],
                WINDOW_MARKERS[(i // len(WINDOW_PALETTE) + i) % len(WINDOW_MARKERS)])
            for i, k in enumerate(keys)}


def rate_colors(rates: list[float]) -> dict:
    """Sequential (light -> dark blue) colour per scan rate, slowest lightest."""
    rates = sorted(rates)
    cmap = plt.get_cmap("Blues")
    pos = np.linspace(0.45, 0.95, len(rates)) if len(rates) > 1 else [0.8]
    return {r: cmap(p) for r, p in zip(rates, pos)}


def _subtitle(fig: plt.Figure, sample_id: str, masses: dict) -> None:
    fig.text(0.01, 0.995,
             f"{sample_id}  ·  m̄ = {masses['m_mean_g'] * 1000:.4g} mg per electrode "
             f"(m1 = {masses['m1_g'] * 1000:.4g} mg, m2 = {masses['m2_g'] * 1000:.4g} mg)",
             ha="left", va="top", fontsize=8, color=MUTED)


def _style_axes(ax) -> None:
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color=MUTED, linewidth=0.6)


def _cycle_frame(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Rows of the evaluated cycle, in time order."""
    s1, s2 = sorted([b["up_sweep"], b["down_sweep"]], key=lambda s: s.i0)
    return pd.concat([df.iloc[s1.i0:s1.i1 + 1], df.iloc[s2.i0:s2.i1 + 1]])


def _last_occurrence(blocks: list[dict]) -> dict:
    """{(window key, nu): block} keeping the last occurrence of each combination."""
    out = {}
    for b in blocks:
        out[(window_key(b), b["nu_nom_mVs"])] = b
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_per_scan_rate(df, blocks, masses, sample_id, out: Path) -> list[Path]:
    """
    j = I12[mA] / (1000 * m_mean[g]) vs V12; one figure per nominal nu.
    CSV, exactly four columns: V12 (V) | I12/m_mean (A/g) | cycle | Qcd (mAh),
    one row per data point of the plotted cycles (windows follow each other;
    they are told apart by the instrument cycle number).
    """
    styles = window_styles(blocks)
    last = _last_occurrence(blocks)
    paths = []
    for nu in sorted({b["nu_nom_mVs"] for b in blocks}):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        data = []
        for key in styles:
            b = last.get((key, nu))
            if b is None:
                continue
            d = _curve_data(df, b)
            j = d["I12_mA"] / (1000.0 * masses["m_mean_g"])
            data.append(pd.DataFrame({
                "V12 (V)": d["V12_V"],
                "I12/m_mean (A/g)": j,
                "cycle": d["cycle"],
                "Qcd (mAh)": d["Qcd_mAh"],
            }))
            ax.plot(d["V12_V"], j, color=styles[key][0], linewidth=1.4,
                    label=f"{window_label(key)} (cycle {b['up']['cycle']})")
        _style_axes(ax)
        ax.set_xlabel("Cell voltage V12 (V)")
        ax.set_ylabel("Current density (A g⁻¹, per electrode)")
        ax.set_title(f"CV at {nu:g} mV s⁻¹ — last complete cycle per window", loc="left")
        ax.legend(title="Window", frameon=False)
        _subtitle(fig, sample_id, masses)
        paths += _save_png_svg(fig, out / f"cv_{nu:g}mVs", pd.concat(data, ignore_index=True))
    return paths


def plot_control(df, b, masses, sample_id, out: Path) -> list[Path]:
    """
    Integration control plot of one block: I12 vs V12 of the evaluated cycle,
    discharge areas shaded between curve and I = 0 (symmetric: +U->0 on the
    down sweep, -U->0 on the up sweep; one-sided: U->0 on the down sweep),
    interpolated 0 V points marked, Q_dis / C / eta and flags annotated.
    """
    key = window_key(b)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cyc = _cycle_frame(df, b)
    ax.plot(cyc["V12"], cyc["I12"], color=INK, linewidth=1.2, label="evaluated cycle")

    sym = b["window_type"] == SYMMETRIC
    parts = ([("+", b["down_sweep"], b["down"]), ("-", b["up_sweep"], b["up"])] if sym
             else [("+", b["down_sweep"], b["down"])])
    for p, sw, rec in parts:
        d = b["pol"][p]
        seg = df.iloc[sw.i0:sw.i1 + 1]
        if sym:
            if not np.isfinite(rec["Q0"]):
                continue
            a = int(np.flatnonzero(seg["excel_row"].to_numpy() == rec["row_a"])[0])
            V = np.r_[seg["V12"].to_numpy()[:a + 1], 0.0]
            I = np.r_[seg["I12"].to_numpy()[:a + 1], rec["I0"]]
        else:
            V, I = seg["V12"].to_numpy(), seg["I12"].to_numpy()
        poly_v = np.r_[V, V[-1], V[0]]
        poly_i = np.r_[I, 0.0, 0.0]
        bad = bool(d["not_evaluable"])
        seg_name = "+U→0" if (sym and p == "+") else ("−U→0" if sym else "U→0")
        if bad:
            ax.fill(poly_v, poly_i, facecolor="none", edgecolor=NOT_EVALUABLE, hatch="xx",
                    linewidth=0.8, label=f"{seg_name}: NOT EVALUABLE")
        else:
            ax.fill(poly_v, poly_i, color=POLARITY_COLORS[p], alpha=0.30, linewidth=0,
                    label=f"{seg_name} discharge (polarity {p})")
        if sym:
            ax.plot([0.0], [rec["I0"]], marker="o", markersize=8, color=POLARITY_COLORS[p],
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(f"rows {rec['row_a']}/{rec['row_b']}", (0.0, rec["I0"]),
                        textcoords="offset points", xytext=(6, 6 if p == "-" else -12),
                        fontsize=7, color=MUTED)

    lines = []
    for p, d in b["pol"].items():
        tag = "C+" if p == "+" else "C−"
        if not sym:
            tag = "C"
        c_txt = f"{d['C']:.4g} F" if np.isfinite(d["C"]) else "n.e."
        lines.append(f"{tag}: Q_dis = {d['Q_dis']:.4f} mAh, Q_ch = {d['Q_ch']:.4f} mAh, "
                     f"C = {c_txt}, η = {d['eta']:.2f}")
    if sym:
        lines.append(f"C = {b['C']:.4g} F" if np.isfinite(b["C"]) else "C = n.e.")
    flags = b["flags"] + [f"{p}: {f}" for p, d in b["pol"].items() for f in d["flags"]]
    if flags:
        lines.append("Flags: " + "; ".join(flags))
    text = "\n".join(textwrap.fill(ln, 95, subsequent_indent="   ") for ln in lines)
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=7.5, color=INK, family="monospace")

    _style_axes(ax)
    ax.set_xlabel("Cell voltage V12 (V)")
    ax.set_ylabel("Current I12 (mA)")
    ax.set_title(f"Block {b['block']}: {window_label(key)}, {b['nu_nom_mVs']:g} mV s⁻¹"
                 + (f" (occurrence {b['occurrence']})" if b["occurrence"] > 1 else "")
                 + f" — instrument cycle {b['up']['cycle']}, steps {b['up']['step']}/"
                   f"{b['down']['step']}", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="best")
    _subtitle(fig, sample_id, masses)
    occ = f"_occ{b['occurrence']}" if b["occurrence"] > 1 else ""
    return _save_png_svg(
        fig, out / f"control_block{b['block']:02d}_{window_slug(key)}_{b['nu_nom_mVs']:g}mVs{occ}",
        _control_data(df, b))


def _control_data(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    """
    CSV of the control plot: every row of the evaluated cycle with the
    discharge area it is shaded in ("+" / "-" / ""), followed by the
    interpolated 0 V points (excel_row empty, point = "0 V crossing").
    """
    d = _curve_data(df, b)
    d["discharge_area"] = ""
    d["point"] = "data"
    sym = b["window_type"] == SYMMETRIC
    if sym:
        for p, name in (("+", "down"), ("-", "up")):
            row_a = b[name]["row_a"]
            if np.isfinite(row_a):
                d.loc[(d["sweep"] == name) & (d["excel_row"] <= row_a), "discharge_area"] = p
    else:
        d.loc[d["sweep"] == "down", "discharge_area"] = "+"
    extra = []
    if sym:
        for p, name in (("+", "down"), ("-", "up")):
            rec = b[name]
            if np.isfinite(rec["Q0"]):
                extra.append({**d.iloc[0][["window", "U_max_V", "nu_nom_mVs", "block",
                                           "occurrence"]].to_dict(),
                              "excel_row": np.nan, "cycle": rec["cycle"], "step": rec["step"],
                              "sweep": name, "V12_V": 0.0, "I12_mA": rec["I0"],
                              "Qcd_mAh": rec["Q0"] + rec["Qcd_first"],
                              "discharge_area": p, "point": "0 V crossing (interpolated)"})
    if extra:
        d = pd.concat([d, pd.DataFrame(extra)], ignore_index=True)
    d["not_evaluable"] = d["discharge_area"].map(
        lambda p: bool(b["pol"][p]["not_evaluable"]) if p in b["pol"] else False)
    return d


def plot_rate_capability(blocks, masses, sample_id, out: Path) -> list[Path]:
    """C_el vs nu (log x) per window; lower panel: retention in % of the slowest nu."""
    styles = window_styles(blocks)
    last = _last_occurrence(blocks)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
    rows = []
    for key, (color, marker) in styles.items():
        pts = sorted(((nu, b) for (k, nu), b in last.items() if k == key), key=lambda t: t[0])
        good = [(nu, b["C_el"]) for nu, b in pts if np.isfinite(b["C_el"])]
        omitted = [nu for nu, b in pts if not np.isfinite(b["C_el"])]
        ref = good[0][1] if good else np.nan
        rows += [dict(window=window_label(key), U_max_V=key[1], nu_nom_mVs=nu,
                      nu_meas_mVs=b["nu_meas_Vs"] * 1000.0, block=b["block"],
                      occurrence=b["occurrence"], C_F=b["C"], C_el_Fg=b["C_el"],
                      retention_pct=100.0 * b["C_el"] / ref,
                      evaluable=bool(np.isfinite(b["C_el"])))
                 for nu, b in pts]
        label = window_label(key)
        if omitted:
            label += f" (n.e. at {', '.join(f'{x:g}' for x in omitted)} mV/s omitted)"
        if not good:
            ax1.plot([], [], color=color, marker=marker, label=label)
            continue
        x, y = zip(*good)
        ax1.plot(x, y, color=color, marker=marker, markersize=8, linewidth=2, label=label)
        ax2.plot(x, 100.0 * np.asarray(y) / y[0], color=color, marker=marker, markersize=8,
                 linewidth=2)
    for ax in (ax1, ax2):
        _style_axes(ax)
        ax.set_xscale("log")
    ax1.set_ylabel("C_el (F g⁻¹, per electrode)")
    ax2.set_ylabel("Retention (% of slowest scan rate)")
    ax2.set_xlabel("Scan rate ν (mV s⁻¹)")
    ax1.set_title("Rate capability", loc="left")
    ax1.legend(title="Window", frameon=False)
    _subtitle(fig, sample_id, masses)
    return _save_png_svg(fig, out / "rate_capability", pd.DataFrame(rows))


def plot_window(blocks, masses, sample_id, out: Path) -> list[Path]:
    """C_el (left axis) and eta (right axis) vs U_max at the slowest nu; eta = 0.95 dashed."""
    rep = sorted((b for b in blocks if b["reported_for_window"]),
                 key=lambda b: (b["window_type"] != SYMMETRIC, b["U_max"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax2 = ax.twinx()
    for wtype, ls in ((SYMMETRIC, "-"), ("one-sided", "--")):
        sel = [b for b in rep if b["window_type"] == wtype]
        if not sel:
            continue
        u = [b["U_max"] for b in sel]
        tag = "±U" if wtype == SYMMETRIC else "0→U"
        ax.plot(u, [b["C_el"] for b in sel], color=INK, linestyle=ls, marker="o",
                markersize=8, linewidth=2, label=f"C_el ({tag})")
        for p, mk in (("+", "^"), ("-", "v")):
            eta = [b["pol"][p]["eta"] if p in b["pol"] else np.nan for b in sel]
            if np.all(np.isnan(eta)):
                continue
            ax2.plot(u, eta, color=POLARITY_COLORS[p], linestyle=ls, marker=mk, markersize=8,
                     linewidth=1.5, label=f"η{p} ({tag})")
    ax2.axhline(cfg.EFFICIENCY_MIN, color=MUTED, linestyle="--", linewidth=1,
                label=f"η = {cfg.EFFICIENCY_MIN}")
    _style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("U_max (V)")
    ax.set_ylabel("C_el (F g⁻¹, per electrode)")
    ax2.set_ylabel("Coulombic efficiency η (–)")
    etas = [b["pol"][p]["eta"] for b in rep for p in b["pol"] if np.isfinite(b["pol"][p]["eta"])]
    ax2.set_ylim(0, max([1.05] + etas))
    nus = sorted({b["nu_nom_mVs"] for b in rep})
    ax.set_title(f"Voltage window at the slowest scan rate ({', '.join(f'{n:g}' for n in nus)} "
                 "mV s⁻¹)", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=7)
    _subtitle(fig, sample_id, masses)
    data = pd.DataFrame([dict(
        window=window_label(window_key(b)), window_type=b["window_type"], U_max_V=b["U_max"],
        nu_nom_mVs=b["nu_nom_mVs"], block=b["block"], occurrence=b["occurrence"],
        C_F=b["C"], C_el_Fg=b["C_el"],
        eta_plus=b["pol"]["+"]["eta"],
        eta_minus=b["pol"]["-"]["eta"] if "-" in b["pol"] else np.nan,
        eta_min=cfg.EFFICIENCY_MIN) for b in rep])
    return _save_png_svg(fig, out / "window_plot", data)


def plot_normalised(df, blocks, masses, sample_id, out: Path) -> list[Path]:
    """
    Per window: 2 * I12[A] / (nu[V/s] * m_mean[g]) vs V12 for all scan rates
    (measured nu of the evaluated cycle). The factor 2 makes the plateau
    comparable to C_el.
    """
    last = _last_occurrence(blocks)
    paths = []
    for key in window_styles(blocks):
        sel = sorted(((nu, b) for (k, nu), b in last.items() if k == key), key=lambda t: t[0])
        colors = rate_colors([nu for nu, _ in sel])
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        data = []
        for nu, b in sel:
            d = _curve_data(df, b)
            d["nu_meas_mVs"] = b["nu_meas_Vs"] * 1000.0
            d["C_diff_el_Fg"] = 2.0 * d["I12_mA"] / 1000.0 / (b["nu_meas_Vs"] * masses["m_mean_g"])
            data.append(d)
            ax.plot(d["V12_V"], d["C_diff_el_Fg"], color=colors[nu], linewidth=1.4,
                    label=f"{nu:g} mV s⁻¹")
        _style_axes(ax)
        ax.set_xlabel("Cell voltage V12 (V)")
        ax.set_ylabel("Differential electrode capacitance (F g⁻¹)")
        ax.set_title(f"Normalised CV, window {window_label(key)}", loc="left")
        ax.legend(title="Scan rate", frameon=False)
        _subtitle(fig, sample_id, masses)
        paths += _save_png_svg(fig, out / f"normalised_{window_slug(key)}",
                               pd.concat(data, ignore_index=True))
    return paths


def plot_all(df: pd.DataFrame, blocks: list[dict], masses: dict, sample_id: str,
             out: Path) -> list[Path]:
    """All CV figures of lab guide 6.2 into `out`; returns the saved paths."""
    out.mkdir(parents=True, exist_ok=True)
    if not blocks:
        return []
    paths = plot_cv_per_scan_rate(df, blocks, masses, sample_id, out)
    for b in blocks:
        paths += plot_control(df, b, masses, sample_id, out)
    paths += plot_rate_capability(blocks, masses, sample_id, out)
    paths += plot_window(blocks, masses, sample_id, out)
    paths += plot_normalised(df, blocks, masses, sample_id, out)
    return paths
