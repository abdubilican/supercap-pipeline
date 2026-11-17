"""
gcd_plots.py
------------
Figures of the GCD evaluation (lab guide GCD 6.2), PNG (300 dpi) + SVG +
Origin-ready xlsx (plot_excel), in output/<sample-id>/gcd/plots/:

  gcd_profiles_time        V12 vs time in the cycle (selected cycle per rate block)
  gcd_profiles_charge      V12 vs specific charge (mAh/g per m_mean), discharge runs back
  gcd_rate_capability      C_el vs j (log x) + retention (second y-axis)
  gcd_esr                  ESR vs j (log x)
  gcd_ragone               E_sp vs P_sp (log-log), points labelled with j
  gcd_cycling_retention    retention and ESR/ESR0 vs cycle (if a cycling test is present)
  gcd_cycling_efficiency   eta_C vs cycle
  gcd_floating_retention   retention and ESR/ESR0 vs hold time (if a floating test is present)
  gcd_floating_efficiency  eta_C vs hold time
  gcd_qc_<j>               QC plot per rate block (not in the guide): IR drop and U_min crossing

One fixed colour per current density in every figure (sequential palette,
low j light -> high j dark).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import gcd_config as cfg
from .gcd_segmentation import cluster_values
from .plot_style import (
    ACCENT, INK, MUTED, Column, Curve, PlotContext, save_figure, sequential_colors, style_axes,
    subtitle,
)

PER_ELECTRODE = "per electrode, m̄ = (m1 + m2)/2"
PER_TOTAL = "per total active mass, m1 + m2 = 2 m̄"


def j_colors(js) -> callable:
    """Colour function for current densities: clusters (1 %) -> sequential palette."""
    js = np.asarray(list(js), dtype=float)
    if len(js) == 0:
        return lambda j: INK
    labels = cluster_values(js)
    reps = np.array([np.median(js[labels == k]) for k in range(labels.max() + 1)])
    cols = sequential_colors(list(reps))
    return lambda j: cols[float(reps[np.argmin(np.abs(reps - j))])]


def _label(r) -> str:
    occ = f" (occurrence {int(r['occurrence'])})" if r["occurrence"] > 1 else ""
    return f"{r['j_block_Ag']:.3g} A/g, cycle {int(r['selected_cycle'])}{occ}"


def _selected(rate: pd.DataFrame, ev: pd.DataFrame) -> list[tuple[dict, pd.Series]]:
    if rate.empty or "selected_cycle" not in rate:
        return []
    by_cycle = ev.set_index("cycle")
    return [(r, by_cycle.loc[int(r["selected_cycle"])])
            for r in rate.sort_values("j_block_Ag").to_dict("records")
            if np.isfinite(r["selected_cycle"])]


def _cycles_text(sel) -> str:
    return "; ".join(f"block {int(r['block'])}: cycle {int(r['selected_cycle'])}" for r, _ in sel)


# ─────────────────────────────────────────────────────────────────────────────
# Rate series
# ─────────────────────────────────────────────────────────────────────────────

def plot_profiles_time(df, ev, rate, ctx: PlotContext, color) -> list[Path]:
    """V12 vs time within the cycle (t = 0 at the start of the charge), selected cycles."""
    sel = _selected(rate, ev)
    if not sel:
        return []
    t, V = df["t"].to_numpy(dtype=float), df["V12"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    curves = []
    for r, c in sel:
        s = slice(int(c.ch_i0), int(c.dis_i1) + 1)
        x = t[s] - t[int(c.ch_i0)]
        ax.plot(x, V[s], color=color(r["j_block_Ag"]), linewidth=1.6, label=_label(r))
        curves.append(Curve(_label(r), Column("Time in cycle", "s", x),
                            [Column("Voltage", "V", V[s])]))
    style_axes(ax)
    ax.set_xlabel("Time from start of charge (s)")
    ax.set_ylabel("Cell voltage V12 (V)")
    ax.set_title("GCD profiles (selected cycle per current density)", loc="left")
    ax.legend(title="j (A g⁻¹, per electrode)", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, "gcd_profiles_time", curves=curves, cycles_used=_cycles_text(sel),
                       mass_basis={"x": "-", "y": "-", "legend_j": PER_ELECTRODE})


def plot_profiles_charge(df, ev, rate, ctx: PlotContext, color) -> list[Path]:
    """
    V12 vs specific charge per m_mean [mAh/g]: charge x = Qcd/m_mean, then
    (rest and) discharge x = (Q_ch + Qcd)/m_mean, so the curve runs back and
    the gap at the end shows the charge lost in the cycle.
    """
    sel = _selected(rate, ev)
    if not sel:
        return []
    Q, V = df["Qcd"].to_numpy(dtype=float), df["V12"].to_numpy(dtype=float)
    m = ctx.masses["m_mean_g"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    curves = []
    for r, c in sel:
        ch = slice(int(c.ch_i0), int(c.ch_i1) + 1)
        after = slice(int(c.ch_i1) + 1, int(c.dis_i1) + 1)
        x = np.r_[Q[ch], c.Q_ch + Q[after]] / m
        y = np.r_[V[ch], V[after]]
        ax.plot(x, y, color=color(r["j_block_Ag"]), linewidth=1.6, label=_label(r))
        curves.append(Curve(_label(r), Column("Specific charge", "mAh/g", x),
                            [Column("Voltage", "V", y)]))
    style_axes(ax)
    ax.set_xlabel("Specific charge (mAh g⁻¹, per electrode)")
    ax.set_ylabel("Cell voltage V12 (V)")
    ax.set_title("GCD profiles vs charge (gap at the end = charge lost)", loc="left")
    ax.legend(title="j (A g⁻¹, per electrode)", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, "gcd_profiles_charge", curves=curves,
                       cycles_used=_cycles_text(sel),
                       mass_basis={"x": PER_ELECTRODE, "y": "-"})


def plot_rate_capability(ev, rate, ctx: PlotContext, color) -> list[Path]:
    """
    C_el vs j (log x), markers + line; retention in % of the lowest j on the
    second y-axis. Retention = C_el / C_el(lowest j), so the right axis is an
    exact rescaling of the left one and both are read off the same curve.
    """
    sel = _selected(rate, ev)
    if not sel:
        return []
    j = np.array([r["j_block_Ag"] for r, _ in sel])
    c_el = np.array([r["C_el"] for r, _ in sel])
    ret = np.array([r.get("retention_pct", np.nan) for r, _ in sel])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(j, c_el, color=INK, linewidth=1.5, zorder=2)
    ax.scatter(j, c_el, c=[color(x) for x in j], s=64, edgecolors=INK, linewidths=0.8, zorder=3)
    for x, y, rr in zip(j, c_el, ret):
        ax.annotate(f"{rr:.1f} %", (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=INK)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Current density j (A g⁻¹, per electrode)")
    ax.set_ylabel("Specific capacitance (F g⁻¹, per electrode)")
    ax.set_ylim(bottom=0, top=np.nanmax(c_el) * 1.1)
    ok = np.isfinite(c_el) & np.isfinite(ret) & (ret > 0)
    if ok.any():
        c0 = float(c_el[ok][0] * 100.0 / ret[ok][0])       # C_el at 100 % retention
        sec = ax.secondary_yaxis("right", functions=(lambda c: 100.0 * c / c0,
                                                     lambda r: r * c0 / 100.0))
        sec.set_ylabel("Retention (% of lowest j)")
    ax.set_title("Rate capability", loc="left")
    subtitle(fig, ctx.sample_id, ctx.masses)
    curves = [Curve("rate series", Column("Current density", "A/g", j),
                    [Column("Specific capacitance C_el", "F/g", c_el, "C_el"),
                     Column("Retention", "%", ret, "retention vs lowest j")])]
    return save_figure(fig, ctx, "gcd_rate_capability", curves=curves,
                       cycles_used=_cycles_text(sel),
                       mass_basis={"x": PER_ELECTRODE, "y_left": PER_ELECTRODE, "y_right": "-"})


def plot_esr(ev, rate, ctx: PlotContext, color) -> list[Path]:
    """ESR vs j (log x)."""
    sel = _selected(rate, ev)
    if not sel:
        return []
    j = np.array([r["j_block_Ag"] for r, _ in sel])
    esr = np.array([r["ESR"] for r, _ in sel])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(j, esr, color=INK, linewidth=1.5, zorder=2)
    ax.scatter(j, esr, c=[color(x) for x in j], s=64, edgecolors=INK, linewidths=0.8, zorder=3)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Current density j (A g⁻¹, per electrode)")
    ax.set_ylabel("ESR (Ω)")
    ax.set_title("ESR from the IR drop at the start of the discharge", loc="left")
    subtitle(fig, ctx.sample_id, ctx.masses)
    curves = [Curve("ESR", Column("Current density", "A/g", j), [Column("ESR", "Ohm", esr)])]
    return save_figure(fig, ctx, "gcd_esr", curves=curves, cycles_used=_cycles_text(sel),
                       mass_basis={"x": PER_ELECTRODE, "y": "-"})


def plot_ragone(ev, rate, ctx: PlotContext, color) -> list[Path]:
    """E_sp vs P_sp, both log, one point per current density labelled with j."""
    sel = _selected(rate, ev)
    if not sel:
        return []
    j = np.array([r["j_block_Ag"] for r, _ in sel])
    E = np.array([r["E_sp"] for r, _ in sel])
    P = np.array([r["P_sp"] for r, _ in sel])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(P, E, color=MUTED, linewidth=1.0, zorder=2)
    ax.scatter(P, E, c=[color(x) for x in j], s=64, edgecolors=INK, linewidths=0.8, zorder=3)
    for p, e, jj in zip(P, E, j):
        ax.annotate(f"{jj:.3g} A/g", (p, e), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=INK)
    style_axes(ax, zero_line=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Specific power (W kg⁻¹, per total active mass)")
    ax.set_ylabel("Specific energy (Wh kg⁻¹, per total active mass)")
    ax.set_title("Ragone plot", loc="left")
    subtitle(fig, ctx.sample_id, ctx.masses)
    curves = [Curve("Ragone", Column("Specific power", "W/kg", P),
                    [Column("Specific energy", "Wh/kg", E),
                     Column("Current density", "A/g", j, "point label j (per electrode)")])]
    return save_figure(fig, ctx, "gcd_ragone", curves=curves, cycles_used=_cycles_text(sel),
                       mass_basis={"x": PER_TOTAL, "y": PER_TOTAL, "labels": PER_ELECTRODE})


def plot_qc(df, ev, r: dict, c: pd.Series, ctx: PlotContext) -> list[Path]:
    """
    QC plot of one rate block (not in the guide). Left: V12 around the current
    reversal (time relative to the start of the discharge step, symlog), with
    U_before and U1 marked and dU_IR, ESR, t_U1 annotated. Right: end of the
    discharge with rows a/b and the interpolated U_min point.
    """
    t = df["t"].to_numpy(dtype=float)
    TS = df["t_step"].to_numpy(dtype=float)
    V = df["V12"].to_numpy(dtype=float)
    rows = df["excel_row"].to_numpy()
    d0, d1, pre = int(c.dis_i0), int(c.dis_i1), int(c.dis_i0) - 1
    t_switch = t[d0] - TS[d0]
    left = slice(max(int(c.ch_i0), pre - 5), min(d1, d0 + 6) + 1)
    xl = (t[left] - t_switch) * 1000.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    a1.plot(xl, V[left], color=INK, marker="o", markersize=4, linewidth=1.2)
    xp = np.array([(t[pre] - t_switch) * 1000.0, (t[d0] - t_switch) * 1000.0])
    a1.plot(xp, [c.U_before, c.U1], linestyle="none", marker="o", markersize=9,
            color=ACCENT, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    a1.annotate("U_before", (xp[0], c.U_before), textcoords="offset points", xytext=(-8, -14),
                ha="right", fontsize=7)
    a1.annotate("U1", (xp[1], c.U1), textcoords="offset points", xytext=(-8, -4), ha="right",
                fontsize=7)
    a1.text(0.02, 0.04, f"U_before = {c.U_before:.6f} V (row {rows[pre]})\n"
                        f"U1       = {c.U1:.6f} V (row {rows[d0]})\n"
                        f"ΔU_IR = {c.dU_IR:.4f} V\nESR = {c.ESR:.3f} Ω "
                        f"(ΔI = {c.dI_A * 1000:.4g} mA)\nt_U1 = {c.t_U1_ms:.2f} ms",
            transform=a1.transAxes, fontsize=7.5, family="monospace", va="bottom")
    a1.set_xscale("symlog", linthresh=10.0)
    style_axes(a1, zero_line=False)
    a1.set_xlabel("Time from start of discharge step (ms, symlog)")
    a1.set_ylabel("Cell voltage V12 (V)")
    a1.set_title("IR drop at the current reversal", loc="left", fontsize=10)

    right = slice(max(d0, int(np.flatnonzero(rows == c.row_a)[0]) - 4)
                  if c.row_a > 0 else d0, d1 + 1)
    a2.plot(TS[right], V[right], color=INK, marker="o", markersize=4, linewidth=1.2)
    a2.axhline(c.U_min, color=MUTED, linestyle="--", linewidth=1)
    ab_x, ab_y = [], []
    if c.row_a > 0:
        for rr, name in ((c.row_a, "a"), (c.row_b, "b")):
            k = int(np.flatnonzero(rows == rr)[0])
            ab_x.append(TS[k])
            ab_y.append(V[k])
            a2.annotate(f"row {name} = {int(rr)}", (TS[k], V[k]), textcoords="offset points",
                        xytext=(6, 6) if name == "a" else (-8, -12),
                        ha="left" if name == "a" else "right", fontsize=7)
        a2.plot([c.t_dis], [c.U_min], marker="D", markersize=9, color=ACCENT,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4,
                label=f"interpolated U_min: t_dis = {c.t_dis:.2f} s, Q_dis = {c.Q_dis:.6f} mAh")
        a2.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(a2, zero_line=False)
    a2.set_xlabel("t_step (s)")
    a2.set_ylabel("Cell voltage V12 (V)")
    a2.set_title(f"End of discharge, U_min = {c.U_min:.2f} V", loc="left", fontsize=10)
    fig.suptitle(f"QC plot (not in the guide) — j = {r['j_block_Ag']:.3g} A/g, block "
                 f"{int(r['block'])}, cycle {int(r['selected_cycle'])}", x=0.01, ha="left",
                 y=0.93, fontsize=11)
    subtitle(fig, ctx.sample_id, ctx.masses)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    curves = [Curve("around the reversal", Column("Time from start of discharge", "ms", xl),
                    [Column("Voltage", "V", V[left])]),
              Curve("U_before and U1", Column("Time from start of discharge", "ms", xp),
                    [Column("Voltage", "V", [c.U_before, c.U1])]),
              Curve("end of discharge", Column("t_step", "s", TS[right]),
                    [Column("Voltage", "V", V[right])])]
    if ab_x:
        curves += [Curve("rows a and b", Column("t_step", "s", ab_x), [Column("Voltage", "V", ab_y)]),
                   Curve("interpolated U_min point", Column("t_step", "s", [c.t_dis]),
                         [Column("Voltage", "V", [c.U_min])])]
    occ = f"_occ{int(r['occurrence'])}" if r["occurrence"] > 1 else ""
    return save_figure(fig, ctx, f"gcd_qc_{r['j_block_Ag']:.3g}Ag{occ}", curves=curves,
                       cycles_used=f"block {int(r['block'])}: cycle {int(r['selected_cycle'])}",
                       mass_basis={"x": "-", "y": "-"})


# ─────────────────────────────────────────────────────────────────────────────
# Long-term tests
# ─────────────────────────────────────────────────────────────────────────────

def _longterm_retention(table, xcol, xlabel, xunit, name, title, group, label_fn, eol_col,
                        ctx: PlotContext, color) -> list[Path]:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True)
    curves = []
    for key, g in table.groupby(group):
        g = g.sort_values(xcol)
        j = float(g["j_Ag"].median()) if "j_Ag" in g else float(g["j_block_Ag"].median())
        lab = label_fn(key, j)
        col = color(j)
        a1.plot(g[xcol], g["retention_pct"], color=col, linewidth=1.4, marker="o",
                markersize=3 if len(g) > 50 else 6, label=lab)
        a2.plot(g[xcol], g["ESR_rel"], color=col, linewidth=1.4, marker="o",
                markersize=3 if len(g) > 50 else 6)
        eol = g[eol_col].iloc[0]
        if np.isfinite(eol):
            for a in (a1, a2):
                a.axvline(eol, color=col, linestyle=":", linewidth=1.2)
            a1.annotate(f"end of life {eol:.4g} {xunit}\n({g['eol_criterion'].iloc[0]})",
                        (eol, cfg.EOL_RETENTION * 100), textcoords="offset points",
                        xytext=(4, 6), fontsize=7)
        curves.append(Curve(lab, Column(xlabel, xunit, g[xcol].to_numpy()),
                            [Column("Retention", "%", g["retention_pct"].to_numpy()),
                             Column("ESR increase ESR/ESR0", "-", g["ESR_rel"].to_numpy())]))
    a1.axhline(cfg.EOL_RETENTION * 100, color=MUTED, linestyle="--", linewidth=1,
               label=f"{cfg.EOL_RETENTION:.0%}")
    a2.axhline(cfg.EOL_ESR, color=MUTED, linestyle="--", linewidth=1, label=f"{cfg.EOL_ESR:g}×")
    for a in (a1, a2):
        style_axes(a, zero_line=False)
    a1.set_ylabel("Retention C/C0 (%)")
    a2.set_ylabel("ESR / ESR0 (–)")
    a2.set_xlabel(f"{xlabel} ({xunit})")
    a1.set_title(title, loc="left")
    a1.legend(frameon=False, fontsize=7)
    a2.legend(frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, name, curves=curves,
                       mass_basis={"x": "-", "y": "-", "legend_j": PER_ELECTRODE})


def _longterm_eff(table, xcol, xlabel, xunit, name, title, group, label_fn,
                  ctx: PlotContext, color) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    curves = []
    for key, g in table.groupby(group):
        g = g.sort_values(xcol)
        j = float(g["j_Ag"].median()) if "j_Ag" in g else float(g["j_block_Ag"].median())
        lab = label_fn(key, j)
        ax.plot(g[xcol], g["eta_C"], color=color(j), linewidth=1.4, marker="o",
                markersize=3 if len(g) > 50 else 6, label=lab)
        curves.append(Curve(lab, Column(xlabel, xunit, g[xcol].to_numpy()),
                            [Column("Coulombic efficiency", "-", g["eta_C"].to_numpy())]))
    ax.axhline(cfg.ETA_C_WARN, color=MUTED, linestyle="--", linewidth=1,
               label=f"η_C = {cfg.ETA_C_WARN}")
    style_axes(ax, zero_line=False)
    ax.set_xlabel(f"{xlabel} ({xunit})")
    ax.set_ylabel("Coulombic efficiency η_C (–)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, name, curves=curves,
                       mass_basis={"x": "-", "y": "-", "legend_j": PER_ELECTRODE})


def plot_cycling(cyc_table: pd.DataFrame, ctx: PlotContext, color) -> list[Path]:
    if cyc_table.empty:
        return []
    lab = lambda b, j: f"block {b}, {j:.3g} A/g"  # noqa: E731
    return (_longterm_retention(cyc_table, "cycle_in_block", "Cycle number", "cycles",
                                "gcd_cycling_retention", "Cycling test", "block", lab,
                                "eol_cycle", ctx, color)
            + _longterm_eff(cyc_table, "cycle_in_block", "Cycle number", "cycles",
                            "gcd_cycling_efficiency", "Cycling test: coulombic efficiency",
                            "block", lab, ctx, color))


def plot_floating(fl_table: pd.DataFrame, ctx: PlotContext, color) -> list[Path]:
    if fl_table.empty:
        return []
    t = fl_table.assign(j_group=cluster_values(fl_table["j_block_Ag"].to_numpy()))
    lab = lambda k, j: f"check-ups at {j:.3g} A/g"  # noqa: E731
    return (_longterm_retention(t, "hold_time_h", "Hold time", "h", "gcd_floating_retention",
                                "Floating test", "j_group", lab, "eol_hold_time_h", ctx, color)
            + _longterm_eff(t, "hold_time_h", "Hold time", "h", "gcd_floating_efficiency",
                            "Floating test: coulombic efficiency", "j_group", lab, ctx, color))


def plot_all(df, ev, rate, cyc_table, fl_table, ctx: PlotContext) -> list[Path]:
    """All GCD figures of lab guide 6.2 (+ QC plots); returns the saved paths."""
    if ev.empty:
        return []
    color = j_colors(ev.groupby("block")["j_block_Ag"].first())
    paths = []
    paths += plot_profiles_time(df, ev, rate, ctx, color)
    paths += plot_profiles_charge(df, ev, rate, ctx, color)
    paths += plot_rate_capability(ev, rate, ctx, color)
    paths += plot_esr(ev, rate, ctx, color)
    paths += plot_ragone(ev, rate, ctx, color)
    paths += plot_cycling(cyc_table, ctx, color)
    paths += plot_floating(fl_table, ctx, color)
    for r, c in _selected(rate, ev):
        paths += plot_qc(df, ev, r, c, ctx)
    return paths
