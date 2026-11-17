"""
peis_plots.py
-------------
Figures of the PEIS evaluation (lab guide PEIS 7.2), PNG (300 dpi) + SVG +
Origin-ready xlsx (plot_excel), in output/<sample-id>/peis/plots/.

Overlays (all spectra): peis_nyquist, peis_nyquist_hf, peis_bode,
peis_complex_capacitance, peis_kk_residuals, peis_cs_vs_f,
peis_parameters_vs_voltage, peis_nyquist_area, peis_complex_capacitance_specific.
Per spectrum: peis_fit_nyquist_<label>, peis_fit_bode_<label>, peis_kk_<label>.

Colour per label by DC voltage (sequential), repeated voltages: same hue,
other marker / line style. Excluded points are drawn as hollow markers.
Nyquist plots use equal axis scaling and mark the measured points nearest
10 kHz, 1 kHz, 100 Hz, 1 Hz, 0.1 Hz and 10 mHz (labelled on the first
spectrum of the plot).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import peis_config as cfg
from .peis_analysis import fit_curve, nearest
from .plot_style import (
    INK, MUTED, WINDOW_PALETTE, Column, Curve, PlotContext, save_figure, style_axes, subtitle,
)

PER_ELECTRODE = "per electrode, m̄ = (m1 + m2)/2"


def _fmt_f(f: float) -> str:
    for scale, unit in ((1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz"), (1e-3, "mHz")):
        if f >= scale * 0.999:
            return f"{f / scale:g} {unit}"
    return f"{f * 1e3:g} mHz"


def _split(pts: pd.DataFrame, s) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pts[pts["spectrum"] == s.spectrum]
    return p[~p["excluded"]], p[p["excluded"]]


def _draw(ax, x_in, y_in, x_ex, y_ex, s, label=None, line=True, ms=4.0):
    face = s.color if s.filled else "white"
    ax.plot(x_in, y_in, color=s.color, linestyle=s.linestyle if line else "none",
            marker=s.marker, markersize=ms, markerfacecolor=face, markeredgecolor=s.color,
            linewidth=1.2, label=label)
    if len(x_ex):
        ax.plot(x_ex, y_ex, linestyle="none", marker=s.marker, markersize=ms + 1,
                markerfacecolor="none", markeredgecolor=s.color, markeredgewidth=0.9)


def _meta(spectra: pd.DataFrame) -> str:
    return "; ".join(f"{r.label} (step {r.step}, rows {r.row_start}-{r.row_end})"
                     for r in spectra.itertuples())


def _mark_frequencies(ax, inc: pd.DataFrame, xcol: str, ycol: str, scale: float = 1.0,
                      span: float | None = None):
    """
    Mark the measured points nearest NYQUIST_MARK_FREQS (where measured) and
    label them; a label is skipped (marker kept) if its point lies within 10 %
    of `span` (default: the data extent) of an already labelled point.
    """
    xs, ys, fs = [], [], []
    f = inc["f"].to_numpy()
    if span is None:
        span = max(np.ptp(inc[xcol]), np.ptp(inc[ycol])) * scale or 1.0
    labelled: list[tuple[float, float]] = []
    for target in cfg.NYQUIST_MARK_FREQS:
        if not len(f):
            break
        k = nearest(f, target)
        if abs(np.log10(f[k] / target)) <= cfg.NYQUIST_MARK_TOL_DECADES:
            x, y = inc[xcol].iloc[k] * scale, inc[ycol].iloc[k] * scale
            xs.append(x), ys.append(y), fs.append(f[k])
            ax.plot([x], [y], marker="o", markersize=6, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=0.8, zorder=5)
            if all(np.hypot(x - lx, y - ly) > 0.10 * span for lx, ly in labelled):
                ax.annotate(_fmt_f(target), (x, y), textcoords="offset points", xytext=(5, -9),
                            fontsize=6.5, color=INK)
                labelled.append((x, y))
    return xs, ys, fs


# ─────────────────────────────────────────────────────────────────────────────
# Overlays
# ─────────────────────────────────────────────────────────────────────────────

def _nyquist(pts, spectra, ctx, name, scale=1.0, unit="Ω", unit_x="Ohm", zoom=None, basis=None):
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    curves = []
    for s in spectra.itertuples():
        inc, exc = _split(pts, s)
        xi, yi, xe, ye = inc["Zre"] * scale, inc["mZim"] * scale, exc["Zre"] * scale, exc["mZim"] * scale
        if zoom is not None:
            keep_i = (xi <= zoom) & (yi <= zoom)
            keep_e = (xe <= zoom) & (ye <= zoom)
            xi, yi, xe, ye = xi[keep_i], yi[keep_i], xe[keep_e], ye[keep_e]
        _draw(ax, xi, yi, xe, ye, s, label=s.label)
        curves.append(Curve(s.label, Column("Z'", unit_x, xi.to_numpy()),
                            [Column("-Z''", unit_x, yi.to_numpy())]))
        if len(xe):
            curves.append(Curve(f"{s.label} excluded", Column("Z'", unit_x, xe.to_numpy()),
                                [Column("-Z''", unit_x, ye.to_numpy())]))
    first = spectra.iloc[0]
    inc0, _ = _split(pts, first)
    if zoom is not None:
        inc0 = inc0[(inc0["Zre"] * scale <= zoom) & (inc0["mZim"] * scale <= zoom)]
    mx, my, mf = _mark_frequencies(ax, inc0, "Zre", "mZim", scale)
    if mx:
        curves.append(Curve(f"frequency marks ({first.label})", Column("Z'", unit_x, mx),
                            [Column("-Z''", unit_x, my), Column("Frequency", "Hz", mf)]))
    style_axes(ax)
    ax.axvline(0, color=MUTED, linewidth=0.6)
    if zoom is not None:
        ax.set_xlim(0, zoom)
        ax.set_ylim(min(0.0, -0.02 * zoom), zoom)
    ax.set_aspect("equal", adjustable="datalim" if zoom is None else "box")
    ax.set_xlabel(f"Z' ({unit})")
    ax.set_ylabel(f"−Z'' ({unit})")
    title = "Nyquist plot" + (" — high-frequency part" if zoom is not None else "")
    ax.set_title(title + (" (area-normalised)" if scale != 1.0 else ""), loc="left")
    ax.legend(title="DC voltage", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, name, curves=curves, cycles_used=_meta(spectra),
                       mass_basis=basis or {"x": "absolute", "y": "absolute"})


def plot_nyquist(pts, spectra, ctx):
    return _nyquist(pts, spectra, ctx, "peis_nyquist")


def plot_nyquist_hf(pts, spectra, ctx):
    knee = spectra["Zre_knee"].max()
    ref = knee if np.isfinite(knee) else spectra["Zre_1k"].max()
    return _nyquist(pts, spectra, ctx, "peis_nyquist_hf", zoom=cfg.NYQUIST_HF_ZOOM * ref)


def plot_nyquist_area(pts, spectra, ctx, area_cm2):
    return _nyquist(pts, spectra, ctx, "peis_nyquist_area", scale=area_cm2, unit="Ω·cm²",
                    unit_x="Ohm cm2", basis={"x": f"x electrode area A = {area_cm2:.4g} cm2",
                                             "y": f"x electrode area A = {area_cm2:.4g} cm2"})


def plot_bode(pts, spectra, ctx):
    """|Z| (log) and phi vs f (log), two panels sharing the x-axis."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    curves = []
    for s in spectra.itertuples():
        inc, exc = _split(pts, s)
        _draw(a1, inc["f"], inc["Zabs"], exc["f"], exc["Zabs"], s, label=s.label)
        _draw(a2, inc["f"], inc["phi"], exc["f"], exc["phi"], s)
        curves.append(Curve(s.label, Column("Frequency", "Hz", inc["f"].to_numpy()),
                            [Column("|Z|", "Ohm", inc["Zabs"].to_numpy()),
                             Column("Phase", "deg", inc["phi"].to_numpy())]))
        if len(exc):
            curves.append(Curve(f"{s.label} excluded", Column("Frequency", "Hz", exc["f"].to_numpy()),
                                [Column("|Z|", "Ohm", exc["Zabs"].to_numpy()),
                                 Column("Phase", "deg", exc["phi"].to_numpy())]))
    for a in (a1, a2):
        style_axes(a, zero_line=False)
        a.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylabel("|Z| (Ω)")
    a2.set_ylabel("Phase φ (°)")
    a2.set_xlabel("Frequency f (Hz)")
    a1.set_title("Bode plot", loc="left")
    a1.legend(title="DC voltage", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, "peis_bode", curves=curves, cycles_used=_meta(spectra),
                       mass_basis={"x": "-", "y": "absolute"})


def _complex_cap(pts, spectra, ctx, name, factor=1.0, unit="F", unit_x="F", basis="absolute"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    curves = []
    f0x, f0y, f0lab = [], [], []
    for s in spectra.itertuples():
        inc, exc = _split(pts, s)
        cr, ci = inc["C_re"] * factor, inc["C_im"] * factor
        _draw(a1, inc["f"], cr, exc["f"], exc["C_re"] * factor, s, label=s.label)
        _draw(a2, inc["f"], ci, exc["f"], exc["C_im"] * factor, s)
        curves.append(Curve(s.label, Column("Frequency", "Hz", inc["f"].to_numpy()),
                            [Column("C'", unit_x, cr.to_numpy()), Column("C''", unit_x, ci.to_numpy())]))
        if len(exc):
            curves.append(Curve(f"{s.label} excluded", Column("Frequency", "Hz", exc["f"].to_numpy()),
                                [Column("C'", unit_x, (exc["C_re"] * factor).to_numpy()),
                                 Column("C''", unit_x, (exc["C_im"] * factor).to_numpy())]))
        if np.isfinite(s.f0):
            k = nearest(inc["f"].to_numpy(), s.f0)
            a2.plot([s.f0], [ci.iloc[k]], marker="*", markersize=12, color=s.color,
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=5)
            a2.annotate(f"f₀ {_fmt_f(s.f0)}", (s.f0, ci.iloc[k]), textcoords="offset points",
                        xytext=(6, 4), fontsize=6.5)
            f0x.append(s.f0), f0y.append(ci.iloc[k]), f0lab.append(s.tau0)
    if f0x:
        curves.append(Curve("f0 (maximum of C'')", Column("Frequency", "Hz", f0x),
                            [Column("C''", unit_x, f0y), Column("tau0", "s", f0lab)]))
    for a in (a1, a2):
        style_axes(a, zero_line=False)
        a.set_xscale("log")
    lab = "2C'/m̄" if factor != 1.0 else "C'"
    a1.set_ylabel(f"{lab} ({unit})")
    a2.set_ylabel(f"{lab.replace(chr(39), chr(39) * 2)} ({unit})")
    a2.set_xlabel("Frequency f (Hz)")
    a1.set_title("Complex capacitance" + (" (specific, per electrode)" if factor != 1.0 else ""),
                 loc="left")
    a1.legend(title="DC voltage", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, name, curves=curves, cycles_used=_meta(spectra),
                       mass_basis={"x": "-", "y": basis})


def plot_complex_capacitance(pts, spectra, ctx):
    return _complex_cap(pts, spectra, ctx, "peis_complex_capacitance")


def plot_complex_capacitance_specific(pts, spectra, ctx):
    m = ctx.masses["m_mean_g"]
    return _complex_cap(pts, spectra, ctx, "peis_complex_capacitance_specific", factor=2.0 / m,
                        unit="F g⁻¹, per electrode", unit_x="F/g", basis=f"2C/m̄, {PER_ELECTRODE}")


def plot_cs_vs_f(pts, spectra, ctx):
    """C_s vs f (log x, log y): does C_s still rise at f_min?"""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    curves = []
    for s in spectra.itertuples():
        inc, exc = _split(pts, s)
        _draw(ax, inc["f"], inc["C_s"], exc["f"], exc["C_s"], s, label=s.label)
        curves.append(Curve(s.label, Column("Frequency", "Hz", inc["f"].to_numpy()),
                            [Column("Series capacitance C_s", "F", inc["C_s"].to_numpy())]))
        if len(exc):
            curves.append(Curve(f"{s.label} excluded", Column("Frequency", "Hz", exc["f"].to_numpy()),
                                [Column("Series capacitance C_s", "F", exc["C_s"].to_numpy())]))
    style_axes(ax, zero_line=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency f (Hz)")
    ax.set_ylabel("C_s = −1/(ωZ'') (F)")
    ax.set_title("Series capacitance", loc="left")
    ax.legend(title="DC voltage", frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, "peis_cs_vs_f", curves=curves, cycles_used=_meta(spectra),
                       mass_basis={"x": "-", "y": "absolute (cell)"})


def plot_kk_residuals(pts, spectra, ctx):
    """One panel per spectrum: Lin-KK residuals in % vs f, dashed at +-KK_WARN and +-KK_FLAG."""
    n = len(spectra)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.9 * n + 0.8), sharex=True, squeeze=False)
    curves = []
    for a, s in zip(axes[:, 0], spectra.itertuples()):
        curves += _kk_panel(a, pts, s)
    axes[-1, 0].set_xlabel("Frequency f (Hz)")
    axes[0, 0].set_title("Lin-KK residuals", loc="left")
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, "peis_kk_residuals", curves=curves, cycles_used=_meta(spectra),
                       mass_basis={"x": "-", "y": "relative to |Z|"})


def _kk_panel(a, pts, s) -> list[Curve]:
    inc, _ = _split(pts, s)
    a.plot(inc["f"], inc["kk_res_re"], color=s.color, marker="o", markersize=3, linewidth=1,
           label="ΔZ'/|Z|")
    a.plot(inc["f"], inc["kk_res_im"], color=s.color, marker="s", markersize=3, linewidth=1,
           linestyle="--", markerfacecolor="white", label="ΔZ''/|Z|")
    for lvl, ls in ((cfg.KK_WARN, "--"), (cfg.KK_FLAG, ":")):
        for sgn in (1, -1):
            a.axhline(sgn * lvl, color=MUTED, linestyle=ls, linewidth=0.8)
    style_axes(a)
    a.set_xscale("log")
    lim = max(2.5, np.nanmax(np.abs(inc[["kk_res_re", "kk_res_im"]].to_numpy())) * 1.15)
    a.set_ylim(-lim, lim)
    a.set_ylabel(f"{s.label}\nresidual (%)", fontsize=8)
    a.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    return [Curve(f"{s.label} KK residuals", Column("Frequency", "Hz", inc["f"].to_numpy()),
                  [Column("dZ'/|Z|", "%", inc["kk_res_re"].to_numpy()),
                   Column("dZ''/|Z|", "%", inc["kk_res_im"].to_numpy())])]


def plot_parameters_vs_voltage(spectra, ctx):
    """
    x = DC voltage; (a) R_s, R_c, R_ion, EDR (fit) and Z'(1 kHz); (b) C_s(f_min),
    C'(f_min); (c) tau0; (d) phi(f_min). Repeated voltages get their own
    marker and the order index next to the point; points of one quantity are
    joined in measurement order by a thin grey line (path 0 V -> ... -> 0 V).
    """
    s = spectra.sort_values("order")
    panels = [
        ("(a) Resistances", "Resistance (Ω)", "Ohm",
         [("R_s (fit)", "fit_R_s"), ("R_c (fit)", "fit_R_c"), ("R_ion (fit)", "fit_R_ion"),
          ("EDR", "EDR"), ("Z'(1 kHz)", "Zre_1k")]),
        ("(b) Capacitance at f_min", "Capacitance (F)", "F",
         [("C_s(f_min)", "C_s_fmin"), ("C'(f_min)", "C_re_fmin")]),
        ("(c) Time constant", "τ₀ (s)", "s", [("tau0", "tau0")]),
        ("(d) Phase at f_min", "φ(f_min) (°)", "deg", [("phi(f_min)", "phi_fmin")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    curves = []
    for ax, (title, ylab, unit, qs) in zip(axes.ravel(), panels):
        for qi, (qlab, col) in enumerate(qs):
            if col not in s:
                continue
            x, y = s["V_dc"].to_numpy(), s[col].to_numpy(dtype=float)
            color = WINDOW_PALETTE[qi % len(WINDOW_PALETTE)]
            ax.plot(x, y, color=MUTED, linewidth=0.7, zorder=1)
            for r, yy in zip(s.itertuples(), y):
                ax.plot([r.V_dc], [yy], linestyle="none", marker=r.marker, markersize=7,
                        color=color, markerfacecolor=color if r.filled else "white",
                        markeredgecolor=color, zorder=3)
                if r.n_at_voltage > 1:
                    off = [(6, 3), (6, -10), (-12, 3), (-12, -10)][(r.occurrence - 1) % 4]
                    ax.annotate(str(r.order), (r.V_dc, yy), textcoords="offset points",
                                xytext=off, fontsize=7, color=INK)
            ax.plot([], [], color=color, marker="o", linestyle="none", label=qlab)
            curves.append(Curve(qlab, Column("DC voltage", "V", x),
                                [Column(qlab, unit, y), Column("Order index", "-", s["order"].to_numpy())]))
        style_axes(ax, zero_line=False)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("DC voltage (V)")
        ax.set_ylabel(ylab)
        if len(qs) > 1:
            ax.legend(frameon=False, fontsize=7)
    note = ("Markers: filled = first, open = repeated measurement at that voltage; numbers = "
            "order index; grey line = measurement order")
    fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    subtitle(fig, ctx.sample_id, ctx.masses)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return save_figure(fig, ctx, "peis_parameters_vs_voltage", curves=curves,
                       cycles_used=_meta(spectra),
                       mass_basis={"x": "-", "y": "absolute (cell); see the Spectra sheet for "
                                                    "area-normalised and specific values"})


# ─────────────────────────────────────────────────────────────────────────────
# Per spectrum
# ─────────────────────────────────────────────────────────────────────────────

def _fit_text(s, fit: dict) -> str:
    if not fit.get("success"):
        return "fit failed - graphical values only"
    lines = [fit["circuit"]]
    for n in fit["names"]:
        v, c, r = fit["params"][n], fit["conf"][n], fit["rel_unc"][n]
        lines.append(f"{n:8s} = {v:.4g} ± {c:.2g} ({r:.1%})")
    lines.append(f"EDR      = {fit['EDR']:.4g} Ω")
    if np.isfinite(fit["C_c_eff"]):
        lines.append(f"C_c,eff  = {fit['C_c_eff'] * 1e6:.3g} µF")
    lines.append(f"max res. = {fit['max_res']:.2f} %")
    return "\n".join(lines)


def plot_fit_nyquist(pts, s, fit, ctx):
    """
    Measured points (excluded hollow) and fit curve in the Nyquist plane:
    full spectrum (left) and the high-frequency part up to NYQUIST_HF_ZOOM x
    Z'_knee (right, same data), both with equal axis scaling; fit residuals
    below; fit parameters with uncertainties in a text box.
    """
    inc, exc = _split(pts, s)
    fig = plt.figure(figsize=(10.0, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], width_ratios=[1.0, 1.0],
                          hspace=0.45, wspace=0.28)
    a1 = fig.add_subplot(gs[0:2, 0])
    az = fig.add_subplot(gs[0, 1])
    at = fig.add_subplot(gs[1, 1])
    a2 = fig.add_subplot(gs[2, :])
    curves = [Curve(f"{s.label} measured", Column("Z'", "Ohm", inc["Zre"].to_numpy()),
                    [Column("-Z''", "Ohm", inc["mZim"].to_numpy())])]
    if len(exc):
        curves.append(Curve(f"{s.label} excluded", Column("Z'", "Ohm", exc["Zre"].to_numpy()),
                            [Column("-Z''", "Ohm", exc["mZim"].to_numpy())]))
    zg = None
    if fit.get("success"):
        fg, zg = fit_curve(fit, inc["f"].min(), inc["f"].max())
        curves.append(Curve(f"{s.label} fit", Column("Z'", "Ohm", zg.real),
                            [Column("-Z''", "Ohm", -zg.imag), Column("Frequency", "Hz", fg)]))
    ref = s.Zre_knee if np.isfinite(s.Zre_knee) else s.Zre_1k
    zoom = cfg.NYQUIST_HF_ZOOM * ref
    for ax, lim in ((a1, None), (az, zoom)):
        _draw(ax, inc["Zre"], inc["mZim"], exc["Zre"], exc["mZim"], s,
              label=f"{s.label} measured", line=False, ms=5 if lim is None else 4)
        if zg is not None:
            ax.plot(zg.real, -zg.imag, color=INK, linewidth=1.3, label=f"{s.label} fit")
        style_axes(ax)
        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("−Z'' (Ω)")
        if lim is not None:
            ax.set_xlim(0, lim)
            ax.set_ylim(min(0.0, -0.02 * lim), lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("High-frequency part", loc="left", fontsize=9)
            sub = inc[(inc["Zre"] <= lim) & (inc["mZim"] <= lim)]
            _mark_frequencies(ax, sub, "Zre", "mZim", span=lim)
        else:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"Fit {s.label}", loc="left")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            mx, my, mf = _mark_frequencies(ax, inc, "Zre", "mZim")
    if mx:
        curves.append(Curve("frequency marks", Column("Z'", "Ohm", mx),
                            [Column("-Z''", "Ohm", my), Column("Frequency", "Hz", mf)]))
    at.axis("off")
    at.text(0.0, 1.0, _fit_text(s, fit), transform=at.transAxes, ha="left", va="top",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=MUTED, linewidth=0.6))
    if fit.get("success"):
        a2.plot(inc["f"], inc["fit_res_re"], color=s.color, marker="o", markersize=3, label="ΔZ'/|Z|")
        a2.plot(inc["f"], inc["fit_res_im"], color=s.color, marker="s", markersize=3, linestyle="--",
                markerfacecolor="white", label="ΔZ''/|Z|")
        curves.append(Curve(f"{s.label} fit residuals", Column("Frequency", "Hz", inc["f"].to_numpy()),
                            [Column("dZ'/|Z|", "%", inc["fit_res_re"].to_numpy()),
                             Column("dZ''/|Z|", "%", inc["fit_res_im"].to_numpy())]))
    style_axes(a2)
    a2.set_xscale("log")
    a2.set_xlabel("Frequency f (Hz)")
    a2.set_ylabel("Fit residual (%)")
    if fit.get("success"):
        a2.legend(frameon=False, fontsize=7, ncol=2)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, f"peis_fit_nyquist_{s.slug}", curves=curves,
                       cycles_used=_meta(pd.DataFrame([s._asdict()])),
                       mass_basis={"x": "absolute", "y": "absolute"})


def plot_fit_bode(pts, s, fit, ctx):
    inc, exc = _split(pts, s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    _draw(a1, inc["f"], inc["Zabs"], exc["f"], exc["Zabs"], s, label=f"{s.label} measured", line=False)
    _draw(a2, inc["f"], inc["phi"], exc["f"], exc["phi"], s, line=False)
    curves = [Curve(f"{s.label} measured", Column("Frequency", "Hz", inc["f"].to_numpy()),
                    [Column("|Z|", "Ohm", inc["Zabs"].to_numpy()),
                     Column("Phase", "deg", inc["phi"].to_numpy())])]
    if len(exc):
        curves.append(Curve(f"{s.label} excluded", Column("Frequency", "Hz", exc["f"].to_numpy()),
                            [Column("|Z|", "Ohm", exc["Zabs"].to_numpy()),
                             Column("Phase", "deg", exc["phi"].to_numpy())]))
    if fit.get("success"):
        fg, zg = fit_curve(fit, inc["f"].min(), inc["f"].max())
        a1.plot(fg, np.abs(zg), color=INK, linewidth=1.3, label=f"{s.label} fit")
        a2.plot(fg, np.degrees(np.angle(zg)), color=INK, linewidth=1.3)
        curves.append(Curve(f"{s.label} fit", Column("Frequency", "Hz", fg),
                            [Column("|Z|", "Ohm", np.abs(zg)),
                             Column("Phase", "deg", np.degrees(np.angle(zg)))]))
    for a in (a1, a2):
        style_axes(a, zero_line=False)
        a.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylabel("|Z| (Ω)")
    a2.set_ylabel("Phase φ (°)")
    a2.set_xlabel("Frequency f (Hz)")
    a1.set_title(f"Fit {s.label} (Bode)", loc="left")
    a1.legend(frameon=False, fontsize=7)
    subtitle(fig, ctx.sample_id, ctx.masses)
    return save_figure(fig, ctx, f"peis_fit_bode_{s.slug}", curves=curves,
                       cycles_used=_meta(pd.DataFrame([s._asdict()])),
                       mass_basis={"x": "-", "y": "absolute"})


def plot_kk_single(pts, s, ctx):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    curves = _kk_panel(ax, pts, s)
    ax.set_xlabel("Frequency f (Hz)")
    ax.set_title(f"Lin-KK residuals {s.label} (M = {s.kk_M}, µ = {s.kk_mu:.2f})", loc="left")
    subtitle(fig, ctx.sample_id, ctx.masses)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, ctx, f"peis_kk_{s.slug}", curves=curves,
                       cycles_used=_meta(pd.DataFrame([s._asdict()])),
                       mass_basis={"x": "-", "y": "relative to |Z|"})


def plot_all(pts: pd.DataFrame, spectra: pd.DataFrame, fits: dict, ctx: PlotContext,
             area_cm2: float) -> list[Path]:
    """All PEIS figures of lab guide 7.2; returns the saved paths."""
    if spectra.empty:
        return []
    spectra = spectra.sort_values("order")
    paths = []
    paths += plot_nyquist(pts, spectra, ctx)
    paths += plot_nyquist_hf(pts, spectra, ctx)
    paths += plot_bode(pts, spectra, ctx)
    paths += plot_complex_capacitance(pts, spectra, ctx)
    paths += plot_kk_residuals(pts, spectra, ctx)
    paths += plot_cs_vs_f(pts, spectra, ctx)
    paths += plot_parameters_vs_voltage(spectra, ctx)
    paths += plot_nyquist_area(pts, spectra, ctx, area_cm2)
    paths += plot_complex_capacitance_specific(pts, spectra, ctx)
    for s in spectra.itertuples():
        fit = fits.get(s.spectrum, {})
        paths += plot_fit_nyquist(pts, s, fit, ctx)
        paths += plot_fit_bode(pts, s, fit, ctx)
        paths += plot_kk_single(pts, s, ctx)
    return paths
