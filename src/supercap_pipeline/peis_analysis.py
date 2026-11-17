"""
peis_analysis.py
----------------
Evaluation of one PEIS spectrum (lab guide PEIS sections 4-6), always from
the non-excluded points:

  circle_fit_arc()      high-frequency arc (f >= ARC_FMIN): R_s, R_c, f_apex, C_c
  find_knee()           first local minimum of -Z'' below f_apex
  esr_values()          Z' at the point nearest 1 kHz, Z'_knee, area-normalised values
  capacitance_values()  C_s / C' at f_min and near 0.1 Hz, C_el, f0 / tau0, phi(f_min)
  fit_tlm()             impedance.py CustomCircuit R0-p(R1,CPE1)-TLMQ0 (or R0-TLMQ0)
  reversibility()       repeated DC voltages: first vs last spectrum
  compare_cv_gcd()      comparison with the GCD / CV results of the same sample

Every function returns its values plus a list of (severity, text) flags.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import peis_config as cfg
from .flags import FLAG, INFO, WARNING

log = logging.getLogger("supercap_pipeline.peis").getChild("analysis")

PARAM_NAMES_ARC = ["R_s", "R_c", "Q_c", "alpha_c", "R_ion", "Q_dl", "alpha_dl"]
PARAM_NAMES_NO_ARC = ["R_s", "R_ion", "Q_dl", "alpha_dl"]
PARAM_UNITS = {"R_s": "Ohm", "R_c": "Ohm", "Q_c": "F s^(a-1)", "alpha_c": "-", "R_ion": "Ohm",
               "Q_dl": "F s^(a-1)", "alpha_dl": "-", "L": "H"}


def nearest(f: np.ndarray, target: float) -> int:
    """Index of the measured frequency nearest to `target` (in log scale)."""
    return int(np.argmin(np.abs(np.log10(np.asarray(f, dtype=float) / target))))


# ─────────────────────────────────────────────────────────────────────────────
# 4.2 High-frequency arc
# ─────────────────────────────────────────────────────────────────────────────

def circle_fit_arc(f, Zre, mZim) -> dict:
    """
    High-frequency arc (lab guide 4.2). Arc points: f >= ARC_FMIN (sorted from
    high to low f). An arc exists if -Z'' of the arc points has its maximum
    neither at the first nor at the last arc point. Then a circle is fitted by
    linear least squares
        x^2 + y^2 = 2a x + 2b y + c,   x = Z', y = -Z''
    centre (a, b), r = sqrt(c + a^2 + b^2), real-axis crossings a -+ sqrt(r^2 - b^2):
        R_s = left crossing, R_c = right - left;  b = depression of the arc.
    f_apex = f of the largest -Z'' among the arc points; C_c = 1 / (2 pi f_apex R_c).
    No arc: R_c = 0 and R_s = Z' where -Z'' crosses 0 (interpolated), or the
    smallest Z' if it does not cross (flag "R_s upper bound").
    """
    f, x, y = (np.asarray(v, dtype=float) for v in (f, Zre, mZim))
    order = np.argsort(-f)
    f, x, y = f[order], x[order], y[order]
    arc = f >= cfg.ARC_FMIN
    fa, xa, ya = f[arc], x[arc], y[arc]
    flags: list[tuple[str, str]] = []
    out = dict(arc=False, R_s=np.nan, R_c=0.0, a=np.nan, b=np.nan, r=np.nan, f_apex=np.nan,
               C_c=np.nan, n_arc=int(arc.sum()), R_s_upper_bound=False)
    if len(fa) >= 3:
        k = int(np.argmax(ya))
        if 0 < k < len(fa) - 1:
            A = np.column_stack([2 * xa, 2 * ya, np.ones_like(xa)])
            (a, b, c), *_ = np.linalg.lstsq(A, xa ** 2 + ya ** 2, rcond=None)
            r2 = c + a * a + b * b
            if r2 > b * b:
                h = np.sqrt(r2 - b * b)
                rs, rc = a - h, 2 * h
                out.update(arc=True, R_s=rs, R_c=rc, a=a, b=b, r=np.sqrt(r2), f_apex=fa[k],
                           C_c=1.0 / (2 * np.pi * fa[k] * rc))
                return out | {"flags": flags}
            flags.append((FLAG, "arc circle does not cross the real axis - treated as no arc"))
    # no arc: R_s where -Z'' crosses 0
    cross = np.flatnonzero((y[:-1] <= 0) & (y[1:] > 0))
    if len(cross):
        i = cross[0]
        out["R_s"] = x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    else:
        out["R_s"] = float(np.min(x))
        out["R_s_upper_bound"] = True
        flags.append((FLAG, f"R_s upper bound: -Z'' does not cross 0, smallest Z' = {out['R_s']:.4g} Ohm"))
    flags.append((INFO, "no high-frequency arc: R_c = 0, fit model without arc"))
    return out | {"flags": flags}


def find_knee(f, Zre, mZim, rows, f_apex: float) -> dict:
    """
    Knee (lab guide 4.3): the first local minimum of -Z'' (y[i] <= y[i-1] and
    y[i] <= y[i+1]) at frequencies below f_apex, points sorted from high to
    low f. NaN if there is no arc.
    """
    out = dict(f_knee=np.nan, Zre_knee=np.nan, mZim_knee=np.nan, row_knee=np.nan)
    if not np.isfinite(f_apex):
        return out
    f, x, y, rows = (np.asarray(v) for v in (f, Zre, mZim, rows))
    order = np.argsort(-f.astype(float))
    f, x, y, rows = f[order], x[order], y[order], rows[order]
    for i in range(1, len(f) - 1):
        if f[i] < f_apex and y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            return dict(f_knee=float(f[i]), Zre_knee=float(x[i]), mZim_knee=float(y[i]),
                        row_knee=int(rows[i]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.4 / 4.5
# ─────────────────────────────────────────────────────────────────────────────

def esr_values(f, Zre, rows) -> dict:
    """Z' at the measured point nearest F_ESR (1 kHz), with its frequency and row (no interpolation)."""
    k = nearest(f, cfg.F_ESR)
    return dict(Zre_1k=float(np.asarray(Zre)[k]), f_1k=float(np.asarray(f)[k]),
                row_1k=int(np.asarray(rows)[k]))


def capacitance_values(f, C_s, C_re, C_im, phi, m_mean_g: float) -> dict:
    """
    Capacitances and time constant (lab guide 4.5):
      C_s, C' at f_min (lowest measured frequency); C_s at the point nearest 0.1 Hz;
      C_el = 2 C_s(f_min) / m_mean, C_cell,sp = C_s(f_min) / (2 m_mean)  (same for C');
      f0 = f of the largest C'' (measured point), tau0 = 1 / (2 pi f0):
        maximum at one of the two lowest frequencies or log10(f0/f_min) < 0.5
        -> flag "tau0 uncertain: extend the lower frequency limit";
        C'' still rising at f_min (maximum at f_min) -> tau0 = NaN and flag;
      phi(f_min) > PHI_CAPACITIVE_DEG -> info "capacitive line tilted".
    """
    f, C_s, C_re, C_im, phi = (np.asarray(v, dtype=float) for v in (f, C_s, C_re, C_im, phi))
    order = np.argsort(f)                      # ascending: order[0] = f_min
    kmin = order[0]
    k01 = nearest(f, cfg.F_CS_REF)
    m = m_mean_g
    out = dict(f_min_used=f[kmin], C_s_fmin=C_s[kmin], C_re_fmin=C_re[kmin],
               f_Cs_01=f[k01], C_s_01=C_s[k01],
               C_el=2 * C_s[kmin] / m, C_cell_sp=C_s[kmin] / (2 * m),
               C_el_re=2 * C_re[kmin] / m, C_cell_sp_re=C_re[kmin] / (2 * m),
               phi_fmin=phi[kmin])
    flags: list[tuple[str, str]] = []
    k0 = int(np.argmax(C_im))
    f0 = f[k0]
    rank = int(np.flatnonzero(order == k0)[0])      # 0 = lowest frequency
    decades = np.log10(f0 / f[kmin])
    out.update(f0=f0, tau0=1.0 / (2 * np.pi * f0), f0_decades_above_fmin=decades)
    if rank == 0:
        out["tau0"] = np.nan
        flags.append((FLAG, "tau0 not determined: C'' still rises at f_min - extend the lower "
                            "frequency limit"))
    elif rank <= 1 or decades < cfg.TAU0_MIN_DECADES:
        flags.append((FLAG, f"tau0 uncertain: extend the lower frequency limit "
                            f"(log10(f0/f_min) = {decades:.2f} < {cfg.TAU0_MIN_DECADES})"))
    if phi[kmin] > cfg.PHI_CAPACITIVE_DEG:
        flags.append((INFO, f"capacitive line tilted: phi(f_min) = {phi[kmin]:.1f} deg > "
                            f"{cfg.PHI_CAPACITIVE_DEG:g} deg"))
    return out | {"flags": flags}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Equivalent-circuit fit
# ─────────────────────────────────────────────────────────────────────────────

def _circuit(arc: bool, include_L: bool) -> tuple[str, list[str]]:
    circ = cfg.CIRCUIT_ARC if arc else cfg.CIRCUIT_NO_ARC
    names = list(PARAM_NAMES_ARC if arc else PARAM_NAMES_NO_ARC)
    if include_L:
        circ, names = cfg.CIRCUIT_L_PREFIX + circ, ["L", *names]
    return circ, names


def fit_tlm(f, Z, graphical: dict, cap: dict, include_L: bool = False) -> dict:
    """
    Equivalent-circuit fit (lab guide 5) with impedance.py:
      CustomCircuit('R0-p(R1,CPE1)-TLMQ0').fit(f, Z, weight_by_modulus=True, bounds)
    TLMQ: Z = sqrt(R_ion Z_s) coth sqrt(R_ion / Z_s), Z_s = 1 / (Q_dl (j w)^alpha_dl).
    Parameters R_s, R_c, Q_c, alpha_c, R_ion, Q_dl, alpha_dl; no arc: R0-TLMQ0;
    inductive points kept: L0- in front. Start values from the graphical
    evaluation (R_s, R_c, Q_c = C_c, alpha_c 0.85, R_ion 1, Q_dl = C_s(f_min),
    alpha_dl 0.9), clipped into the bounds of FIT_BOUNDS.
    Derived: C_c,eff = (Q_c R_c^(1-alpha_c))^(1/alpha_c), EDR = R_s + R_c + R_ion/3,
    residuals dZ'/|Z|, dZ''/|Z| [%]; flags for uncertainty, bounds, alpha,
    residual and disagreement with the graphical R_s / R_c.
    Returns dict(success, circuit, params, conf, rel_unc, Z_fit, res_re, res_im,
    max_res, C_c_eff, EDR, model, flags); a failed fit gives success False.
    """
    from impedance.models.circuits import CustomCircuit

    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    arc = bool(graphical["arc"])
    circ, names = _circuit(arc, include_L)
    start = {"R_s": graphical["R_s"], "R_c": graphical["R_c"], "Q_c": graphical["C_c"],
             "alpha_c": cfg.FIT_START_ALPHA_C, "R_ion": cfg.FIT_START_R_ION,
             "Q_dl": cap["C_s_fmin"], "alpha_dl": cfg.FIT_START_ALPHA_DL, "L": 1e-7}
    lb = np.array([cfg.FIT_BOUNDS[n][0] for n in names], dtype=float)
    ub = np.array([cfg.FIT_BOUNDS[n][1] for n in names], dtype=float)
    span = ub - lb
    p0 = np.array([start[n] if np.isfinite(start[n]) else (l + u) / 2
                   for n, l, u in zip(names, lb, ub)], dtype=float)
    p0 = np.clip(p0, lb + 1e-6 * span, ub - 1e-6 * span)
    out = dict(success=False, circuit=circ, names=names, start=dict(zip(names, p0)), flags=[])
    try:
        model = CustomCircuit(circ, initial_guess=list(p0))
        with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(f, Z, weight_by_modulus=True, bounds=(list(lb), list(ub)))
    except Exception as exc:  # noqa: BLE001  (a failed fit must not stop the evaluation)
        out["flags"].append((FLAG, f"fit failed ({type(exc).__name__}: {exc}) - graphical "
                                   "values kept"))
        log.warning("Fit %s failed: %s", circ, exc)
        return out
    p = np.asarray(model.parameters_, dtype=float)
    conf = np.asarray(model.conf_, dtype=float) if model.conf_ is not None else np.full(len(p), np.nan)
    params, conf_d = dict(zip(names, p)), dict(zip(names, conf))
    rel = {n: abs(conf_d[n] / params[n]) if params[n] != 0 else np.inf for n in names}
    Zf = predict(model, f)
    res_re = 100.0 * (Z.real - Zf.real) / np.abs(Z)
    res_im = 100.0 * (Z.imag - Zf.imag) / np.abs(Z)
    max_res = float(np.max(np.maximum(np.abs(res_re), np.abs(res_im))))
    R_c = params.get("R_c", 0.0)
    if arc:
        C_eff = (params["Q_c"] * R_c ** (1 - params["alpha_c"])) ** (1 / params["alpha_c"])
    else:
        C_eff = np.nan
    edr = params["R_s"] + R_c + params["R_ion"] / 3.0
    out.update(success=True, params=params, conf=conf_d, rel_unc=rel, Z_fit=Zf, res_re=res_re,
               res_im=res_im, max_res=max_res, C_c_eff=C_eff, EDR=edr, model=model)

    fl = out["flags"]
    for n in names:
        if not np.isfinite(rel[n]) or rel[n] > cfg.FIT_REL_UNCERTAINTY_MAX:
            fl.append((FLAG, f"fit: relative uncertainty of {n} = {rel[n]:.0%} > "
                             f"{cfg.FIT_REL_UNCERTAINTY_MAX:.0%}"))
        lo, hi = cfg.FIT_BOUNDS[n]
        near_lo = params[n] <= (lo * (1 + cfg.FIT_BOUND_PROXIMITY) if lo > 0
                                else cfg.FIT_BOUND_PROXIMITY * hi)
        near_hi = params[n] >= hi * (1 - cfg.FIT_BOUND_PROXIMITY)
        if near_lo or near_hi:
            fl.append((FLAG, f"fit: {n} = {params[n]:.4g} within 1 % of its "
                             f"{'lower' if near_lo else 'upper'} bound"))
    for n in ("alpha_c", "alpha_dl"):
        if n in params and not (0.5 <= params[n] <= 1.0):
            fl.append((FLAG, f"fit: {n} = {params[n]:.3f} outside [0.5, 1]"))
    if max_res > cfg.FIT_MAX_RESIDUAL:
        fl.append((FLAG, f"fit: max residual {max_res:.2f} % > {cfg.FIT_MAX_RESIDUAL:g} %"))
    if np.isfinite(graphical["R_s"]) and abs(params["R_s"] - graphical["R_s"]) > cfg.FIT_RS_AGREE_OHM:
        fl.append((FLAG, f"fit and graphical R_s disagree: {params['R_s']:.4g} vs "
                         f"{graphical['R_s']:.4g} Ohm (> {cfg.FIT_RS_AGREE_OHM:g} Ohm)"))
    if arc and graphical["R_c"] > 0:
        dev = abs(R_c - graphical["R_c"]) / graphical["R_c"]
        if dev > cfg.FIT_RC_AGREE_REL:
            fl.append((FLAG, f"fit and graphical R_c disagree: {R_c:.4g} vs "
                             f"{graphical['R_c']:.4g} Ohm ({dev:.0%})"))
    return out


def predict(model, f) -> np.ndarray:
    """model.predict(f); numpy's harmless tanh overflow (tanh -> 1) in TLMQ is silenced."""
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(np.asarray(f, dtype=float))


def fit_curve(fit: dict, f_min: float, f_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit model on a dense log grid (FIT_CURVE_POINTS_PER_DECADE) for plotting."""
    n = max(2, int(np.ceil(np.log10(f_max / f_min) * cfg.FIT_CURVE_POINTS_PER_DECADE)) + 1)
    fg = np.logspace(np.log10(f_max), np.log10(f_min), n)
    return fg, predict(fit["model"], fg)


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 Reversibility, 6. Comparison
# ─────────────────────────────────────────────────────────────────────────────

def reversibility(spectra: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """
    Repeated DC voltages (lab guide 3.2): first vs last spectrum at that
    voltage, relative change of C_s(f_min), Z' (point nearest 1 kHz) and
    R_c (circle); > REVERSIBILITY_TOL -> flag. Returns (table, flags with label).
    """
    rows, flags = [], []
    for v, g in spectra.groupby("V_dc"):
        if len(g) < 2:
            continue
        a, b = g.sort_values("order").iloc[0], g.sort_values("order").iloc[-1]
        for name, col in (("C_s(f_min)", "C_s_fmin"), ("Z'(1 kHz)", "Zre_1k"), ("R_c", "R_c_circle")):
            ch = (b[col] - a[col]) / a[col] if a[col] else np.nan
            bad = np.isfinite(ch) and abs(ch) > cfg.REVERSIBILITY_TOL
            rows.append(dict(V_dc=v, first=a["label"], last=b["label"], quantity=name,
                             value_first=a[col], value_last=b[col], change_pct=100 * ch,
                             flag=bad))
            if bad:
                flags.append((FLAG, b["label"],
                              f"not reversible: {name} changed {ch:+.1%} from {a['label']} to "
                              f"{b['label']} (> {cfg.REVERSIBILITY_TOL:.0%})"))
    return pd.DataFrame(rows), flags


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except (OSError, ValueError) as exc:
        log.warning("%s (sheet %s) not readable: %s", path, sheet, exc)
        return None


def compare_cv_gcd(sample_dir: Path | None, spectra: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Comparison with GCD and CV of the same sample (lab guide 6 / section 11):
      GCD ESR at the lowest j vs Z' (nearest 1 kHz) of the spectrum with the DC
        voltage closest to the GCD U_max (within COMPARE_U_MAX_TOL_V);
        deviation (ESR_GCD - Z'_PEIS) / Z'_PEIS > COMPARE_TOL -> flag
      GCD C at the lowest j within [C_s(f_min) of the lowest-voltage spectrum,
        C_s(f_min) of the highest-voltage spectrum] widened by COMPARE_TOL
      CV C at the slowest scan rate of the window matching U_max (0->U_max,
        else +-U_max): same range check.
    Missing results are logged, never an error. Returns (table, flags).
    """
    rows, flags = [], []
    if sample_dir is None:
        return pd.DataFrame(), flags
    lo_s = spectra.sort_values(["V_dc", "order"]).iloc[0]
    hi_s = spectra.sort_values(["V_dc", "order"], ascending=[False, True]).iloc[0]
    c_lo, c_hi = sorted([lo_s["C_s_fmin"], hi_s["C_s_fmin"]])
    rng = (c_lo * (1 - cfg.COMPARE_TOL), c_hi * (1 + cfg.COMPARE_TOL))
    rng_txt = (f"C_s(f_min) {lo_s['label']} = {lo_s['C_s_fmin']:.4g} F, {hi_s['label']} = "
               f"{hi_s['C_s_fmin']:.4g} F, +-{cfg.COMPARE_TOL:.0%}")
    u_max = None

    gcd = _read_sheet(sample_dir / "gcd" / "results.xlsx", "Rate")
    if gcd is None or gcd.empty or "C (F)" not in gcd:
        log.info("No GCD results for this sample (%s) - GCD comparison skipped", sample_dir / "gcd")
    else:
        g = gcd.dropna(subset=["C (F)"]).sort_values("j_Ag (A/g)").iloc[0]
        u_max = float(g["U_max (V)"])
        d = (spectra["V_dc"] - u_max).abs()
        if d.min() <= cfg.COMPARE_U_MAX_TOL_V:
            s = spectra.loc[d.idxmin()]
            dev = (g["ESR (Ohm)"] - s["Zre_1k"]) / s["Zre_1k"]
            bad = abs(dev) > cfg.COMPARE_TOL
            text = (f"GCD ESR at {g['j_Ag (A/g)']:.3g} A/g = {g['ESR (Ohm)']:.4g} Ohm vs Z'({s['f_1k']:.4g} Hz) "
                    f"at {s['label']} = {s['Zre_1k']:.4g} Ohm: {dev:+.1%}")
            rows.append(dict(check="GCD ESR vs Z'(1 kHz)", peis=s["Zre_1k"], peis_spectrum=s["label"],
                             other=g["ESR (Ohm)"], other_source=f"GCD {g['j_Ag (A/g)']:.3g} A/g",
                             deviation_pct=100 * dev, range_low=np.nan, range_high=np.nan,
                             severity=FLAG if bad else INFO, text=text))
        else:
            text = f"GCD ESR: no spectrum at U_max = {u_max:.3g} V"
            rows.append(dict(check="GCD ESR vs Z'(1 kHz)", severity=INFO, text=text,
                             other=g["ESR (Ohm)"], other_source=f"GCD {g['j_Ag (A/g)']:.3g} A/g"))
        c = g["C (F)"]
        bad = not (rng[0] <= c <= rng[1])
        rows.append(dict(check="GCD C within C_s(f_min) range", peis=np.nan, other=c,
                         other_source=f"GCD {g['j_Ag (A/g)']:.3g} A/g", range_low=rng[0], range_high=rng[1],
                         severity=FLAG if bad else INFO,
                         text=f"GCD C at {g['j_Ag (A/g)']:.3g} A/g = {c:.4g} F "
                              f"{'outside' if bad else 'within'} {rng[0]:.4g}-{rng[1]:.4g} F ({rng_txt})"))

    cv = _read_sheet(sample_dir / "cv" / "results.xlsx", "Summary")
    if cv is None or cv.empty:
        log.info("No CV results for this sample (%s) - CV comparison skipped", sample_dir / "cv")
    else:
        u = round(u_max if u_max is not None else float(spectra["V_dc"].max()), 2)
        match = cv[(cv["U_max_V"] - u).abs() <= 0.005]
        one = match[match["window_type"] == "one-sided"]
        sel, wtxt = (one, "0->U_max") if len(one) else (match[match["window_type"] == "symmetric"], "+-U_max")
        sel = sel.dropna(subset=["C_F"])
        if sel.empty:
            log.info("CV results without an evaluable window at U_max = %.2f V", u)
        else:
            r = sel[sel["nu_nom_mVs"] == sel["nu_nom_mVs"].min()].iloc[-1]
            c = r["C_F"]
            bad = not (rng[0] <= c <= rng[1])
            rows.append(dict(check="CV C within C_s(f_min) range", other=c,
                             other_source=f"CV {wtxt} = {u:g} V, {r['nu_nom_mVs']:g} mV/s",
                             range_low=rng[0], range_high=rng[1], severity=FLAG if bad else INFO,
                             text=f"CV C ({wtxt} = {u:g} V, {r['nu_nom_mVs']:g} mV/s) = {c:.4g} F "
                                  f"{'outside' if bad else 'within'} {rng[0]:.4g}-{rng[1]:.4g} F "
                                  f"({rng_txt})"))
    table = pd.DataFrame(rows, columns=["check", "peis", "peis_spectrum", "other", "other_source",
                                        "deviation_pct", "range_low", "range_high", "severity", "text"])
    for r in rows:
        flags.append((r["severity"], r["text"]))
    return table, flags
