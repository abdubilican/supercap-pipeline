"""
tests/gcd_synth.py
------------------
Synthetic PAT-Tester GCD exports of an ideal capacitor C with series
resistance R (ESR) and parallel leakage resistance R_leak, in the export
column format (padded names, units row, "CC " with trailing space, signed
Qcd reset per step, cumulative W12).

Exact solutions per step (V_C = capacitor voltage, tau = R_leak * C):
  constant current I [A]:  V_C(t) = I R_leak + (V_C0 - I R_leak) exp(-t/tau),  V = V_C + I R
                           Q = I t,  W = I^2 R t + I [I R_leak t + (V_C0 - I R_leak) tau (1 - e)]
  rest (I = 0):            V = V_C = V_C0 exp(-t/tau)
  voltage hold at U:       V_C -> U R_leak/(R + R_leak) with tau_h = C R R_leak/(R + R_leak),
                           I = (U - V_C)/R (decays to the leakage current)
A CC step ends when V reaches the limit; the last logged point lies a little
beyond it (like the tester), so U_min has to be interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["t", "t_cycle", "t_step", "cycle", "step", "type", "c/d", "con", "V12", "I12",
           "|Qstep|", "Qcd", "|Qcd|", "Qc", "Qd", "P", "W12"]
UNITS = ["s", "s", "s", "", "", "", "", "", "V", "mA", "mAh", "mAh", "mAh", "mAh", "mAh",
         "mW", "mWh"]
FIRST_DT = 0.00143  # s: first logged point after a step change (as in the PAT-Tester files)


@dataclass
class GcdCell:
    C: float = 1.0           # F
    R: float = 1.0           # Ohm (ESR)
    R_leak: float = 5000.0   # Ohm
    pts: int = 60            # logged points per CC step
    t: float = 0.0
    Vc: float = 0.0
    W: float = 0.0           # mWh, cumulative
    cycle: int = 0
    step_no: int = 0
    rows: list = field(default_factory=list)

    def _emit(self, ts, V, I_mA, Q_mAh, W_mWh, typ, cd, con):
        self.step_no += 10
        for k in range(len(ts)):
            q = Q_mAh[k]
            self.rows.append([self.t + ts[k], 0.0, ts[k], self.cycle, self.step_no, typ + " ",
                              cd + " ", con + " ", V[k], I_mA[k], abs(q), q, abs(q),
                              max(q, 0.0), max(-q, 0.0), V[k] * I_mA[k], W_mWh[k]])
        self.t += ts[-1]

    def cc(self, I_mA: float, v_limit: float) -> None:
        """Constant current until V12 reaches v_limit (charge: I > 0, discharge: I < 0)."""
        I = I_mA / 1000.0
        tau = self.R_leak * self.C
        target_vc = v_limit - I * self.R
        arg = (target_vc - I * self.R_leak) / (self.Vc - I * self.R_leak)
        if not 0 < arg < 1:
            raise ValueError(f"limit {v_limit} V not reachable with {I_mA} mA")
        T = -tau * np.log(arg)
        ts = np.r_[FIRST_DT, np.linspace(T / self.pts, T * (1 + 0.3 / self.pts), self.pts)]
        e = np.exp(-ts / tau)
        vc = I * self.R_leak + (self.Vc - I * self.R_leak) * e
        V = vc + I * self.R
        Q = I * ts / 3.6                                                    # mAh
        W = (I * I * self.R * ts
             + I * (I * self.R_leak * ts + (self.Vc - I * self.R_leak) * tau * (1 - e))) / 3.6
        self._emit(ts, V, np.full_like(ts, I_mA), Q, self.W + W, "CC",
                   "c" if I > 0 else "d", "i12")
        self.Vc, self.W = float(vc[-1]), float(self.W + W[-1])

    def rest(self, T: float, n: int = 6) -> None:
        ts = np.r_[FIRST_DT, np.linspace(T / n, T, n)]
        vc = self.Vc * np.exp(-ts / (self.R_leak * self.C))
        z = np.zeros_like(ts)
        self._emit(ts, vc, z, z, np.full_like(ts, self.W), "Rest", "-", "Rest")
        self.Vc = float(vc[-1])

    def hold(self, U: float, T: float, every: float = 300.0) -> None:
        """Voltage hold (floating) at U for T seconds."""
        Rp = self.R * self.R_leak / (self.R + self.R_leak)
        tau = self.C * Rp
        vinf = U * self.R_leak / (self.R + self.R_leak)
        ts = np.unique(np.r_[FIRST_DT, 1.0, 10.0, 60.0, np.arange(every, T + every / 2, every)])
        e = np.exp(-ts / tau)
        vc = vinf + (self.Vc - vinf) * e
        I = (U - vc) / self.R                                               # A
        Q = ((U - vinf) * ts - (self.Vc - vinf) * tau * (1 - e)) / self.R   # C
        self._emit(ts, np.full_like(ts, U), I * 1000.0, Q / 3.6, self.W + U * Q / 3.6,
                   "CV", "c", "i12_V12")
        self.W += U * Q[-1] / 3.6
        self.Vc = float(vc[-1])

    def cycle_(self, I_mA: float, u_max: float, u_min: float, rest_s: float = 0.0) -> None:
        """One GCD cycle: charge to u_max, optional rest, discharge to u_min."""
        self.cycle += 1
        self.cc(I_mA, u_max)
        if rest_s:
            self.rest(rest_s)
        self.cc(-I_mA, u_min)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


def rate_program(cell: GcdCell, currents_mA: list[float], n_cycles: int,
                 u_max: float = 2.5, u_min: float = 0.0, rest_s: float = 0.0) -> GcdCell:
    for I in currents_mA:
        for _ in range(n_cycles):
            cell.cycle_(I, u_max, u_min, rest_s)
    return cell


def write_txt(df: pd.DataFrame, path: Path) -> Path:
    """Tab-separated like Output.txt: padded names, units row, CRLF."""
    lines = ["\t".join(f"{c:<9}" for c in COLUMNS), "\t".join(f"{u:<9}" for u in UNITS)]
    for row in df.itertuples(index=False):
        lines.append("\t".join(v if isinstance(v, str) else f"{v:.10g}" for v in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", newline="")
    return path


def write_masses(folder: Path, m: float) -> Path:
    """Input-Variables.txt in the EL-Cell layout, m1 = m2 = m [g]."""
    p = folder / "Input-Variables.txt"
    p.write_text(
        "t        \tKey \tValue    \tt_unix          \r\n"
        "s        \t    \t         \ts               \r\n"
        f"-7.38\tm2  \t{m}  \t1787840128.476  \r\n"
        f"-7.38\tm1  \t{m}  \t1787840128.476  \r\n",
        encoding="ascii", newline="")
    return p
