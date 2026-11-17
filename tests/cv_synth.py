"""
tests/cv_synth.py
-----------------
Synthetic PAT-Tester CV exports of an ideal capacitor C with series
resistance R, in the same column format as the real export (padded column
names, units row, "VS " with trailing space, signed Qcd reset per step).

The cell voltage V12 is controlled; x = V12 - V_C obeys dx/dt = r - x/(RC)
for a ramp of rate r, so within one step (exactly):
    x(t) = r*R*C + (x0 - r*R*C) * exp(-t/RC)
    I    = x / R
    Q(t) = r*C*t + (x0 - r*R*C) * C * (1 - exp(-t/RC))      [C]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["t", "t_cycle", "t_step", "cycle", "step", "type", "c/d",
           "V12", "I12", "|Qstep|", "Qcd", "|Qcd|", "Qc", "Qd"]
UNITS = ["s", "s", "s", "", "", "", "", "V", "mA", "mAh", "mAh", "mAh", "mAh", "mAh"]


@dataclass
class SynthCell:
    C: float = 1.0          # F
    R: float = 1.0          # Ohm
    pts_per_sweep: int = 120
    t: float = 0.0
    V: float = 0.0          # applied cell voltage
    x: float = 0.0          # V - V_C (= I*R)
    cycle: int = 0
    rows: list = field(default_factory=list)

    # one step: ramp from the current V at rate r (V/s, may be 0) for duration T
    def _step(self, r: float, T: float, n: int, step: int, typ: str = "VS") -> None:
        RC = self.R * self.C
        ts = np.linspace(0.0, T, n + 1)
        x0 = self.x
        e = np.exp(-ts / RC)
        if typ == "Rest":
            x = np.zeros_like(ts)
            q = np.zeros_like(ts)
            v = np.full_like(ts, self.V - x0)  # open circuit: V = V_C
        else:
            x = r * RC + (x0 - r * RC) * e
            q = r * self.C * ts + (x0 - r * RC) * self.C * (1 - e)
            v = self.V + r * ts
        cd = "-" if typ == "Rest" else ("c" if r > 0 else ("d" if r < 0 else "c"))
        q_mAh = q / 3.6
        for k in range(len(ts)):
            self.rows.append([
                self.t + ts[k], 0.0, ts[k], self.cycle, step, typ + " ", cd + " ",
                v[k], x[k] / self.R * 1000.0, abs(q_mAh[k]), q_mAh[k], abs(q_mAh[k]),
                max(q_mAh[k], 0.0), max(-q_mAh[k], 0.0),
            ])
        self.t += T
        self.V = float(v[-1])
        self.x = float(x[-1])

    def sweep(self, v_to: float, nu: float, step: int) -> None:
        dv = v_to - self.V
        T = abs(dv) / nu
        self._step(np.sign(dv) * nu, T, self.pts_per_sweep, step)

    def hold(self, nu: float, step: int, n: int = 4) -> None:
        """Short transition step (a few VS rows at constant voltage)."""
        dt = 2.0 / nu / self.pts_per_sweep
        self._step(0.0, n * dt, n - 1, step)

    def rest(self, T: float = 60.0, step: int = 5) -> None:
        self._step(0.0, T, 6, step, typ="Rest")
        self.x = 0.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


def symmetric_program(blocks: list[tuple[float, float]], n_cycles: int,
                      cell: SynthCell | None = None) -> SynthCell:
    """blocks = [(U_max [V], nu [V/s]), ...]; each cycle: hold at -U, -U->+U, +U->-U."""
    cell = cell or SynthCell()
    for i, (u, nu) in enumerate(blocks):
        if i:
            cell.rest()
        cell.cycle += 1
        cell.sweep(-u, nu, step=10)        # transition to -U
        for _ in range(n_cycles):
            cell.cycle += 1
            cell.hold(nu, step=10)
            cell.sweep(+u, nu, step=20)
            cell.sweep(-u, nu, step=30)
    return cell


def one_sided_program(u: float, rates: list[float], n_cycles: int) -> SynthCell:
    """0 -> U -> 0 cycles at each rate; each cycle: hold at 0, 0->U, U->0."""
    cell = SynthCell()
    for i, nu in enumerate(rates):
        if i:
            cell.rest()
        for _ in range(n_cycles):
            cell.cycle += 1
            cell.hold(nu, step=10)
            cell.sweep(u, nu, step=20)
            cell.sweep(0.0, nu, step=30)
    return cell


def write_txt(df: pd.DataFrame, path: Path) -> Path:
    """Tab-separated like Output.txt: padded names, units row, CRLF."""
    lines = ["\t".join(f"{c:<9}" for c in COLUMNS), "\t".join(f"{u:<9}" for u in UNITS)]
    for row in df.itertuples(index=False):
        lines.append("\t".join(v if isinstance(v, str) else f"{v:.9g}" for v in row))
    path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", newline="")
    return path


def write_xlsx(df: pd.DataFrame, path: Path) -> Path:
    """xlsx with sheet "Output": row 1 names (padded), row 2 units, data from row 3."""
    units = pd.DataFrame([UNITS], columns=COLUMNS)
    out = pd.concat([units, df.astype(object)], ignore_index=True)
    out.columns = [f"{c} " for c in COLUMNS]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Output", index=False)
    return path


def write_masses(folder: Path, m1: str = "0.00138", m2: str = "0.00138") -> Path:
    """Input-Variables.txt in the EL-Cell layout."""
    p = folder / "Input-Variables.txt"
    p.write_text(
        "t        \tKey \tValue    \tt_unix          \r\n"
        "s        \t    \t         \ts               \r\n"
        f"-7.384975\tm2  \t{m2}  \t1787840128.476  \r\n"
        "-7.384964\tmR  \t1        \t1787840128.476  \r\n"
        f"-7.384961\tm1  \t{m1}  \t1787840128.476  \r\n",
        encoding="ascii", newline="")
    return p
