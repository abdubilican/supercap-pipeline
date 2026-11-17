"""
tests/peis_synth.py
-------------------
Synthetic PAT-Tester impedance exports (Output-Impedance.txt layout: names,
units row, one NaN end row per spectrum, half-cell columns NaN) generated
from the lab-guide model itself with impedance.py:
    R0-p(R1,CPE1)-TLMQ0  (R_s, R_c, Q_c, alpha_c, R_ion, Q_dl, alpha_dl)
    R0-TLMQ0             if R_c = 0
plus an optional series inductance (Z += j w L) and +-noise relative to |Z|.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from impedance.models.circuits import CustomCircuit

COLUMNS = ["freq", "t", "t_cycle", "cycle", "step", "type", "p/g", "c/d", "con", "V12", "V1R",
           "V2R", "dV12", "I12", "dI12", "Re(Z12)", "Im(Z12)", "|Z12|", "phi(Z12)", "SFDR(Z12)",
           "Re(Z1)", "Im(Z1)", "|Z1|", "phi(Z1)", "SFDR(Z1)", "Re(Z2)", "Im(Z2)", "|Z2|",
           "phi(Z2)", "SFDR(Z2)", "Cs", "Cp", "dt"]
UNITS = ["Hz", "s", "s", "", "", "", "", "", "", "V", "V", "V", "mV", "mA", "mA", "Ohm", "Ohm",
         "Ohm", "deg", "dB", "Ohm", "Ohm", "Ohm", "deg", "dB", "Ohm", "Ohm", "Ohm", "deg", "dB",
         "uF", "uF", "s"]
TRUE = dict(R_s=0.2, R_c=3.0, Q_c=1e-5, alpha_c=0.85, R_ion=1.5, Q_dl=1.0, alpha_dl=0.93)


def model_z(f: np.ndarray, p: dict, L: float = 0.0) -> np.ndarray:
    if p["R_c"] > 0:
        c = CustomCircuit("R0-p(R1,CPE1)-TLMQ0", initial_guess=[
            p["R_s"], p["R_c"], p["Q_c"], p["alpha_c"], p["R_ion"], p["Q_dl"], p["alpha_dl"]])
    else:
        c = CustomCircuit("R0-TLMQ0", initial_guess=[p["R_s"], p["R_ion"], p["Q_dl"], p["alpha_dl"]])
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = c.predict(f)
    return Z + 1j * 2 * np.pi * f * L


@dataclass
class ImpedanceFile:
    rows: list = field(default_factory=list)
    t: float = 0.0
    step: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    def spectrum(self, V_dc: float, params: dict | None = None, f_max: float = 1e5,
                 f_min: float = 1e-2, ppd: int = 10, noise: float = 0.003, L: float = 0.0,
                 bad_first: bool = True) -> None:
        """One spectrum from f_max to f_min; the first point gets a low amplitude and SFDR."""
        p = dict(TRUE, **(params or {}))
        n = int(round(np.log10(f_max / f_min) * ppd)) + 1
        f = np.logspace(np.log10(f_max), np.log10(f_min), n)
        Z = model_z(f, p, L)
        Z = Z + np.abs(Z) * noise * (self.rng.uniform(-1, 1, n) + 1j * self.rng.uniform(-1, 1, n))
        self.step += 50
        for k in range(n):
            z = Z[k]
            dt = max(0.1, 2.0 / f[k])
            self.t += dt
            amp, sfdr = (3.2, 15.0) if (bad_first and k == 0) else (10.0, 60.0)
            cs = -1.0 / (2 * np.pi * f[k] * z.imag) * 1e6
            self.rows.append([f[k], self.t, 0.0, 1, self.step, "PEIS ", "-", "-", "i12_V12 ",
                              V_dc + 1e-5 * self.rng.standard_normal(), np.nan, np.nan, amp,
                              0.01 * self.rng.standard_normal(), amp / abs(z), z.real, z.imag,
                              abs(z), np.degrees(np.angle(z)), sfdr] + [np.nan] * 10
                             + [cs, np.nan, dt])
        self.t += 1.0
        self.rows.append([np.nan, self.t, 0.0, 1, self.step, "PEIS ", "-", "-", "i12_V12 "]
                         + [np.nan] * (len(COLUMNS) - 9))

    def write(self, folder: Path, m: float = 0.032164) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        lines = ["\t".join(f"{c:<9}" for c in COLUMNS), "\t".join(f"{u:<9}" for u in UNITS)]
        for r in self.rows:
            lines.append("\t".join(v if isinstance(v, str) else
                                   ("NaN" if isinstance(v, float) and np.isnan(v) else f"{v:.10g}")
                                   for v in r))
        p = folder / "Output-Impedance.txt"
        p.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", newline="")
        (folder / "Input-Variables.txt").write_text(
            "t        \tKey \tValue    \tt_unix          \r\n"
            "s        \t    \t         \ts               \r\n"
            f"-7.38\tm2  \t{m}  \t1787840128.476  \r\n"
            f"-7.38\tm1  \t{m}  \t1787840128.476  \r\n", encoding="ascii", newline="")
        return p
