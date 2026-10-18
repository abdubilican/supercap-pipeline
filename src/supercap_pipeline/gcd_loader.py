"""
gcd_loader.py
-------------
Loads a galvanostatic charge/discharge (GCD) experiment folder and returns
a unified raw DataFrame + metadata, mirroring loader.load_experiment().

Returned DataFrame columns
---------------------------
  All Output.txt columns, stripped and numeric-coerced (see
  loader.read_tester_txt), with `cycle`/`step` cast to int. Step
  segmentation, technique classification and rate-stage detection happen in
  gcd_analysis.segment_steps() -- this module only does raw I/O.

Metadata (unlike loader.load_experiment(), masses and the normalization
convention are REQUIRED -- there is no silent fallback. A missing mass or
missing `gcdConfig.basis` in info.json is a hard error, by design: the task
explicitly forbids guessing a normalization convention.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .loader import composition_field, read_info_json, read_tester_txt

log = logging.getLogger(__name__)

VALID_BASES: set[str] = {"cell", "single_electrode", "total_active_mass"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_gcd_experiment(
    gcd_folder: str | Path,
    basis: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load one GCD measurement folder.

    Parameters
    ----------
    gcd_folder : path to the folder containing Output.txt and *info.json
                 (optionally Output-Cycles.txt for the tester's own per-cycle
                 Qc/Qd/Wc/Wd/efficiency values, used as a cross-check only)
    basis      : normalization convention chosen explicitly by the caller
                 (e.g. `supercap-run --gcd-basis`). Overrides
                 `gcdConfig.basis` in info.json; if both are absent, loading
                 fails -- the convention is never guessed.

    Returns
    -------
    df   : raw time-series DataFrame (all Output.txt columns)
    meta : metadata dict -- cell/electrode info, REQUIRED normalization
           convention (basis, mass_total_g, mass_per_electrode_g), and the
           tester's own per-cycle cross-check table (meta["tester_cycles"],
           or None if Output-Cycles.txt is absent)
    """
    gcd_folder = Path(gcd_folder)
    info = read_info_json(gcd_folder)
    meta = _build_meta(info, gcd_folder, basis_override=basis)
    df   = _load_output_txt(gcd_folder)
    meta["tester_cycles"] = _load_tester_cycles(gcd_folder)

    log.info(
        "Loaded %d rows from %s; cycles: %s; basis=%s mass_total=%.4fg",
        len(df), gcd_folder.name,
        sorted(df.loc[df["cycle"] > 0, "cycle"].unique()),
        meta["basis"], meta["mass_total_g"],
    )
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Metadata (masses + normalization convention -- no defaults, hard errors)
# ─────────────────────────────────────────────────────────────────────────────

def _build_meta(info: dict, folder: Path, basis_override: str | None = None) -> dict:
    cell_tag = info["testCell"]["cellTag"]
    exp_name = info["experiment"]["experimentName"]

    e1_name = composition_field(info, "Electrode1")
    e2_name = composition_field(info, "Electrode2")
    electrolyte = composition_field(info, "Electrolyte")

    m1_raw = composition_field(info, "Weight1", subkey=None)
    m2_raw = composition_field(info, "Weight2", subkey=None)

    def _require_mass(raw, label: str) -> float:
        if raw is None:
            raise ValueError(
                f"{folder}: info.json is missing electrode mass '{label}' "
                f"(testCell.composition.fields[definition.key={label}].value). "
                "No default mass is assumed -- add the real value."
            )
        m = float(raw)
        if not m > 0:
            raise ValueError(
                f"{folder}: electrode mass '{label}' is {m} g (must be > 0). "
                "No default mass is assumed -- fix the value in info.json."
            )
        return m

    m1 = _require_mass(m1_raw, "Weight1")
    m2 = _require_mass(m2_raw, "Weight2")

    rel_asymmetry = abs(m1 - m2) / max(m1, m2)
    if rel_asymmetry > 0.02:
        log.warning(
            "%s: electrode masses differ by %.1f%% (m1=%.4fg, m2=%.4fg). "
            "The C_electrode = 2*C_cell relation assumes a symmetric "
            "two-electrode cell; specific values may not be strictly valid.",
            folder.name, rel_asymmetry * 100, m1, m2,
        )

    gcd_config = info.get("gcdConfig") or {}
    file_basis = gcd_config.get("basis")
    if basis_override is not None:
        if file_basis is not None and file_basis != basis_override:
            log.warning(
                "%s: --gcd-basis=%s overrides gcdConfig.basis=%s from info.json.",
                folder.name, basis_override, file_basis,
            )
        basis = basis_override
    elif file_basis is not None:
        basis = file_basis
    else:
        raise ValueError(
            f"{folder}: no normalization basis given. The convention is "
            "mandatory and is never defaulted -- pass it on the command line, "
            "e.g. `supercap-run <folder> --gcd-basis total_active_mass`, or add "
            '`"gcdConfig": {"basis": "total_active_mass"}` to info.json '
            f"(one of {sorted(VALID_BASES)})."
        )
    if basis not in VALID_BASES:
        raise ValueError(
            f"{folder}: GCD basis {basis!r} is not one of "
            f"{sorted(VALID_BASES)}."
        )

    return dict(
        cell_tag           = cell_tag,
        exp_name           = exp_name,
        e1_name            = e1_name or "?",
        e2_name            = e2_name or "?",
        electrolyte        = electrolyte,
        m1_g               = m1,
        m2_g               = m2,
        mass_total_g       = m1 + m2,
        mass_per_electrode_g = min(m1, m2),
        basis              = basis,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output.txt / Output-Cycles.txt loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_output_txt(folder: Path) -> pd.DataFrame:
    df = read_tester_txt(folder / "Output.txt")
    df = (
        df.dropna(subset=["t", "V12", "I12"])
          .sort_values("t")
          .reset_index(drop=True)
    )
    df["cycle"] = df["cycle"].fillna(0).astype(int)
    df["step"]  = df["step"].fillna(0).astype(int)
    return df


def _load_tester_cycles(folder: Path) -> pd.DataFrame | None:
    """
    Load the tester's own Output-Cycles.txt (Qc_cyc/Qd_cyc/Wc_cyc/Wd_cyc/
    Eff_Q/Eff_W per cycle), used purely as a cross-check against our own
    integrals -- never as the primary source. Returns None if absent.
    """
    path = folder / "Output-Cycles.txt"
    if not path.exists():
        return None

    df = read_tester_txt(path)
    df = df.dropna(subset=["cycle"]).copy()
    df["cycle"] = df["cycle"].astype(int)

    keep = {
        "cycle":  "cycle",
        "Qc_cyc": "Qc_cyc_tester_mAh",
        "Qd_cyc": "Qd_cyc_tester_mAh",
        "Wc_cyc": "Wc_cyc_tester_mWh",
        "Wd_cyc": "Wd_cyc_tester_mWh",
        "Eff_Q":  "CoulEff_tester",
        "Eff_W":  "EnergyEff_tester",
    }
    present = {k: v for k, v in keep.items() if k in df.columns}
    return df[list(present)].rename(columns=present)
