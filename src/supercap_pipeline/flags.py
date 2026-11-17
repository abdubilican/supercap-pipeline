"""
flags.py
--------
Flag handling shared by the evaluation modules: severities and a collector
that turns (severity, text, context) entries into the "Flags" sheet, most
severe first. Flags never remove data; they document what to look at.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

FLAG, WARNING, INFO = "flag", "warning", "info"
SEVERITY_ORDER = {FLAG: 0, WARNING: 1, INFO: 2}


def sort_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags first, then warnings, then info; stable within a severity."""
    if df.empty:
        return df
    order = df["severity"].map(SEVERITY_ORDER).fillna(9)
    return df.assign(_o=order).sort_values("_o", kind="stable").drop(columns="_o") \
             .reset_index(drop=True)


class FlagCollector:
    """Collects flags as rows {severity, <context columns>, flag, explanation}."""

    def __init__(self, context: tuple[str, ...], explain: Callable[[str], str] = lambda t: ""):
        self.context = context
        self.explain = explain
        self.rows: list[dict] = []

    def add(self, severity: str, text: str, **ctx) -> None:
        self.rows.append({"severity": severity, **{c: ctx.get(c) for c in self.context},
                          "flag": text, "explanation": self.explain(text)})

    def frame(self) -> pd.DataFrame:
        cols = ["severity", *self.context, "flag", "explanation"]
        return sort_flags(pd.DataFrame(self.rows, columns=cols))
