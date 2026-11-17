"""
run_log.py
----------
Per-measurement log shared by the CV and GCD evaluation: every message of
the evaluation modules (INFO and up -- automatic decisions, flags,
warnings) goes to a log file next to the results; warnings also go to the
console and to a list that ends up in the results workbook.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

IO_LOGGER = "supercap_pipeline.io"


class _WarningCollector(logging.Handler):
    def __init__(self, sink: list[str]):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage())


@contextmanager
def evaluation_log(path: Path, *logger_names: str):
    """
    Route the named loggers (plus the shared reader's logger) to `path`
    for the duration of the block; yields the list of warning messages.
    Logger levels, handlers and propagation are restored afterwards.
    """
    names = list(dict.fromkeys([*logger_names, IO_LOGGER]))
    loggers = [logging.getLogger(n) for n in names]
    saved = [(lg.level, lg.propagate, list(lg.handlers)) for lg in loggers]

    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("  %(levelname)s: %(message)s"))
    sink: list[str] = []
    collector = _WarningCollector(sink)
    level = logging.DEBUG if logging.getLogger().level <= logging.DEBUG else logging.INFO
    for lg in loggers:
        lg.handlers = [fh, console, collector]
        lg.setLevel(level)
        lg.propagate = False
    try:
        yield sink
    finally:
        fh.close()
        for lg, (lvl, prop, handlers) in zip(loggers, saved):
            lg.setLevel(lvl)
            lg.propagate = prop
            lg.handlers = handlers
