"""Logging setup.

Why a research project needs real logging rather than ``print``:

* The pair screen tests ~20,000 pairs. When the funnel comes out at 3 pairs
  instead of 30, the only way to find out why is a record of what was rejected
  and at which stage.
* Walk-forward runs are long. A run that dies in window 11 of 16 must leave
  behind enough to diagnose it without re-running the first ten.
* Every number in the final report has to be traceable to a run. The log file
  carries the run id and the config hash, so a figure can be tied back to the
  exact parameters that produced it.

Console output stays terse; the file handler keeps DEBUG detail. Setup is
idempotent --- calling ``setup_logging`` twice does not double every line,
which matters because notebooks re-execute cells.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from dsa.paths import logs_dir

__all__ = ["setup_logging", "get_logger", "reset_logging"]

_ROOT_NAME = "dsa"
_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(funcName)s:%(lineno)d %(message)s"
_DATE_FMT = "%H:%M:%S"
_FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_configured = False


def setup_logging(
    level: str | int = "INFO",
    *,
    log_to_file: bool = True,
    run_id: str | None = None,
    log_dir: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``dsa`` logger hierarchy. Safe to call repeatedly.

    Parameters
    ----------
    level
        Console level. The file handler always records DEBUG.
    log_to_file
        Write a rotating log under ``logs/``.
    run_id
        If given, the run gets its own file ``logs/run_<run_id>.log`` in
        addition to the rolling ``logs/dsa.log``. Use this for anything whose
        output ends up in the report.
    force
        Tear down existing handlers first.
    """
    global _configured

    logger = logging.getLogger(_ROOT_NAME)

    if _configured and not force:
        return logger
    if force:
        reset_logging()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(logging.DEBUG)  # handlers do the filtering
    logger.propagate = False  # do not double-print via the root logger

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    logger.addHandler(console)

    if log_to_file:
        directory = log_dir or logs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE_FMT)

        rolling = logging.handlers.RotatingFileHandler(
            directory / "dsa.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        rolling.setLevel(logging.DEBUG)
        rolling.setFormatter(file_fmt)
        logger.addHandler(rolling)

        if run_id:
            per_run = logging.FileHandler(directory / f"run_{run_id}.log", encoding="utf-8")
            per_run.setLevel(logging.DEBUG)
            per_run.setFormatter(file_fmt)
            logger.addHandler(per_run)

    _configured = True
    return logger


def reset_logging() -> None:
    """Remove all handlers. Used by tests and by ``setup_logging(force=True)``."""
    global _configured
    logger = logging.getLogger(_ROOT_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    _configured = False


def get_logger(name: str) -> logging.Logger:
    """Logger for a module. Pass ``__name__``.

    Names are normalised under the ``dsa`` root so that one call to
    ``setup_logging`` configures every module in the package.
    """
    if name in {"__main__", "__mp_main__"}:
        return logging.getLogger(f"{_ROOT_NAME}.main")
    if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
