"""Parquet storage with a manifest, and the raw/clean boundary.

The rule this module enforces
-----------------------------
``data/raw`` is **write-once**. A file there is the provider's response as
received, and nothing in the codebase may edit it in place. Everything derived
goes to ``data/clean``. If a cleaning step turns out to be wrong, you fix the
step and rebuild ``clean`` --- you never touch ``raw``, because then the thing
you would have to re-download to check is gone.

Overwriting a raw file requires ``force=True``, which only the downloader's
explicit ``--refresh`` path passes. Everything else raises.

Why a manifest
--------------
A directory of parquet files does not tell you whether a download finished,
when it ran, what it covered, or whether the bytes changed since. The manifest
records per ticker: row count, first and last bar, when it was fetched, the
source, and a content digest. That is what makes the download *resumable* ---
"already done" is a fact about the manifest, not a guess from a file existing.

A note on what "raw" can mean with a free feed
----------------------------------------------
``data/raw`` here means *as received from the provider, unmodified*. It does
not mean *as traded*. Yahoo's ``Close`` for NSE names arrives already
split-adjusted (verified: the 2:1 RELIANCE split on 2017-09-07 shows as
-0.56%, not -50%). No free feed exposes a genuinely unadjusted NSE series.
This matters for the corporate-action check (spec Sec 11.2) and is documented
in ``dsa.data.clean``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dsa.logging_utils import get_logger
from dsa.paths import data_clean, data_raw

__all__ = [
    "StoreError",
    "TickerRecord",
    "Manifest",
    "write_parquet",
    "read_parquet",
    "raw_ohlcv_path",
    "raw_reference_dir",
    "clean_path",
]

_log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


class StoreError(RuntimeError):
    """Raised on a storage-contract violation (e.g. overwriting raw data)."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def raw_ohlcv_dir() -> Path:
    d = data_raw() / "ohlcv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_reference_dir() -> Path:
    """Where the NSE reference CSVs land, byte-for-byte as downloaded."""
    d = data_raw() / "reference"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_ohlcv_path(symbol: str) -> Path:
    return raw_ohlcv_dir() / f"{symbol}.parquet"


def clean_path(name: str) -> Path:
    return data_clean() / f"{name}.parquet"


# ---------------------------------------------------------------------------
# atomic parquet io
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_parquet(df: pd.DataFrame, path: Path, *, force: bool = False) -> Path:
    """Write ``df`` to ``path`` atomically.

    Atomic because a half-written parquet that looks present is worse than one
    that is absent: the manifest would call it done and the gap would surface
    weeks later as a mysterious hole in one ticker.

    Raises ``StoreError`` when the target is under ``data/raw`` and already
    exists, unless ``force=True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force and path.resolve().is_relative_to(data_raw().resolve()):
        raise StoreError(
            f"refusing to overwrite raw data at {path}. data/raw is write-once: it is "
            "the provider's response as received, and the only copy you can re-check a "
            "cleaning bug against. Rebuild data/clean instead, or pass force=True from "
            "an explicit refresh."
        )

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=True)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no parquet at {path}")
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass
class TickerRecord:
    """What is known about one ticker's raw download."""

    symbol: str
    status: str  # "ok" | "empty" | "failed"
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    requested_start: str | None = None
    requested_end: str | None = None
    downloaded_at: str | None = None
    source: str = "yfinance"
    digest: str | None = None
    error: str | None = None
    attempts: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status == "ok" and self.rows > 0

    def covers(self, start: str, end: str) -> bool:
        """True when this window has already been fetched successfully.

        The question is deliberately *"have I already asked for this window and
        got an answer?"* rather than *"does the stored data reach the end
        date?"*. Those differ for any name that stopped trading partway through
        the sample: a stock delisted in 2019 will never have a 2026 bar, and
        judging coverage by the last stored date would re-download it on every
        single run, forever, burning rate-limit budget to receive the same
        answer each time.

        So:

        * identical request already satisfied -> covered;
        * an *earlier* request that started no later and whose data already
          runs to within a week of the new end -> covered (a widened window
          that adds nothing);
        * anything else -> fetch again.

        The week of slack absorbs weekends and exchange holidays, so asking for
        data "to last Sunday" does not force a re-pull.
        """
        if not self.is_complete or self.last_date is None:
            return False

        if self.requested_start == start and self.requested_end == end:
            return True

        if self.requested_start is not None and pd.Timestamp(self.requested_start) > pd.Timestamp(
            start
        ):
            return False  # the new request reaches further back than the old one

        return pd.Timestamp(self.last_date) >= pd.Timestamp(end) - pd.Timedelta(days=7)


@dataclass
class Manifest:
    """Index of a raw download directory. Read/modify/write as a whole."""

    version: int = MANIFEST_VERSION
    created_at: str = ""
    updated_at: str = ""
    source: str = "yfinance"
    requested_start: str | None = None
    requested_end: str | None = None
    tickers: dict[str, TickerRecord] = field(default_factory=dict)

    # -- io -----------------------------------------------------------------

    @classmethod
    def path(cls, directory: Path | None = None) -> Path:
        return (directory or raw_ohlcv_dir()) / MANIFEST_NAME

    @classmethod
    def load(cls, directory: Path | None = None) -> Manifest:
        path = cls.path(directory)
        if not path.is_file():
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return cls(created_at=now, updated_at=now)
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = {k: TickerRecord(**v) for k, v in raw.get("tickers", {}).items()}
        return cls(
            version=raw.get("version", MANIFEST_VERSION),
            created_at=raw.get("created_at", ""),
            updated_at=raw.get("updated_at", ""),
            source=raw.get("source", "yfinance"),
            requested_start=raw.get("requested_start"),
            requested_end=raw.get("requested_end"),
            tickers=records,
        )

    def save(self, directory: Path | None = None) -> Path:
        path = self.path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload: dict[str, Any] = {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "tickers": {k: asdict(v) for k, v in sorted(self.tickers.items())},
        }
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return path

    # -- updates ------------------------------------------------------------

    def record_success(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        path: Path,
        start: str,
        end: str,
        attempts: int,
        source: str = "yfinance",
    ) -> TickerRecord:
        rec = TickerRecord(
            symbol=symbol,
            status="ok" if len(df) else "empty",
            rows=len(df),
            first_date=str(pd.Timestamp(df.index.min()).date()) if len(df) else None,
            last_date=str(pd.Timestamp(df.index.max()).date()) if len(df) else None,
            requested_start=start,
            requested_end=end,
            downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            digest=_digest(path) if path.is_file() else None,
            attempts=attempts,
        )
        self.tickers[symbol] = rec
        return rec

    def record_failure(
        self, symbol: str, error: str, *, start: str, end: str, attempts: int
    ) -> TickerRecord:
        rec = TickerRecord(
            symbol=symbol,
            status="failed",
            requested_start=start,
            requested_end=end,
            downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            error=error[:500],
            attempts=attempts,
        )
        self.tickers[symbol] = rec
        return rec

    # -- queries ------------------------------------------------------------

    def completed(self) -> list[str]:
        return sorted(s for s, r in self.tickers.items() if r.is_complete)

    def failed(self) -> list[str]:
        return sorted(s for s, r in self.tickers.items() if r.status == "failed")

    def empty(self) -> list[str]:
        return sorted(s for s, r in self.tickers.items() if r.status == "empty")

    def pending(self, symbols: list[str], start: str, end: str) -> list[str]:
        """Symbols still to fetch --- the resumability decision, in one place."""
        out = []
        for s in symbols:
            rec = self.tickers.get(s)
            if rec is None or not rec.covers(start, end):
                out.append(s)
        return out

    def summary(self) -> str:
        n_ok, n_failed, n_empty = len(self.completed()), len(self.failed()), len(self.empty())
        rows = sum(r.rows for r in self.tickers.values())
        return (
            f"manifest: {len(self.tickers)} tickers "
            f"({n_ok} ok, {n_failed} failed, {n_empty} empty), {rows:,} rows total"
        )

    def __iter__(self) -> Iterator[TickerRecord]:
        return iter(self.tickers.values())

    def __len__(self) -> int:
        return len(self.tickers)
