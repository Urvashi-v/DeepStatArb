"""Historical daily OHLCV acquisition from Yahoo Finance.

CREDENTIALS: none required. See ``dsa.data.sources``.

What is stored, and why it is stored that way
---------------------------------------------
Each ticker is fetched with ``auto_adjust=False, actions=True`` and written to
its own parquet under ``data/raw/ohlcv/``. That gives eight columns --- Open,
High, Low, Close, Adj Close, Volume, Dividends, Stock Splits --- and keeping
all eight is the point:

* ``Adj Close / Close`` is the total-return adjustment factor. Storing both
  ends means the adjustment can be *recomputed and audited* later rather than
  taken on faith from a flag passed to a library.
* ``Dividends`` and ``Stock Splits`` are what the corporate-action quality
  check (spec Sec 11.2) tests the price series against. Downloading with
  ``auto_adjust=True`` throws them away, and with them the ability to notice
  that the provider handled an event badly.

One file per ticker, not one big panel, because that is what makes the
download resumable: 180 of 208 finished and the process died is a normal
Tuesday, and it must cost 28 requests to recover, not 208.

Honest limits of this feed
--------------------------
1. Yahoo's ``Close`` for NSE names is **already split-adjusted**. Verified:
   the 2:1 RELIANCE split on 2017-09-07 appears as a -0.56% move, not -50%.
   So ``data/raw`` is "as received", not "as traded". No free feed offers a
   genuinely unadjusted NSE series.
2. Back-adjustment embeds future information in past levels. The adjusted
   2015-01-01 RELIANCE close is 0.93184 x the split-adjusted close, and that
   factor is the product of every dividend paid *after* 2015. Spec Sec 11.2
   mandates adjusted prices anyway, and the alternative (dividend jumps in the
   spread) is worse --- but the caveat is real and is restated in
   ``dsa.data.clean``.
3. Free feeds have gaps and bad prints. Cross-checking against a second source
   is a Day 3 task, not something this module claims to have done.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from dsa.data.sources import DataSourceError, RateLimiter, RetryPolicy, with_retry
from dsa.data.store import Manifest, raw_ohlcv_path, write_parquet
from dsa.logging_utils import get_logger

__all__ = [
    "DownloadConfig",
    "DownloadResult",
    "download_ticker",
    "download_universe",
    "RAW_COLUMNS",
]

_log = get_logger(__name__)

# The columns yfinance returns with auto_adjust=False, actions=True.
RAW_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

# Substrings in a yfinance error/warning that mean "this symbol will never
# work", so retrying it only burns rate-limit budget.
_PERMANENT_MARKERS = (
    "no timezone found",
    "possibly delisted",
    "no data found",
    "symbol may be delisted",
)


@dataclass(frozen=True)
class DownloadConfig:
    """Acquisition settings. Deliberately conservative by default."""

    start: str = "2015-01-01"
    end: str = "2026-08-01"
    min_interval_s: float = 0.40
    jitter_s: float = 0.25
    max_attempts: int = 4
    base_delay_s: float = 1.5
    max_delay_s: float = 60.0
    timeout_s: float = 30.0
    min_rows: int = 250  # ~1 year; below this a "successful" fetch is suspect

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.max_attempts,
            base_delay_s=self.base_delay_s,
            max_delay_s=self.max_delay_s,
        )

    def limiter(self) -> RateLimiter:
        return RateLimiter(min_interval_s=self.min_interval_s, jitter_s=self.jitter_s)


@dataclass
class DownloadResult:
    """Outcome of a batch download."""

    requested: int
    downloaded: int
    skipped: int
    empty: list[str]
    failed: list[str]
    elapsed_s: float
    manifest: Manifest

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (
            f"requested {self.requested} | downloaded {self.downloaded} | "
            f"skipped (already complete) {self.skipped} | empty {len(self.empty)} | "
            f"failed {len(self.failed)} | {self.elapsed_s:.1f}s"
        )


class PermanentSymbolError(DataSourceError):
    """The symbol does not resolve. Retrying will not help."""


def _fetch_once(symbol: str, start: str, end: str, timeout: float) -> pd.DataFrame:
    """One yfinance call. Raises rather than returning an empty frame silently."""
    import yfinance as yf  # imported here so the module loads without network deps

    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        raise_errors=False,
        timeout=timeout,
    )

    if df is None or df.empty:
        raise PermanentSymbolError(
            f"{symbol}: yfinance returned no rows for {start}..{end}. The symbol is "
            "unknown, delisted, or has no history in this window."
        )

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(
            f"{symbol}: response is missing {missing}. Got {list(df.columns)}. The "
            "provider's schema changed --- do not clean around it, investigate it."
        )

    return df[RAW_COLUMNS].copy()


def download_ticker(
    symbol: str,
    *,
    config: DownloadConfig | None = None,
    limiter: RateLimiter | None = None,
    fetcher: Callable[[str, str, str, float], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch one ticker with retry and rate limiting.

    Returns ``(frame, attempts)``. Raises ``DataSourceError`` on failure ---
    never an empty frame passed off as a result.
    """
    cfg = config or DownloadConfig()
    fetch = fetcher or _fetch_once
    attempts = 0

    def _call() -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        if limiter is not None:
            limiter.wait()
        return fetch(symbol, cfg.start, cfg.end, cfg.timeout_s)

    df = with_retry(
        _call,
        policy=cfg.retry_policy(),
        what=f"download {symbol}",
        give_up_on=(PermanentSymbolError, PermissionError),
    )
    return df, attempts


def download_universe(
    symbols: Sequence[str],
    *,
    config: DownloadConfig | None = None,
    resume: bool = True,
    refresh: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    fetcher: Callable[[str, str, str, float], pd.DataFrame] | None = None,
) -> DownloadResult:
    """Download many tickers, resumably.

    Parameters
    ----------
    resume
        Skip symbols the manifest already records as covering the window.
        This is the whole reason the manifest exists.
    refresh
        Re-fetch and overwrite even completed symbols. This is the only path
        that may overwrite ``data/raw``.
    fetcher
        Injected for tests, so the retry/resume/manifest logic can be tested
        without touching the network. Production always uses the real one.
    """
    cfg = config or DownloadConfig()
    started = time.monotonic()
    symbols = list(dict.fromkeys(symbols))  # de-duplicate, preserve order

    manifest = Manifest.load()
    manifest.requested_start = cfg.start
    manifest.requested_end = cfg.end
    if not manifest.created_at:
        manifest.created_at = manifest.updated_at

    if refresh:
        todo = list(symbols)
    elif resume:
        todo = manifest.pending(list(symbols), cfg.start, cfg.end)
    else:
        todo = list(symbols)

    skipped = len(symbols) - len(todo)
    _log.info(
        "download: %d symbols requested, %d already complete, %d to fetch (%s..%s)",
        len(symbols),
        skipped,
        len(todo),
        cfg.start,
        cfg.end,
    )

    limiter = cfg.limiter()
    downloaded = 0

    for i, symbol in enumerate(todo, start=1):
        if progress is not None:
            progress(i, len(todo), symbol)
        try:
            df, attempts = download_ticker(symbol, config=cfg, limiter=limiter, fetcher=fetcher)
        except (DataSourceError, PermissionError) as exc:
            _log.error("download %s FAILED: %s", symbol, str(exc)[:200])
            manifest.record_failure(
                symbol, str(exc), start=cfg.start, end=cfg.end, attempts=cfg.max_attempts
            )
            manifest.save()
            continue

        if len(df) < cfg.min_rows:
            _log.warning(
                "%s: only %d rows for %s..%s (below min_rows=%d). Stored, but flagged "
                "as too short to use.",
                symbol,
                len(df),
                cfg.start,
                cfg.end,
                cfg.min_rows,
            )

        path = raw_ohlcv_path(symbol)
        write_parquet(df, path, force=True)  # manifest gates re-fetching, not this
        manifest.record_success(
            symbol, df, path=path, start=cfg.start, end=cfg.end, attempts=attempts
        )
        manifest.save()  # save after every ticker: a crash loses one, not all
        downloaded += 1
        _log.debug(
            "%s: %d rows %s..%s (%d attempt(s))",
            symbol,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
            attempts,
        )

    result = DownloadResult(
        requested=len(symbols),
        downloaded=downloaded,
        skipped=skipped,
        empty=manifest.empty(),
        failed=manifest.failed(),
        elapsed_s=time.monotonic() - started,
        manifest=manifest,
    )
    _log.info("download complete: %s", result.summary())
    return result


def load_raw(symbol: str) -> pd.DataFrame:
    """Read one ticker's raw parquet back."""
    from dsa.data.store import read_parquet

    return read_parquet(raw_ohlcv_path(symbol))


def available_symbols() -> list[str]:
    """Symbols with a complete raw download, per the manifest."""
    return Manifest.load().completed()


def iter_raw(symbols: Iterable[str] | None = None) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield ``(symbol, raw_frame)`` for each available symbol."""
    for symbol in symbols if symbols is not None else available_symbols():
        path = raw_ohlcv_path(symbol)
        if path.is_file():
            from dsa.data.store import read_parquet

            yield symbol, read_parquet(path)


def default_end_date() -> str:
    """Today, as an ISO date. Kept in one place so runs are comparable."""
    return date.today().isoformat()
