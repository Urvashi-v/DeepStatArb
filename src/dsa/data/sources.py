"""Network access: rate limiting, retry, and the NSE reference files.

CREDENTIALS: none. Every endpoint used here is public and unauthenticated ---
Yahoo's chart API for prices, and NSE's static archive CSVs for the F&O
eligibility list and index constituents. No key, token, or subscription is
required at any point. If that ever changes, it changes *here*, and the
failure will be a loud HTTP 401/403 rather than a silent fallback.

There is deliberately no offline mock and no synthetic generator in this
module. If the network is unavailable, the caller gets an exception and the
pipeline stops. A data layer that quietly invents prices when the feed is down
is the single most dangerous thing this project could contain.

Rate limiting
-------------
Yahoo publishes no rate limit, so the policy is politeness rather than
compliance with a stated number: a minimum interval between requests, plus
exponential backoff with jitter on 429 and 5xx. Jitter matters --- retrying a
batch of failures on a fixed schedule produces a synchronised second wave that
looks exactly like the abuse the limiter is there to prevent.
"""

from __future__ import annotations

import io
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
import requests

from dsa.logging_utils import get_logger

__all__ = [
    "DataSourceError",
    "RateLimiter",
    "RetryPolicy",
    "with_retry",
    "http_get",
    "fetch_fo_underlyings",
    "fetch_index_constituents",
    "NSE_FO_LOTS_URL",
    "NSE_NIFTY500_URL",
]

_log = get_logger(__name__)

T = TypeVar("T")

# NSE static archives. These are plain CSV files, no session or cookie needed.
NSE_FO_LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
NSE_NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# NSE's CDN rejects requests without a browser-like User-Agent. This is not
# authentication and not an attempt to evade anything --- the files are public
# downloads offered on the NSE website.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Rows in fo_mktlots.csv that are indices or the embedded section header,
# not individual securities.
_FO_NON_STOCK_SYMBOLS = {
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "NIFTYFPI",
    "SYMBOL",
}


class DataSourceError(RuntimeError):
    """A data source was unreachable or returned something unusable."""


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Enforce a minimum interval between calls. Thread-safe."""

    def __init__(self, min_interval_s: float = 0.4, jitter_s: float = 0.2) -> None:
        if min_interval_s < 0 or jitter_s < 0:
            raise ValueError("intervals must be non-negative")
        self.min_interval_s = min_interval_s
        self.jitter_s = jitter_s
        self._lock = threading.Lock()
        self._last: float = 0.0

    def wait(self) -> float:
        """Block until the next call is allowed. Returns seconds slept."""
        with self._lock:
            now = time.monotonic()
            target = self._last + self.min_interval_s + random.uniform(0, self.jitter_s)
            delay = max(0.0, target - now)
            if delay:
                time.sleep(delay)
            self._last = time.monotonic()
            return delay


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter."""

    max_attempts: int = 4
    base_delay_s: float = 1.5
    max_delay_s: float = 60.0
    backoff: float = 2.0

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        r = rng or random
        raw = min(self.max_delay_s, self.base_delay_s * (self.backoff ** (attempt - 1)))
        # Full jitter: uniform in [0, raw]. Without it, a batch that fails
        # together retries together and re-creates the original burst.
        return r.uniform(0.0, raw)


def with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    what: str = "request",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, retrying with backoff. Re-raises the last error on failure.

    ``give_up_on`` short-circuits errors that will never succeed on a retry
    (an unknown ticker, a 404). Retrying those wastes the rate-limit budget
    that the recoverable failures need.
    """
    policy = policy or RetryPolicy()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except give_up_on:
            raise
        except retry_on as exc:
            last = exc
            if attempt >= policy.max_attempts:
                break
            delay = policy.delay_for(attempt)
            _log.warning(
                "%s failed (attempt %d/%d): %s --- retrying in %.1fs",
                what,
                attempt,
                policy.max_attempts,
                str(exc)[:200],
                delay,
            )
            sleep(delay)

    raise DataSourceError(
        f"{what} failed after {policy.max_attempts} attempts: {last}"
    ) from last


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def http_get(
    url: str,
    *,
    timeout: float = 30.0,
    policy: RetryPolicy | None = None,
    limiter: RateLimiter | None = None,
) -> bytes:
    """GET ``url`` with retry and rate limiting. Returns the body."""

    def _once() -> bytes:
        if limiter is not None:
            limiter.wait()
        resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=timeout)
        if resp.status_code in _RETRYABLE_STATUS:
            raise DataSourceError(f"HTTP {resp.status_code} from {url}")
        if resp.status_code == 401 or resp.status_code == 403:
            # Surface this loudly: it is the signal that a previously open
            # endpoint now wants credentials.
            raise PermissionError(
                f"HTTP {resp.status_code} from {url}. This endpoint required no "
                "authentication when the pipeline was built. If it now does, STOP and "
                "resolve the credential question before going further --- do not "
                "substitute another source silently."
            )
        resp.raise_for_status()
        if not resp.content:
            raise DataSourceError(f"empty response from {url}")
        return resp.content

    return with_retry(
        _once,
        policy=policy,
        what=f"GET {url}",
        give_up_on=(PermissionError,),
    )


# ---------------------------------------------------------------------------
# NSE reference data
# ---------------------------------------------------------------------------


def _save_raw_bytes(content: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def fetch_fo_underlyings(
    *, save_to: Path | None = None, policy: RetryPolicy | None = None
) -> pd.DataFrame:
    """The NSE F&O eligible list --- individual securities only.

    Source: ``fo_mktlots.csv``, the published market-lot file. It carries every
    underlying on which derivatives are available, which is exactly the
    universe the spec wants (Sec 6.1: restricting to F&O names is the cleanest
    way to make the short leg implementable).

    The file contains six index rows and one embedded section-header row
    (``Derivatives on Individual Securities``); those are dropped.

    Returns a frame with ``symbol`` and ``underlying``.
    """
    content = http_get(NSE_FO_LOTS_URL, policy=policy)
    if save_to is not None:
        _save_raw_bytes(content, save_to)

    df = pd.read_csv(io.BytesIO(content))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "UNDERLYING" not in df.columns:
        raise DataSourceError(
            f"{NSE_FO_LOTS_URL} no longer has SYMBOL/UNDERLYING columns; got {list(df.columns)}. "
            "The file format changed --- inspect it before trusting the universe."
        )

    df["symbol"] = df["SYMBOL"].astype(str).str.strip()
    df["underlying"] = df["UNDERLYING"].astype(str).str.strip()
    stocks = df[~df["symbol"].str.upper().isin(_FO_NON_STOCK_SYMBOLS)].copy()
    stocks = stocks[stocks["symbol"].str.len() > 0]
    stocks = stocks[["symbol", "underlying"]].drop_duplicates("symbol").reset_index(drop=True)

    if len(stocks) < 100:
        raise DataSourceError(
            f"only {len(stocks)} F&O stock symbols parsed from {NSE_FO_LOTS_URL}; expected "
            "roughly 180-220. The file format probably changed."
        )

    _log.info("NSE F&O list: %d individual securities", len(stocks))
    return stocks


def fetch_index_constituents(
    url: str = NSE_NIFTY500_URL, *, save_to: Path | None = None, policy: RetryPolicy | None = None
) -> pd.DataFrame:
    """Index constituents with NSE's own industry classification.

    The NIFTY 500 file carries ``Industry`` directly, which is where the
    sector map comes from. Using NSE's classification rather than a hand-made
    one matters: the same-sector prior (Sec 4.1) and the same-sector survival
    stratification (Sec 7.2) are only defensible if the grouping came from
    somewhere other than the person hoping to find pairs.

    Returns a frame with ``symbol``, ``company``, ``sector``.
    """
    content = http_get(url, policy=policy)
    if save_to is not None:
        _save_raw_bytes(content, save_to)

    df = pd.read_csv(io.BytesIO(content))
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Symbol", "Industry"}
    if not required.issubset(df.columns):
        raise DataSourceError(
            f"{url} is missing {required - set(df.columns)}; got {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "symbol": df["Symbol"].astype(str).str.strip(),
            "company": df.get("Company Name", pd.Series([""] * len(df))).astype(str).str.strip(),
            "sector": df["Industry"].astype(str).str.strip(),
        }
    ).drop_duplicates("symbol")

    out = out[(out["symbol"] != "") & (out["sector"] != "")].reset_index(drop=True)
    _log.info("index constituents: %d symbols, %d sectors", len(out), out["sector"].nunique())
    return out
