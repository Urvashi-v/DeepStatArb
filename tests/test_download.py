"""Download logic: retry, backoff, rate limiting, resume.

All of these run against an **injected fetcher**, not the network. That is
deliberate. The behaviour under test is "what does this code do when the feed
misbehaves", and you cannot ask a live feed to fail on demand, fail twice then
succeed, or return a malformed schema. Tests that need the real feed are in
``test_data_connection.py`` and carry the ``network`` marker.

No synthetic *prices* are used to stand in for market data anywhere. The frames
below exist to exercise control flow; their values are never interpreted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa.data.download import (
    RAW_COLUMNS,
    DownloadConfig,
    PermanentSymbolError,
    download_ticker,
    download_universe,
)
from dsa.data.sources import DataSourceError, RateLimiter, RetryPolicy, with_retry
from dsa.data.store import Manifest, raw_ohlcv_path


@pytest.fixture(autouse=True)
def isolated_project(tmp_path: Path, monkeypatch):
    """Point every project path at a temp tree so tests never touch data/raw."""
    monkeypatch.setenv("DSA_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "clean").mkdir(parents=True)
    return tmp_path


def make_frame(n: int = 600, start: str = "2015-01-01") -> pd.DataFrame:
    """A well-formed response shape. Values are arbitrary and never asserted on."""
    idx = pd.bdate_range(start, periods=n, name="Date")
    close = np.linspace(100, 200, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close * 0.97,
            "Volume": np.full(n, 1_000_000),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )[RAW_COLUMNS]


# ===========================================================================
# retry
# ===========================================================================


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise DataSourceError("HTTP 503")
        return "ok"

    result = with_retry(flaky, policy=RetryPolicy(max_attempts=4, base_delay_s=0), sleep=lambda s: None)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_and_reports_the_last_error():
    def always_fails():
        raise DataSourceError("HTTP 503")

    with pytest.raises(DataSourceError, match="failed after 3 attempts"):
        with_retry(
            always_fails,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0),
            sleep=lambda s: None,
        )


def test_permanent_errors_are_not_retried():
    """An unknown ticker will not become known. Retrying it burns budget the
    recoverable failures need."""
    calls = {"n": 0}

    def unknown_symbol():
        calls["n"] += 1
        raise PermanentSymbolError("possibly delisted")

    with pytest.raises(PermanentSymbolError):
        with_retry(
            unknown_symbol,
            policy=RetryPolicy(max_attempts=5, base_delay_s=0),
            give_up_on=(PermanentSymbolError,),
            sleep=lambda s: None,
        )
    assert calls["n"] == 1


def test_permission_error_is_not_retried():
    """A 401 means the credential situation changed. Stop, do not hammer."""

    def needs_auth():
        raise PermissionError("HTTP 401")

    with pytest.raises(PermissionError):
        with_retry(
            needs_auth,
            policy=RetryPolicy(max_attempts=5, base_delay_s=0),
            give_up_on=(PermissionError,),
            sleep=lambda s: None,
        )


def test_backoff_grows_and_is_bounded():
    policy = RetryPolicy(base_delay_s=1.0, backoff=2.0, max_delay_s=10.0)
    import random

    rng = random.Random(0)
    # Full jitter means each delay is uniform in [0, cap]; check the caps grow.
    caps = [min(policy.max_delay_s, policy.base_delay_s * policy.backoff ** (a - 1)) for a in range(1, 7)]
    assert caps == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
    for attempt in range(1, 7):
        d = policy.delay_for(attempt, rng)
        assert 0.0 <= d <= caps[attempt - 1] + 1e-9


def test_jitter_makes_retries_differ():
    """Without jitter a batch that fails together retries together, recreating
    the burst the limiter exists to prevent."""
    import random

    policy = RetryPolicy(base_delay_s=4.0)
    delays = {policy.delay_for(3, random.Random(seed)) for seed in range(20)}
    assert len(delays) > 15, "delays are not being jittered"


# ===========================================================================
# rate limiting
# ===========================================================================


def test_rate_limiter_enforces_a_minimum_interval():
    import time

    limiter = RateLimiter(min_interval_s=0.05, jitter_s=0.0)
    limiter.wait()
    t0 = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.09, f"two waits took {elapsed:.3f}s, expected >= 0.10s"


def test_rate_limiter_rejects_negative_interval():
    with pytest.raises(ValueError):
        RateLimiter(min_interval_s=-1)


# ===========================================================================
# download_ticker
# ===========================================================================


def test_download_ticker_returns_frame_and_attempt_count():
    df, attempts = download_ticker(
        "X.NS",
        config=DownloadConfig(base_delay_s=0),
        fetcher=lambda s, a, b, t: make_frame(),
    )
    assert len(df) == 600
    assert attempts == 1


def test_download_ticker_retries_then_succeeds():
    state = {"n": 0}

    def flaky(symbol, start, end, timeout):
        state["n"] += 1
        if state["n"] < 2:
            raise DataSourceError("HTTP 502")
        return make_frame()

    df, attempts = download_ticker(
        "X.NS", config=DownloadConfig(base_delay_s=0, max_attempts=3), fetcher=flaky
    )
    assert attempts == 2
    assert not df.empty


def test_download_ticker_never_returns_an_empty_frame_as_success():
    """An empty result must raise, not be stored as though it were data."""

    def empty(symbol, start, end, timeout):
        raise PermanentSymbolError("no rows")

    with pytest.raises(DataSourceError):
        download_ticker("X.NS", config=DownloadConfig(base_delay_s=0), fetcher=empty)


# ===========================================================================
# download_universe: resume, failure isolation, manifest
# ===========================================================================


def test_batch_writes_raw_and_manifest():
    result = download_universe(
        ["A.NS", "B.NS"],
        config=DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0),
        fetcher=lambda s, a, b, t: make_frame(),
    )
    assert result.downloaded == 2
    assert result.ok
    for symbol in ("A.NS", "B.NS"):
        assert raw_ohlcv_path(symbol).is_file()
    assert set(Manifest.load().completed()) == {"A.NS", "B.NS"}


def test_resume_skips_completed_tickers():
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0)
    fetched: list[str] = []

    def counting(symbol, start, end, timeout):
        fetched.append(symbol)
        return make_frame()

    download_universe(["A.NS", "B.NS"], config=cfg, fetcher=counting)
    assert fetched == ["A.NS", "B.NS"]

    fetched.clear()
    result = download_universe(["A.NS", "B.NS", "C.NS"], config=cfg, fetcher=counting)
    assert fetched == ["C.NS"], "resume re-fetched work that was already complete"
    assert result.skipped == 2
    assert result.downloaded == 1


def test_refresh_refetches_everything():
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0)
    fetched: list[str] = []

    def counting(symbol, start, end, timeout):
        fetched.append(symbol)
        return make_frame()

    download_universe(["A.NS"], config=cfg, fetcher=counting)
    fetched.clear()
    download_universe(["A.NS"], config=cfg, refresh=True, fetcher=counting)
    assert fetched == ["A.NS"]


def test_one_bad_ticker_does_not_abort_the_batch():
    """208 names, one delisted: the other 207 must still land."""
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0, max_attempts=2)

    def selective(symbol, start, end, timeout):
        if symbol == "BAD.NS":
            raise PermanentSymbolError("possibly delisted")
        return make_frame()

    result = download_universe(["A.NS", "BAD.NS", "C.NS"], config=cfg, fetcher=selective)
    assert result.downloaded == 2
    assert result.failed == ["BAD.NS"]
    assert not result.ok
    assert raw_ohlcv_path("A.NS").is_file()
    assert raw_ohlcv_path("C.NS").is_file()
    assert not raw_ohlcv_path("BAD.NS").exists()


def test_manifest_is_saved_after_every_ticker():
    """A crash mid-batch must cost one ticker, not the whole run."""
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0)
    seen: list[int] = []

    def crash_on_third(symbol, start, end, timeout):
        if symbol == "C.NS":
            raise KeyboardInterrupt("simulated crash")
        seen.append(1)
        return make_frame()

    with pytest.raises(KeyboardInterrupt):
        download_universe(["A.NS", "B.NS", "C.NS"], config=cfg, fetcher=crash_on_third)

    assert set(Manifest.load().completed()) == {"A.NS", "B.NS"}


def test_failed_ticker_is_retried_on_the_next_run():
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0, max_attempts=1)
    state = {"fail": True}

    def recovers(symbol, start, end, timeout):
        if state["fail"]:
            raise DataSourceError("network down")
        return make_frame()

    first = download_universe(["A.NS"], config=cfg, fetcher=recovers)
    assert first.failed == ["A.NS"]

    state["fail"] = False
    second = download_universe(["A.NS"], config=cfg, fetcher=recovers)
    assert second.downloaded == 1
    assert second.ok


def test_duplicate_symbols_are_fetched_once():
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0)
    fetched: list[str] = []
    download_universe(
        ["A.NS", "A.NS", "B.NS"],
        config=cfg,
        fetcher=lambda s, a, b, t: (fetched.append(s), make_frame())[1],
    )
    assert fetched == ["A.NS", "B.NS"]


def test_short_history_is_stored_but_flagged(caplog):
    cfg = DownloadConfig(min_interval_s=0, jitter_s=0, base_delay_s=0, min_rows=250)
    with caplog.at_level("WARNING"):
        download_universe(["A.NS"], config=cfg, fetcher=lambda s, a, b, t: make_frame(n=60))
    assert raw_ohlcv_path("A.NS").is_file()
    assert any("below min_rows" in r.getMessage() for r in caplog.records)
