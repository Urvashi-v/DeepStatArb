"""Live data-source availability.

Every test here is marked ``network`` and makes real requests. Run them with:

    python -m pytest -m network

and exclude them everywhere else:

    python -m pytest -m "not network"

They exist to answer one question fast --- *are the real sources still there and
still shaped the way the pipeline assumes?* --- because the way a free data
feed breaks is rarely an outage. It is a renamed column, a changed CSV layout,
or an endpoint that quietly starts wanting a session cookie. Those produce a
pipeline that runs to completion and gets the wrong answer.

Nothing in this file falls back to cached or synthetic data. If a source is
unreachable, the test fails and says so.
"""

from __future__ import annotations

import pytest

from dsa.data.download import RAW_COLUMNS, DownloadConfig, download_ticker
from dsa.data.sources import (
    NSE_FO_LOTS_URL,
    NSE_NIFTY500_URL,
    RateLimiter,
    fetch_fo_underlyings,
    fetch_index_constituents,
    http_get,
)

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def limiter() -> RateLimiter:
    """One limiter for the module, so the tests stay polite collectively."""
    return RateLimiter(min_interval_s=0.5, jitter_s=0.3)


# ===========================================================================
# Credentials
# ===========================================================================


def test_no_credentials_are_required_for_nse_reference_files():
    """A 401/403 here is the signal that the credential situation changed.

    ``http_get`` raises PermissionError on those specifically, rather than
    retrying or falling through to something else, so that the failure is
    unmissable rather than absorbed.
    """
    for url in (NSE_FO_LOTS_URL, NSE_NIFTY500_URL):
        content = http_get(url)
        assert content, f"{url} returned an empty body"
        assert len(content) > 1000, f"{url} returned only {len(content)} bytes"


# ===========================================================================
# NSE reference data: shape, not just reachability
# ===========================================================================


def test_fo_list_is_reachable_and_the_right_size():
    fo = fetch_fo_underlyings()
    assert 150 <= len(fo) <= 260, (
        f"{len(fo)} F&O securities. The spec expects 180-220; a number far outside that "
        "means the file layout changed and the index/header rows are no longer being "
        "excluded correctly."
    )
    assert {"symbol", "underlying"} <= set(fo.columns)
    assert fo["symbol"].is_unique


def test_fo_list_excludes_index_derivatives():
    """NIFTY and BANKNIFTY are not stocks and must never enter the universe."""
    fo = fetch_fo_underlyings()
    symbols = set(fo["symbol"].str.upper())
    for index_symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
        assert index_symbol not in symbols
    assert "SYMBOL" not in symbols, "the embedded section-header row leaked through"


def test_sector_map_is_reachable_and_classified():
    idx = fetch_index_constituents()
    assert len(idx) > 400, f"only {len(idx)} constituents"
    assert idx["sector"].nunique() >= 10, "too few distinct sectors to form an economic prior"
    assert (idx["sector"].str.len() > 0).all()


def test_sector_coverage_of_the_fo_universe_is_high():
    """The economic prior is only usable if nearly every name is classified."""
    fo = fetch_fo_underlyings()
    idx = fetch_index_constituents()
    covered = fo["symbol"].isin(set(idx["symbol"])).mean()
    assert covered >= 0.90, (
        f"only {covered:.1%} of F&O names have a sector. Below this the same-sector "
        "restriction (Sec 4.1) and the survival stratification (Sec 7.2) lose their basis."
    )


# ===========================================================================
# Price feed
# ===========================================================================


def test_price_feed_returns_plausible_nse_data(limiter: RateLimiter):
    df, attempts = download_ticker(
        "RELIANCE.NS",
        config=DownloadConfig(start="2024-01-01", end="2024-03-01"),
        limiter=limiter,
    )
    assert list(df.columns) == RAW_COLUMNS
    assert 30 <= len(df) <= 45, f"{len(df)} bars for two months; expected roughly 40"
    assert (df["Close"] > 0).all()
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] >= 0).all()
    assert attempts >= 1


def test_adjustment_columns_are_present_and_coherent(limiter: RateLimiter):
    """Adj Close / Close must be a valid total-return factor in (0, 1]."""
    df, _ = download_ticker(
        "RELIANCE.NS",
        config=DownloadConfig(start="2020-01-01", end="2024-01-01"),
        limiter=limiter,
    )
    factor = df["Adj Close"] / df["Close"]
    assert (factor > 0).all()
    assert (factor <= 1.0 + 1e-9).all(), (
        f"factor max {factor.max():.6f} exceeds 1. The feed is no longer returning a "
        "back-adjusted Adj Close and the clean layer's assumption is broken."
    )


def test_corporate_actions_are_reported(limiter: RateLimiter):
    """Dividends and splits are what the Day 3 quality gate checks against."""
    df, _ = download_ticker(
        "RELIANCE.NS",
        config=DownloadConfig(start="2015-01-01", end="2024-01-01"),
        limiter=limiter,
    )
    n_events = int(((df["Dividends"] != 0) | (df["Stock Splits"] != 0)).sum())
    assert n_events > 0, (
        "no dividends or splits reported over nine years for a large-cap Indian "
        "name. The actions feed is empty, and every corporate-action check "
        "downstream would silently pass."
    )


def test_full_history_spans_the_configured_window(limiter: RateLimiter, cfg):
    df, _ = download_ticker(
        "RELIANCE.NS",
        config=DownloadConfig(start=str(cfg.universe.start_date), end=str(cfg.universe.end_date)),
        limiter=limiter,
    )
    years = (df.index.max() - df.index.min()).days / 365.25
    assert years >= 9, f"only {years:.1f} years available; the walk-forward scheme needs ~10"
    per_year = len(df) / years
    assert 200 <= per_year <= 270, (
        f"{per_year:.0f} bars per year. NSE trades roughly 250 days; far outside that "
        "suggests duplicated or missing sessions."
    )


def test_unknown_symbol_fails_loudly_rather_than_returning_nothing(limiter: RateLimiter):
    """The alternative --- an empty frame treated as data --- is how a hole
    becomes a silent zero somewhere downstream."""
    from dsa.data.sources import DataSourceError

    with pytest.raises((DataSourceError, Exception)):
        download_ticker(
            "DEFINITELYNOTATICKER123.NS",
            config=DownloadConfig(start="2024-01-01", end="2024-02-01", max_attempts=1),
            limiter=limiter,
        )


def test_index_and_vix_symbols_resolve(limiter: RateLimiter, cfg):
    """NIFTY 50 and India VIX feed the ML filter's market-context features."""
    for symbol in (cfg.universe.benchmark, cfg.universe.vix_symbol):
        df, _ = download_ticker(
            symbol,
            config=DownloadConfig(start="2024-01-01", end="2024-03-01"),
            limiter=limiter,
        )
        assert len(df) > 20, f"{symbol} returned only {len(df)} bars"
        assert (df["Close"] > 0).all()
