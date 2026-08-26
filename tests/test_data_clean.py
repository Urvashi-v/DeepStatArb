"""The clean layer: adjustment correctness, alignment policy, and its limits.

The adjustment is the one place in the data layer where a quiet arithmetic
error would be invisible downstream and fatal to every result. A factor applied
to Close but not Open silently corrupts every overnight return, which is
exactly the return the execution model trades on (spec Sec 6.3).

Tests marked ``realdata`` run against the downloaded NSE panel and skip on a
fresh clone. They are the ones that check the properties actually claimed in
the module docstring --- that Yahoo's Close is split-adjusted, that the factor
is monotone, that turnover survives a split --- against real corporate actions
rather than ones invented to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa.data.clean import CleanError, adjust_ohlcv, build_panels
from dsa.data.store import raw_ohlcv_path
from dsa.paths import data_clean


def raw_frame(
    n: int = 60,
    *,
    tz: str | None = "Asia/Kolkata",
    factor: float | np.ndarray = 0.98,
) -> pd.DataFrame:
    """A well-formed provider response. SYNTHETIC — structure only."""
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date", tz=tz)
    close = np.linspace(100.0, 150.0, n)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Adj Close": close * factor,
            "Volume": np.full(n, 1_000_000.0),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


# ===========================================================================
# the adjustment
# ===========================================================================


def test_one_factor_is_applied_to_every_price_column():
    """Open and Close must carry the SAME adjustment.

    The engine fills at the open of t+1 having signalled on the close of t. If
    those two prices are adjusted differently, the overnight return --- the only
    return the strategy actually earns --- is contaminated on every day after a
    corporate action.
    """
    raw = raw_frame(factor=0.95)
    out = adjust_ohlcv(raw, "X")
    for col, src in (("open", "Open"), ("high", "High"), ("low", "Low")):
        expected = raw[src].to_numpy() * 0.95
        np.testing.assert_allclose(out[col].to_numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(out["close"].to_numpy(), raw["Adj Close"].to_numpy(), rtol=1e-12)


def test_adjustment_preserves_intraday_ordering():
    out = adjust_ohlcv(raw_frame(), "X")
    assert (out["high"] >= out["low"]).all()
    assert (out["high"] >= out["close"]).all()


def test_adjustment_preserves_returns():
    """Scaling by a constant factor must not change any return."""
    raw = raw_frame(factor=0.90)
    out = adjust_ohlcv(raw, "X")
    raw_ret = raw["Close"].pct_change().dropna().to_numpy()
    adj_ret = out["close"].pct_change().dropna().to_numpy()
    np.testing.assert_allclose(raw_ret, adj_ret, rtol=1e-10)


def test_factor_above_one_is_rejected():
    """A back-adjustment factor is a product of (1 - div/price) terms.

    It must lie in (0, 1]. Above 1 means it is not a total-return adjustment
    and cleaning around it would bury whatever is actually wrong.
    """
    with pytest.raises(CleanError, match="adjustment factor outside"):
        adjust_ohlcv(raw_frame(factor=1.5), "X")


def test_absurdly_small_factor_is_rejected():
    with pytest.raises(CleanError, match="adjustment factor outside"):
        adjust_ohlcv(raw_frame(factor=0.001), "X")


def test_non_positive_close_rows_are_dropped_not_divided_by():
    raw = raw_frame()
    raw.iloc[5, raw.columns.get_loc("Close")] = 0.0
    raw.iloc[7, raw.columns.get_loc("Close")] = -1.0
    out = adjust_ohlcv(raw, "X")
    assert len(out) == len(raw) - 2
    assert np.isfinite(out["close"]).all()


def test_all_bad_rows_raises_rather_than_returning_empty():
    raw = raw_frame(n=10)
    raw["Close"] = 0.0
    with pytest.raises(CleanError, match="no usable rows"):
        adjust_ohlcv(raw, "X")


def test_missing_columns_raise():
    raw = raw_frame().drop(columns=["Adj Close"])
    with pytest.raises(CleanError, match="missing"):
        adjust_ohlcv(raw, "X")


def test_turnover_uses_unadjusted_close():
    """Liquidity must be measured in the rupees that actually changed hands."""
    raw = raw_frame(factor=0.5)
    out = adjust_ohlcv(raw, "X")
    expected = raw["Close"].to_numpy() * raw["Volume"].to_numpy()
    np.testing.assert_allclose(out["turnover"].to_numpy(), expected, rtol=1e-12)
    # If turnover had used the adjusted close it would be half the truth.
    assert not np.allclose(out["turnover"].to_numpy(), out["close"] * out["volume"])


# ===========================================================================
# index normalisation
# ===========================================================================


def test_timezone_is_stripped_to_plain_dates():
    """Two tickers with different tz offsets must align on a join."""
    out = adjust_ohlcv(raw_frame(tz="Asia/Kolkata"), "X")
    assert out.index.tz is None
    assert (out.index.normalize() == out.index).all()


def test_naive_index_passes_through():
    out = adjust_ohlcv(raw_frame(tz=None), "X")
    assert out.index.tz is None


def test_duplicate_timestamps_keep_the_last():
    raw = raw_frame(n=10)
    dup = pd.concat([raw, raw.iloc[[3]]])
    out = adjust_ohlcv(dup, "X")
    assert len(out) == 10
    assert out.index.is_unique


def test_output_is_sorted():
    raw = raw_frame(n=20)
    out = adjust_ohlcv(raw.iloc[::-1], "X")
    assert out.index.is_monotonic_increasing


# ===========================================================================
# panel construction
# ===========================================================================


@pytest.fixture
def panel_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DSA_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "data" / "raw" / "ohlcv").mkdir(parents=True)
    (tmp_path / "data" / "clean").mkdir(parents=True)
    return tmp_path


def _seed_raw(
    symbol: str,
    n: int = 600,
    start: str = "2020-01-01",
    drop_positions: slice | None = None,
) -> None:
    """Write a raw parquet. ``drop_positions`` punches an interior hole,
    which is what a trading halt or suspension looks like in a real feed."""
    from dsa.data.store import write_parquet

    idx = pd.bdate_range(start, periods=n, name="Date", tz="Asia/Kolkata")
    close = np.linspace(100.0, 200.0, n)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": np.full(n, 1_000_000.0),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    if drop_positions is not None:
        df = df.drop(df.index[drop_positions])
    write_parquet(df, raw_ohlcv_path(symbol), force=True)


def test_panels_are_written_for_every_field(panel_project):
    _seed_raw("A.NS")
    _seed_raw("B.NS")
    panels, summary = build_panels(["A.NS", "B.NS"], write=True)
    assert summary.n_symbols == 2
    for field in ("open", "high", "low", "close", "volume", "turnover"):
        assert field in panels
        assert list(panels[field].columns) == ["A.NS", "B.NS"]
        assert (data_clean() / f"panel_{field}.parquet").is_file()


def test_missing_bars_are_left_as_nan_not_forward_filled(panel_project):
    """The alignment decision that most affects the cointegration screen.

    Forward-filling a halted stock invents a flat price. A flat stretch reads
    to an ADF test as an unusually well-behaved spread, which manufactures
    exactly the false pairs the FDR control is there to remove. Gaps stay
    visible so Day 3 can decide what to do about them with the statistics in
    front of it.
    """
    # B is suspended for 10 sessions in the middle --- an INTERIOR hole.
    # A leading gap would not test anything: ffill cannot fill leading NaNs
    # either, so a forward-filled panel and a raw one look identical there.
    _seed_raw("A.NS", n=600, start="2020-01-01")
    _seed_raw("B.NS", n=600, start="2020-01-01", drop_positions=slice(100, 110))

    panels, _ = build_panels(["A.NS", "B.NS"], write=False)
    close = panels["close"]

    hole = close.index[100:110]
    assert close.loc[hole, "B.NS"].isna().all(), (
        "the suspended sessions were filled in. Forward-filling a halted stock "
        "invents a flat price, and a flat stretch reads to an ADF test as an "
        "unusually well-behaved spread --- manufacturing the false pairs the FDR "
        "control exists to remove."
    )
    assert close.loc[hole, "A.NS"].notna().all(), "A should be unaffected"
    # And confirm ffill WOULD have changed it, so the assertion above has teeth.
    assert close["B.NS"].ffill().loc[hole].notna().all()


def test_short_series_are_dropped_with_a_reason(panel_project):
    _seed_raw("GOOD.NS", n=600)
    _seed_raw("SHORT.NS", n=100)
    panels, summary = build_panels(["GOOD.NS", "SHORT.NS"], min_rows=500, write=False)
    assert list(panels["close"].columns) == ["GOOD.NS"]
    assert "SHORT.NS" in summary.dropped
    assert "100 rows" in summary.dropped["SHORT.NS"]


def test_absent_raw_file_is_reported_not_silently_skipped(panel_project):
    _seed_raw("A.NS")
    _, summary = build_panels(["A.NS", "MISSING.NS"], write=False)
    assert summary.dropped["MISSING.NS"] == "no raw file"


def test_no_survivors_raises(panel_project):
    _seed_raw("SHORT.NS", n=10)
    with pytest.raises(CleanError, match="no symbols survived"):
        build_panels(["SHORT.NS"], min_rows=500, write=False)


def test_panel_index_is_the_union_of_trading_dates(panel_project):
    _seed_raw("A.NS", n=600, start="2020-01-01")
    _seed_raw("B.NS", n=600, start="2020-06-01")
    panels, _ = build_panels(["A.NS", "B.NS"], write=False)
    index = panels["close"].index
    assert index.is_monotonic_increasing
    assert index.is_unique


# ===========================================================================
# REAL DATA: the claims in the module docstring, checked against real events
# ===========================================================================

pytestmark_real = pytest.mark.realdata


def _require_real(symbol: str) -> pd.DataFrame:
    from dsa.data.store import read_parquet

    path = raw_ohlcv_path(symbol)
    if not path.is_file():
        pytest.skip(f"no downloaded data for {symbol}; run scripts/build_dataset.py")
    return read_parquet(path)


@pytest.mark.realdata
def test_real_yahoo_close_is_already_split_adjusted():
    """The claim the corporate-action check depends on.

    RELIANCE split 2:1 on 2017-09-07. If Yahoo's Close were unadjusted, that
    day would show roughly -50%. It shows about -0.6%, so the provider has
    already applied it --- which means the ">20% single-day move" screen in
    spec Sec 11.2 will NOT surface splits in this feed. Knowing that is the
    difference between a check that works and one that reports nothing and is
    assumed to be passing.
    """
    raw = _require_real("RELIANCE.NS")
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    split_day = pd.Timestamp("2017-09-07")
    if split_day not in raw.index:
        pytest.skip("split date not in the downloaded window")

    assert raw.loc[split_day, "Stock Splits"] == 2.0, "expected the 2:1 split to be recorded"
    move = raw["Close"].pct_change().loc[split_day]
    assert abs(move) < 0.05, (
        f"close moved {move:.1%} on a 2:1 split date. If this is ever near -50%, the feed "
        "stopped pre-adjusting and the whole clean layer needs revisiting."
    )


@pytest.mark.realdata
def test_real_adjustment_factor_is_monotone_and_bounded():
    """A cumulative dividend factor rises to exactly 1 at the last bar."""
    raw = _require_real("RELIANCE.NS")
    out = adjust_ohlcv(raw, "RELIANCE.NS")
    factor = out["adj_factor"]

    assert (factor > 0).all() and (factor <= 1.0 + 1e-9).all()
    assert factor.iloc[-1] == pytest.approx(1.0, abs=1e-6), (
        "the last bar needs no adjustment, so its factor must be 1.0"
    )
    # Non-decreasing, up to float32 storage noise in the feed.
    assert factor.diff().dropna().min() > -1e-6, "factor decreases; that is not a dividend series"


@pytest.mark.realdata
def test_real_turnover_is_continuous_across_a_split():
    """Volume is split-adjusted too, so Close x Volume has no artificial jump."""
    raw = _require_real("RELIANCE.NS")
    out = adjust_ohlcv(raw, "RELIANCE.NS")
    split_day = pd.Timestamp("2017-09-07")
    if split_day not in out.index:
        pytest.skip("split date not in the downloaded window")

    before = out["turnover"].loc[:split_day].tail(20).median()
    after = out["turnover"].loc[split_day:].head(20).median()
    ratio = after / before
    assert 0.3 < ratio < 3.0, (
        f"turnover jumped {ratio:.2f}x across a split. If volume were NOT split-adjusted "
        "while price is, this would sit near 0.5 or 2.0 and the liquidity filter would be "
        "systematically wrong for every pre-split period."
    )


@pytest.mark.realdata
def test_real_panel_matches_the_frozen_universe(cfg):
    """Every frozen ticker must be present in the clean panel."""
    from dsa.data.clean import load_panel

    if not cfg.universe.frozen:
        pytest.skip("universe not frozen yet")
    try:
        close = load_panel("close")
    except FileNotFoundError:
        pytest.skip("no clean panel; run scripts/build_dataset.py")

    missing = [t for t in cfg.universe.tickers if t not in close.columns]
    assert not missing, f"{len(missing)} frozen tickers absent from the panel: {missing[:10]}"


@pytest.mark.realdata
def test_real_prices_are_positive_and_finite():
    from dsa.data.clean import load_panel

    try:
        close = load_panel("close")
    except FileNotFoundError:
        pytest.skip("no clean panel; run scripts/build_dataset.py")

    observed = close.to_numpy()
    observed = observed[~np.isnan(observed)]
    assert (observed > 0).all(), "a non-positive price survived cleaning"
    assert np.isfinite(observed).all()
