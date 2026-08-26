"""Data-quality engine tests.

Two obligations, and the second matters more than it looks.

**Every check must fire on a planted defect.** A quality engine that reports
nothing is indistinguishable from a clean dataset, and gets trusted for the
wrong reason. So each check gets a frame with exactly the fault it looks for.

**Nothing may be deleted.** The engine's whole contract is that it flags and
explains. There is a test below that runs the full pipeline and asserts the
input frames come out byte-identical, because silently dropping "suspicious"
observations is itself a form of lookahead --- the days that look wrong are
disproportionately the days something real happened, and removing them makes a
backtest smoother and less true.

SYNTHETIC DATA --- METHODOLOGY/UNIT TEST ONLY. The frames here carry planted
faults so the detectors can be checked against a known answer. Tests marked
``realdata`` run against the downloaded NSE panel and skip on a fresh clone.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dsa.data.quality import (
    CHECKS,
    Issue,
    QualityReport,
    Severity,
    check_panel,
    check_ticker,
    infer_sessions,
    listed_span,
    run_quality_checks,
)

pytestmark = pytest.mark.synthetic


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def clean_raw(n: int = 300, start: str = "2020-01-01") -> pd.DataFrame:
    """A well-formed raw frame with no faults. SYNTHETIC — unit test only."""
    idx = pd.bdate_range(start, periods=n, name="Date")
    # A gentle random walk: no 20% days, no flat stretches.
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


def checks_fired(issues: list[Issue]) -> set[str]:
    return {i.check for i in issues}


def severities_for(issues: list[Issue], check: str) -> set[Severity]:
    return {i.severity for i in issues if i.check == check}


@pytest.fixture
def quality_cfg() -> SimpleNamespace:
    """Thresholds matching config/quality.yaml."""
    return SimpleNamespace(
        session_quorum=0.50,
        phantom_date_max_tickers=5,
        extreme_move_pct=0.20,
        severe_move_pct=0.50,
        corporate_action_window_days=3,
        max_gap_sessions=5,
        min_coverage_over_life=0.95,
        max_zero_volume_run=3,
        zero_volume_frac_warn=0.02,
        max_stale_price_run=5,
        cross_check_n_symbols=5,
        cross_check_tolerance_pct=0.02,
        max_issues_listed_per_check=25,
    )


# ===========================================================================
# THE CONTRACT: flag, never delete
# ===========================================================================


def test_checks_do_not_mutate_the_input(quality_cfg):
    """The single most important test in this file.

    Deleting suspicious observations is a form of lookahead: you would be
    choosing which days to keep using knowledge of what happened on them.
    """
    raw = clean_raw()
    raw.iloc[50, raw.columns.get_loc("Close")] *= 1.6  # plant an extreme move
    raw.iloc[80, raw.columns.get_loc("Volume")] = 0.0
    before = raw.copy(deep=True)

    check_ticker("X.NS", raw, extreme_move_pct=0.20)

    pd.testing.assert_frame_equal(raw, before)
    assert len(raw) == len(before), "rows were dropped by a check"


def test_full_pipeline_does_not_mutate_inputs(quality_cfg):
    raw = clean_raw()
    raw.iloc[10, raw.columns.get_loc("Close")] *= 1.5
    frames = {"A.NS": raw}
    close = pd.DataFrame({"A.NS": raw["Adj Close"].copy()})
    close.index = pd.DatetimeIndex(close.index).normalize()
    volume = pd.DataFrame({"A.NS": raw["Volume"].to_numpy()}, index=close.index)

    raw_before = raw.copy(deep=True)
    close_before = close.copy(deep=True)

    report = run_quality_checks(frames, close, volume=volume, quality_cfg=quality_cfg)

    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(close, close_before)
    assert report.issues, "the planted move should have produced at least one issue"


def test_clean_data_produces_no_fatal_or_warn(quality_cfg):
    """No false alarms on a well-formed series."""
    issues = check_ticker("X.NS", clean_raw())
    serious = [i for i in issues if i.severity in (Severity.FATAL, Severity.WARN)]
    assert not serious, f"clean data raised: {[(i.check, i.detail[:60]) for i in serious]}"


# ===========================================================================
# structural faults -> FATAL
# ===========================================================================


def test_duplicate_timestamps_are_fatal():
    raw = clean_raw()
    dup = pd.concat([raw, raw.iloc[[10, 11]]]).sort_index()
    issues = check_ticker("X.NS", dup)
    assert "duplicate_timestamps" in checks_fired(issues)
    assert Severity.FATAL in severities_for(issues, "duplicate_timestamps")


def test_non_monotonic_index_is_fatal():
    raw = clean_raw().iloc[::-1]
    issues = check_ticker("X.NS", raw)
    assert "chronological_order" in checks_fired(issues)
    assert Severity.FATAL in severities_for(issues, "chronological_order")


@pytest.mark.parametrize("bad_value", [0.0, -5.0])
def test_non_positive_prices_are_fatal(bad_value):
    raw = clean_raw()
    raw.iloc[20, raw.columns.get_loc("Close")] = bad_value
    issues = check_ticker("X.NS", raw)
    assert Severity.FATAL in severities_for(issues, "invalid_prices")


def test_high_below_low_is_fatal():
    raw = clean_raw()
    i = raw.columns.get_loc("High")
    raw.iloc[30, i] = raw["Low"].iloc[30] * 0.5
    issues = check_ticker("X.NS", raw)
    assert Severity.FATAL in severities_for(issues, "invalid_prices")
    assert any("High < Low" in i.detail for i in issues)


def test_negative_volume_is_fatal():
    raw = clean_raw()
    raw.iloc[15, raw.columns.get_loc("Volume")] = -100.0
    issues = check_ticker("X.NS", raw)
    assert Severity.FATAL in severities_for(issues, "invalid_prices")


def test_empty_frame_is_fatal():
    issues = check_ticker("X.NS", pd.DataFrame())
    assert Severity.FATAL in {i.severity for i in issues}


# ===========================================================================
# value faults -> WARN
# ===========================================================================


def test_close_outside_high_low_is_flagged():
    raw = clean_raw()
    raw.iloc[40, raw.columns.get_loc("Close")] = raw["High"].iloc[40] * 1.05
    issues = check_ticker("X.NS", raw)
    assert any("Close sits outside" in i.detail for i in issues)


def test_open_outside_high_low_is_flagged_and_says_why_it_matters():
    """The engine fills at the open, so a bad open costs more than a bad close."""
    raw = clean_raw()
    raw.iloc[40, raw.columns.get_loc("Open")] = raw["Low"].iloc[40] * 0.9
    issues = check_ticker("X.NS", raw)
    matching = [i for i in issues if "Open sits outside" in i.detail]
    assert matching
    assert "fills at the open" in matching[0].detail


def test_missing_values_are_flagged():
    raw = clean_raw()
    raw.iloc[25, raw.columns.get_loc("Close")] = np.nan
    issues = check_ticker("X.NS", raw)
    assert "missing_values" in checks_fired(issues)


def test_isolated_zero_volume_is_info_but_a_run_is_a_warning():
    raw = clean_raw()
    raw.iloc[100, raw.columns.get_loc("Volume")] = 0.0
    issues = check_ticker("X.NS", raw, max_zero_volume_run=3)
    assert severities_for(issues, "zero_volume") == {Severity.INFO}

    raw2 = clean_raw()
    raw2.iloc[100:110, raw2.columns.get_loc("Volume")] = 0.0
    issues2 = check_ticker("X.NS", raw2, max_zero_volume_run=3)
    assert Severity.WARN in severities_for(issues2, "zero_volume")
    assert any("consecutive zero-volume" in i.detail for i in issues2)


def test_stale_price_run_is_flagged_with_the_reason_it_matters():
    """A flat stretch reads to an ADF test as a well-behaved spread."""
    raw = clean_raw()
    raw.iloc[50:60, raw.columns.get_loc("Close")] = raw["Close"].iloc[50]
    issues = check_ticker("X.NS", raw, max_stale_price_run=5)
    stale = [i for i in issues if i.check == "stale_price"]
    assert stale
    assert stale[0].severity is Severity.WARN
    assert "ADF" in stale[0].detail


def test_short_stale_run_below_threshold_is_not_flagged():
    raw = clean_raw()
    raw.iloc[50:52, raw.columns.get_loc("Close")] = raw["Close"].iloc[50]
    issues = check_ticker("X.NS", raw, max_stale_price_run=5)
    assert "stale_price" not in checks_fired(issues)


# ===========================================================================
# extreme moves and corporate-action attribution
# ===========================================================================


def test_extreme_move_is_flagged():
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 1.35  # a +35% step
    issues = check_ticker("X.NS", raw, extreme_move_pct=0.20)
    moves = [i for i in issues if i.check == "extreme_move"]
    assert moves
    assert moves[0].value == pytest.approx(0.35, abs=0.01)


def test_move_below_threshold_is_not_flagged():
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 1.10
    issues = check_ticker("X.NS", raw, extreme_move_pct=0.20)
    assert "extreme_move" not in checks_fired(issues)


def test_move_near_a_corporate_action_names_that_action():
    """Reporting the dividend on the flagged bar is useless --- it is usually
    zero, because the action sits a session or two away."""
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 1.35
    raw.iloc[102, raw.columns.get_loc("Dividends")] = 7.5  # two sessions later

    issues = check_ticker("X.NS", raw, extreme_move_pct=0.20, corporate_action_window_days=3)
    moves = [i for i in issues if i.check == "extreme_move"]
    assert moves
    assert moves[0].severity is Severity.WARN
    assert "nearest corporate action" in moves[0].detail
    assert str(raw.index[102].date()) in moves[0].detail
    assert "7.50" in moves[0].detail


def test_market_wide_move_is_downgraded_but_isolated_move_is_not():
    """A 30% drop on a day a third of the market fell is a crash.
    The same drop on a quiet day is a name-specific event or a bad print."""
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 0.75  # -25%
    date = raw.index[100]

    calm = pd.Series(0.0, index=raw.index)
    crash = pd.Series(0.0, index=raw.index)
    crash.loc[date] = 0.40

    isolated = [
        i
        for i in check_ticker("X.NS", raw, extreme_move_pct=0.20, market_stress=calm)
        if i.check == "extreme_move"
    ]
    wide = [
        i
        for i in check_ticker("X.NS", raw, extreme_move_pct=0.20, market_stress=crash)
        if i.check == "extreme_move"
    ]

    assert isolated[0].severity is Severity.WARN
    assert "Isolated" in isolated[0].detail
    assert wide[0].severity is Severity.INFO
    assert "market-wide" in wide[0].detail


def test_severe_move_stays_a_warning_even_on_a_market_wide_day():
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 0.35  # -65%
    crash = pd.Series(0.0, index=raw.index)
    crash.loc[raw.index[100]] = 0.50
    issues = check_ticker("X.NS", raw, extreme_move_pct=0.20, severe_move_pct=0.50,
                          market_stress=crash)
    moves = [i for i in issues if i.check == "extreme_move"]
    assert moves[0].severity is Severity.WARN


# ===========================================================================
# session inference
# ===========================================================================


def _panel(n_symbols: int = 10, n_days: int = 100):
    idx = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    cols = [f"S{i}.NS" for i in range(n_symbols)]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    close += np.arange(n_days).reshape(-1, 1) * 0.1
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=cols)
    return close, volume


def test_all_trading_days_are_sessions():
    close, volume = _panel()
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    assert len(sessions) == len(close.index)
    assert len(padded) == 0


def test_padded_holiday_is_detected_and_excluded():
    """The real bug this catches: Yahoo pads NSE holidays with a carried-forward
    close and zero volume. Those rows have a price, so a notna()-based quorum
    passes them, and each one inserts a zero return into every series."""
    close, volume = _panel()
    holiday = close.index[50]
    volume.loc[holiday] = 0.0
    close.loc[holiday] = close.iloc[49]  # carried forward, zero change

    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    assert holiday in padded
    assert holiday not in sessions
    assert len(sessions) == len(close.index) - 1
    assert participation.loc[holiday] == 0.0


def test_without_volume_a_padded_holiday_is_invisible():
    """Documents exactly why participation must be measured by trading.

    This is the behaviour of the first, buggy version. Pinned so that anyone
    tempted to simplify infer_sessions back to notna() sees what it costs.
    """
    close, volume = _panel()
    holiday = close.index[50]
    volume.loc[holiday] = 0.0
    close.loc[holiday] = close.iloc[49]

    _, _, padded_without = infer_sessions(close, None, quorum=0.5)
    _, _, padded_with = infer_sessions(close, volume, quorum=0.5)

    assert len(padded_without) == 0, "a price-only check cannot see a padded day"
    assert holiday in padded_with


def test_a_date_below_quorum_is_not_a_session():
    close, volume = _panel(n_symbols=10)
    odd = close.index[30]
    close.loc[odd, close.columns[1:]] = np.nan  # only one ticker prints
    volume.loc[odd, volume.columns[1:]] = 0.0
    sessions, _, _ = infer_sessions(close, volume, quorum=0.5)
    assert odd not in sessions


def test_young_listings_do_not_drag_participation_down():
    """A 2024 IPO must not make every 2015 date fail quorum."""
    close, volume = _panel(n_symbols=4, n_days=100)
    close.iloc[:80, 3] = np.nan  # S3 lists late
    volume.iloc[:80, 3] = np.nan
    sessions, participation, _ = infer_sessions(close, volume, quorum=0.75)
    assert len(sessions) == len(close.index)
    assert participation.iloc[0] == pytest.approx(1.0)


def test_listed_span():
    idx = pd.bdate_range("2020-01-01", periods=10)
    series = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, np.nan, np.nan, np.nan],
                       index=idx)
    first, last = listed_span(series)
    assert first == idx[2]
    assert last == idx[6]
    assert listed_span(pd.Series(dtype=float)) == (None, None)


# ===========================================================================
# panel checks
# ===========================================================================


def test_gaps_and_coverage_are_measured_over_the_listed_life():
    close, volume = _panel(n_symbols=4, n_days=200)
    close.iloc[:100, 3] = np.nan  # S3 lists halfway through
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)

    issues, per_ticker = check_panel(
        close, sessions=sessions, participation=participation, padded=padded
    )
    row = per_ticker.set_index("symbol").loc["S3.NS"]
    assert row["coverage"] == pytest.approx(1.0), (
        "a late listing must be 100% covered over its own life, not 50% over the panel"
    )
    assert row["n_missing"] == 0


def test_a_long_gap_is_flagged():
    close, volume = _panel(n_symbols=4, n_days=200)
    close.iloc[100:120, 0] = np.nan  # a 20-session suspension
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    issues, per_ticker = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        max_gap_sessions=5,
    )
    gaps = [i for i in issues if i.check == "gaps" and i.symbol == "S0.NS"]
    assert gaps
    assert gaps[0].n_affected == 20
    assert "forward-filling" in gaps[0].detail


def test_low_coverage_is_flagged():
    close, volume = _panel(n_symbols=4, n_days=200)
    rng = np.random.default_rng(1)
    holes = rng.choice(200, size=40, replace=False)
    close.iloc[holes, 0] = np.nan
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        min_coverage_over_life=0.95,
    )
    assert any(i.check == "coverage" and i.symbol == "S0.NS" for i in issues)


def test_padded_session_issue_explains_the_consequence():
    close, volume = _panel()
    holiday = close.index[50]
    volume.loc[holiday] = 0.0
    close.loc[holiday] = close.iloc[49]
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)

    issues, _ = check_panel(close, sessions=sessions, participation=participation, padded=padded)
    padded_issues = [i for i in issues if i.check == "padded_session"]
    assert len(padded_issues) == 1
    assert "NOT DELETED" in padded_issues[0].detail
    assert "mean-reverting" in padded_issues[0].detail


def test_phantom_session_is_flagged():
    close, volume = _panel(n_symbols=20, n_days=100)
    odd = close.index[30]
    close.loc[odd, close.columns[2:]] = np.nan  # only 2 tickers print
    volume.loc[odd, volume.columns[2:]] = np.nan
    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        phantom_date_max_tickers=5,
    )
    assert any(i.check == "phantom_session" for i in issues)


def test_universe_ticker_missing_from_panel_is_fatal():
    close, volume = _panel(n_symbols=3)
    sessions, participation, padded = infer_sessions(close, volume)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        universe=["S0.NS", "S1.NS", "S2.NS", "GHOST.NS"],
    )
    fatal = [i for i in issues if i.check == "ticker_consistency" and i.severity is Severity.FATAL]
    assert fatal
    assert "GHOST.NS" in fatal[0].detail


def test_extra_panel_columns_are_info_not_an_error():
    """Candidates downloaded then filtered out are expected, not a fault."""
    close, volume = _panel(n_symbols=3)
    sessions, participation, padded = infer_sessions(close, volume)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        universe=["S0.NS"],
    )
    extra = [i for i in issues if i.check == "ticker_consistency"]
    assert extra and extra[0].severity is Severity.INFO


def test_panel_column_without_a_manifest_entry_is_flagged():
    close, volume = _panel(n_symbols=3)
    sessions, participation, padded = infer_sessions(close, volume)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        manifest_symbols=["S0.NS"],
    )
    orphan = [i for i in issues if "manifest" in i.detail]
    assert orphan
    assert orphan[0].severity is Severity.WARN


# ===========================================================================
# the report object
# ===========================================================================


def test_report_is_reproducible(quality_cfg):
    """Same input, same findings --- a report that drifts cannot be cited."""
    raw = clean_raw()
    raw.iloc[100:, raw.columns.get_loc("Close")] *= 1.4
    frames = {"A.NS": raw}
    close = pd.DataFrame({"A.NS": raw["Adj Close"].to_numpy()},
                         index=pd.DatetimeIndex(raw.index).normalize())
    volume = pd.DataFrame({"A.NS": raw["Volume"].to_numpy()}, index=close.index)

    a = run_quality_checks(frames, close, volume=volume, quality_cfg=quality_cfg)
    b = run_quality_checks(frames, close, volume=volume, quality_cfg=quality_cfg)
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())


def test_report_counts_and_serialisation(quality_cfg, tmp_path):
    report = QualityReport(generated_at="2026-08-25T00:00:00", config_hash="abc123")
    report.add(Issue("extreme_move", Severity.WARN, symbol="A.NS", detail="x", n_affected=1))
    report.add(Issue("zero_volume", Severity.INFO, symbol="A.NS", detail="y", n_affected=3))
    report.per_ticker = pd.DataFrame(
        [dict(symbol="A.NS", first="2020-01-01", last="2020-12-31", n_sessions=250,
              n_observed=250, coverage=1.0, n_missing=0, max_gap=0, n_gaps=0)]
    )

    counts = report.counts()
    assert set(counts["check"]) == {"extreme_move", "zero_volume"}
    assert int(counts.set_index("check").loc["zero_volume", "total"]) == 3

    paths = report.write(tmp_path)
    for path in paths.values():
        assert path.is_file()
    assert "Data quality report" in paths["summary"].read_text(encoding="utf-8")
    assert "has been deleted, repaired or interpolated" in paths["summary"].read_text(
        encoding="utf-8"
    )


def test_is_usable_tracks_fatal_findings():
    report = QualityReport()
    assert report.is_usable
    report.add(Issue("invalid_prices", Severity.FATAL, detail="negative price"))
    assert not report.is_usable


def test_every_check_that_can_fire_is_documented():
    """A report that describes a check it never runs, or omits one it does,
    is worse than no report."""
    fired = set()
    raw = clean_raw()
    raw.iloc[10, raw.columns.get_loc("Close")] = -1.0
    raw.iloc[20, raw.columns.get_loc("Volume")] = 0.0
    raw.iloc[50:60, raw.columns.get_loc("Close")] = 50.0
    fired |= checks_fired(check_ticker("X.NS", raw))

    close, volume = _panel()
    volume.loc[close.index[10]] = 0.0
    close.loc[close.index[10]] = close.iloc[9]
    sessions, participation, padded = infer_sessions(close, volume)
    issues, _ = check_panel(
        close, sessions=sessions, participation=participation, padded=padded,
        universe=["S0.NS", "MISSING.NS"],
    )
    fired |= checks_fired(issues)

    undocumented = fired - set(CHECKS)
    assert not undocumented, f"checks fire but are not in the CHECKS registry: {undocumented}"


# ===========================================================================
# REAL DATA
# ===========================================================================


@pytest.mark.realdata
def test_real_panel_has_no_fatal_findings(cfg):
    """The gate: nothing structurally impossible in the downloaded panel."""
    from dsa.data.clean import load_panel
    from dsa.data.store import raw_ohlcv_path, read_parquet

    try:
        close = load_panel("close")
        volume = load_panel("volume")
    except FileNotFoundError:
        pytest.skip("no clean panel; run scripts/build_dataset.py")

    symbols = [s for s in cfg.universe.tickers if s in close.columns][:25]
    frames = {}
    for symbol in symbols:
        path = raw_ohlcv_path(symbol)
        if path.is_file():
            frames[symbol] = read_parquet(path)
    if not frames:
        pytest.skip("no raw frames on disk")

    report = run_quality_checks(
        frames, close[list(frames)], volume=volume[list(frames)], quality_cfg=cfg.quality
    )
    assert report.is_usable, (
        "FATAL findings in the real panel: "
        f"{[(i.symbol, i.detail[:80]) for i in report.fatal[:5]]}"
    )


@pytest.mark.realdata
def test_real_panel_padded_holidays_are_found(cfg):
    """Regression guard for the bug this engine caught.

    If a future change reverts participation to a price-only check, these
    padded exchange holidays go back to counting as trading days and every
    spread quietly gains five zero-return observations.
    """
    from dsa.data.clean import load_panel

    try:
        close = load_panel("close")
        volume = load_panel("volume")
    except FileNotFoundError:
        pytest.skip("no clean panel; run scripts/build_dataset.py")

    sessions, _, padded = infer_sessions(close, volume, quorum=cfg.quality.session_quorum)
    assert len(padded) >= 1, (
        "no padded non-trading days detected in the real panel. Five were present when the "
        "engine was written; if they are genuinely gone, confirm it before relaxing this."
    )
    assert len(sessions) == len(close.index) - len(padded)

    # On a padded day, almost nothing trades. Not *nothing*: 2025-03-18 in this
    # panel is a PARTIAL provider failure --- 201 of 203 names got a padded row
    # while GVT&D and ITC got real data. That is why the threshold is a share
    # rather than an absolute zero.
    for date in padded:
        traded = (volume.loc[date].fillna(0) > 0).sum()
        listed = close.loc[date].notna().sum()
        assert traded / max(listed, 1) <= 0.10, (
            f"{date.date()}: {traded}/{listed} tickers traded, too many for a padded day"
        )


def test_tradeable_coverage_separates_supplied_bars_from_traded_ones():
    """A name can show 100% coverage and still have been untradeable.

    `coverage` asks whether the provider supplied a bar; `tradeable_coverage`
    asks whether anything changed hands. Reporting only the first would let a
    three-week suspension read as a perfectly complete series.
    """
    close, volume = _panel(n_symbols=3, n_days=200)
    volume.iloc[100:120, 0] = 0.0  # S0 halted for 20 sessions, prices still printed

    sessions, participation, padded = infer_sessions(close, volume, quorum=0.5)
    _, per_ticker = check_panel(
        close, sessions=sessions, participation=participation, padded=padded, volume=volume
    )
    row = per_ticker.set_index("symbol").loc["S0.NS"]

    assert row["coverage"] == pytest.approx(1.0), "every bar was supplied"
    assert row["n_zero_volume"] == 20
    assert row["tradeable_coverage"] == pytest.approx(180 / 200)

    clean_row = per_ticker.set_index("symbol").loc["S1.NS"]
    assert clean_row["tradeable_coverage"] == pytest.approx(1.0)
