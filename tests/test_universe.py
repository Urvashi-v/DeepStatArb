"""Research universe: point-in-time selection, determinism, versioning.

The centrepiece is ``test_eligibility_is_unchanged_by_future_data``. Everything
else here is scaffolding around that one property.

Why that property is the whole point
------------------------------------
The first version of this filter computed median turnover over the entire
2015-2026 sample and used it to admit a name to the 2015 universe. A stock that
was thin until 2022 passed on its full-sample median and would then have been
traded in 2015, which could not have been known at the time. That is the
universe-construction analogue of the full-sample z-score (spec Sec 2.5): it is
invisible in a backtest because, like every leak worth catching, it makes the
result better rather than worse.

The test appends future rows to the panel and requires that every eligibility
decision already made comes back bit-identical.

SYNTHETIC DATA --- METHODOLOGY/UNIT TEST ONLY. Panels here are constructed so a
known answer exists. Tests marked ``realdata`` run against the downloaded NSE
panel and skip on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa.universe import (
    REASON_CODES,
    EligibilityCriteria,
    FrozenUniverse,
    SelectionError,
    build_report,
    decision_dates,
    eligibility_at,
    eligibility_schedule,
    liquidity_stats_at,
    next_version,
    select_universe,
    universe_hash,
    write_report,
)

pytestmark = pytest.mark.synthetic


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def make_panels(
    n_days: int = 1500,
    start: str = "2015-01-01",
    symbols: tuple[str, ...] = ("A.NS", "B.NS", "C.NS"),
    turnover_inr: float = 5e8,
):
    """Three well-behaved names. SYNTHETIC — structure only."""
    idx = pd.bdate_range(start, periods=n_days, name="date")
    close = pd.DataFrame(
        {s: np.linspace(100.0, 300.0, n_days) for s in symbols}, index=idx
    )
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=list(symbols))
    turnover = pd.DataFrame(turnover_inr, index=idx, columns=list(symbols))
    return close, turnover, volume


def criteria(**overrides) -> EligibilityCriteria:
    base = dict(
        min_history_years=3.0,
        min_median_turnover_inr=2.5e8,
        max_missing_frac=0.02,
        min_price_inr=20.0,
        lookback_sessions=252,
    )
    base.update(overrides)
    return EligibilityCriteria(**base)


# ===========================================================================
# THE CAUSALITY CONTRACT
# ===========================================================================


def test_eligibility_is_unchanged_by_future_data():
    """The single most important test in this file.

    Selection decisions made at T must not move when data after T arrives. If
    they do, the universe was chosen with knowledge it could not have had, and
    every backtest built on it is measuring hindsight.
    """
    close, turnover, volume = make_panels(n_days=1500)
    as_of = close.index[1000]

    full = eligibility_at(
        as_of, close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    truncated = eligibility_at(
        as_of,
        close=close.loc[:as_of],
        turnover=turnover.loc[:as_of],
        volume=volume.loc[:as_of],
        criteria=criteria(),
    )
    pd.testing.assert_frame_equal(full, truncated)


def test_a_name_that_becomes_liquid_later_is_ineligible_earlier():
    """The exact bug the first version had, reproduced as a test.

    LATE.NS trades Rs 5cr a day until 2021 and Rs 100cr afterwards. Its
    FULL-SAMPLE median clears a Rs 25cr floor comfortably, so a full-sample
    filter would admit it for the whole period --- including 2018, when it was
    a tenth of the size.
    """
    # 1100 thin sessions then 1500 liquid ones, so the FULL-SAMPLE median is
    # high while the trailing window at the early date is entirely thin.
    close, turnover, volume = make_panels(n_days=2600, symbols=("LATE.NS",))
    switch = close.index[1100]
    turnover.loc[:switch, "LATE.NS"] = 5e7  # Rs 5cr
    turnover.loc[switch:, "LATE.NS"] = 1e9  # Rs 100cr

    full_sample_median = turnover["LATE.NS"].median()
    assert full_sample_median >= 2.5e8, "the fixture must be one a naive filter would admit"

    # Index 1000: ~4 years of history (so the history gate is satisfied) and a
    # trailing 252-session window lying wholly inside the thin period.
    early = eligibility_at(
        close.index[1000], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    late = eligibility_at(
        close.index[2500], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert not early.loc["LATE.NS", "eligible"]
    assert "illiquid" in early.loc["LATE.NS", "reasons"]
    assert late.loc["LATE.NS", "eligible"]


def test_history_is_measured_to_the_decision_date_not_the_sample_end():
    """A 2021 listing must be ineligible in 2018 and eligible in 2026."""
    close, turnover, volume = make_panels(n_days=2000)
    close.iloc[:1200, close.columns.get_loc("C.NS")] = np.nan  # C lists late

    early = eligibility_at(
        close.index[1300], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    late = eligibility_at(
        close.index[1999], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert "short_history" in early.loc["C.NS", "reasons"]
    assert early.loc["C.NS", "history_years"] < 3.0
    assert late.loc["C.NS", "history_years"] > early.loc["C.NS", "history_years"]


def test_stats_never_read_beyond_the_decision_date():
    """Corrupting the future must not change a statistic computed in the past."""
    close, turnover, volume = make_panels(n_days=1500)
    as_of = close.index[800]
    before = liquidity_stats_at(
        as_of, close=close, turnover=turnover, volume=volume, lookback_sessions=252
    )

    close2, turnover2, volume2 = close.copy(), turnover.copy(), volume.copy()
    close2.iloc[900:] = 1e6
    turnover2.iloc[900:] = 1e12
    volume2.iloc[900:] = 0.0

    after = liquidity_stats_at(
        as_of, close=close2, turnover=turnover2, volume=volume2, lookback_sessions=252
    )
    pd.testing.assert_frame_equal(before, after)


def test_schedule_decisions_are_each_independent_of_later_dates():
    """Every row of the schedule must match a standalone run at its own date."""
    close, turnover, volume = make_panels(n_days=2000)
    dates = [close.index[800], close.index[1200], close.index[1800]]

    schedule = eligibility_schedule(
        dates, close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    for as_of in dates:
        standalone = eligibility_at(
            as_of, close=close, turnover=turnover, volume=volume, criteria=criteria()
        ).reset_index()
        rows = schedule[schedule["as_of"] == str(as_of.date())].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            rows[["symbol", "eligible", "reasons"]],
            standalone[["symbol", "eligible", "reasons"]],
        )


# ===========================================================================
# liquidity
# ===========================================================================


def test_turnover_median_uses_traded_sessions_only():
    """A zero-volume halt is not a session you could have traded."""
    close, turnover, volume = make_panels(n_days=800)
    # Half the trailing window is halted, with an absurd turnover printed.
    volume.iloc[-126:, 0] = 0.0
    turnover.iloc[-126:, 0] = 1e12

    stats = liquidity_stats_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, lookback_sessions=252
    )
    assert stats.loc["A.NS", "median_turnover_inr"] == pytest.approx(5e8), (
        "the halted sessions leaked into the liquidity median"
    )
    assert stats.loc["A.NS", "n_traded_in_window"] == 126


def test_illiquid_name_is_excluded_with_a_reason():
    close, turnover, volume = make_panels()
    turnover["B.NS"] = 1e7  # Rs 1cr
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert not result.loc["B.NS", "eligible"]
    assert result.loc["B.NS", "reasons"] == "illiquid"
    assert result.loc["A.NS", "eligible"]


def test_penny_stock_is_excluded():
    close, turnover, volume = make_panels()
    close["B.NS"] = 5.0
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert "low_price" in result.loc["B.NS", "reasons"]


def test_poor_coverage_is_excluded():
    close, turnover, volume = make_panels()
    volume.iloc[-60:, close.columns.get_loc("B.NS")] = 0.0  # halted 60/252
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert "poor_coverage" in result.loc["B.NS", "reasons"]


def test_a_name_with_no_data_is_reported_not_crashed_on():
    close, turnover, volume = make_panels()
    close["B.NS"] = np.nan
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, criteria=criteria()
    )
    assert result.loc["B.NS", "reasons"] == "no_data"
    assert not result.loc["B.NS", "eligible"]


def test_a_young_listing_is_judged_on_its_own_window_not_blamed_for_before_it_existed():
    close, turnover, volume = make_panels(n_days=2000)
    close.iloc[:1900, close.columns.get_loc("C.NS")] = np.nan  # 100 sessions old
    stats = liquidity_stats_at(
        close.index[-1], close=close, turnover=turnover, volume=volume, lookback_sessions=252
    )
    assert stats.loc["C.NS", "n_sessions_in_window"] == 100
    assert stats.loc["C.NS", "missing_frac"] == 0.0, (
        "a 100-session-old name must not be charged for the 152 sessions before it listed"
    )


# ===========================================================================
# decision dates
# ===========================================================================


def test_decision_dates_follow_the_walk_forward_schedule():
    sessions = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2026-08-01"))
    dates = decision_dates(sessions, formation_months=36, trading_months=6, step_months=6)
    assert len(dates) >= 15
    assert all(d in sessions for d in dates), "every decision date must be a real session"
    assert dates == sorted(dates)

    # Steps are ~6 months, but a decision date is snapped BACK to the last real
    # session on or before its boundary. When that boundary is a holiday the
    # snap can cross a month end, so the calendar-month difference is not
    # always exactly 6. The invariant that actually matters is the elapsed
    # time, which must stay close to half a year.
    # strict=False: dates[1:] is one shorter by construction, which is the
    # point of pairing consecutive elements.
    gaps = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
    assert all(150 <= g <= 200 for g in gaps), f"steps are not ~6 months apart: {gaps}"


def test_first_decision_date_is_after_a_full_formation_window():
    sessions = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2026-08-01"))
    dates = decision_dates(sessions, formation_months=36, trading_months=6, step_months=6)
    assert (dates[0] - sessions[0]).days >= 36 * 30


def test_no_decision_dates_when_the_sample_is_too_short():
    sessions = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2016-01-01"))
    assert decision_dates(sessions, formation_months=36, trading_months=6, step_months=6) == []


def test_empty_sessions_raise():
    with pytest.raises(SelectionError, match="no sessions"):
        decision_dates(
            pd.DatetimeIndex([]), formation_months=36, trading_months=6, step_months=6
        )


# ===========================================================================
# determinism
# ===========================================================================


def test_selection_is_deterministic():
    close, turnover, volume = make_panels(n_days=2000)
    sectors = {"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"}
    dates = [close.index[1200], close.index[1900]]

    kwargs = dict(
        close=close, turnover=turnover, volume=volume, sectors=sectors,
        criteria=criteria(), dates=dates,
    )
    a = select_universe(**kwargs)
    b = select_universe(**kwargs)
    assert a.symbols == b.symbols
    assert universe_hash(a.symbols, a.sectors) == universe_hash(b.symbols, b.sectors)


def test_column_order_does_not_change_the_result():
    """A universe that depends on dict ordering is not reproducible."""
    close, turnover, volume = make_panels(n_days=2000)
    sectors = {"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"}
    dates = [close.index[1900]]

    forward = select_universe(
        close=close, turnover=turnover, volume=volume, sectors=sectors,
        criteria=criteria(), dates=dates,
    )
    reversed_cols = list(close.columns)[::-1]
    backward = select_universe(
        close=close[reversed_cols], turnover=turnover[reversed_cols],
        volume=volume[reversed_cols], sectors=sectors, criteria=criteria(), dates=dates,
    )
    assert forward.symbols == backward.symbols


def test_output_is_sorted():
    close, turnover, volume = make_panels(n_days=2000, symbols=("Z.NS", "A.NS", "M.NS"))
    sectors = {s: "IT" for s in close.columns}
    result = select_universe(
        close=close, turnover=turnover, volume=volume, sectors=sectors,
        criteria=criteria(), dates=[close.index[1900]],
    )
    assert list(result.symbols) == sorted(result.symbols)


def test_per_date_cap_is_deterministic_and_breaks_ties_alphabetically():
    """When turnover ties, the choice must not depend on the sort algorithm."""
    close, turnover, volume = make_panels(
        n_days=1500, symbols=("D.NS", "A.NS", "C.NS", "B.NS")
    )
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume,
        criteria=criteria(), max_eligible=2,
    )
    chosen = sorted(result.index[result["eligible"]])
    assert chosen == ["A.NS", "B.NS"], f"ties must break alphabetically, got {chosen}"
    assert (result.loc[["C.NS", "D.NS"], "reasons"] == "below_daily_rank").all()


def test_per_date_cap_keeps_the_most_liquid():
    close, turnover, volume = make_panels(n_days=1500, symbols=("A.NS", "B.NS", "C.NS"))
    turnover["A.NS"] = 1e9
    turnover["B.NS"] = 5e8
    turnover["C.NS"] = 3e8
    result = eligibility_at(
        close.index[-1], close=close, turnover=turnover, volume=volume,
        criteria=criteria(), max_eligible=2,
    )
    assert sorted(result.index[result["eligible"]]) == ["A.NS", "B.NS"]


# ===========================================================================
# structural gates and exclusion reasons
# ===========================================================================


def _select(sectors=None, **kw):
    close, turnover, volume = make_panels(n_days=2000)
    sectors = sectors or {"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"}
    return select_universe(
        close=close, turnover=turnover, volume=volume, sectors=sectors,
        criteria=criteria(), dates=[close.index[1200], close.index[1900]], **kw
    )


def test_missing_sector_excludes():
    result = _select(sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": ""})
    assert "C.NS" not in result.symbols
    assert "no_sector" in result.excluded.set_index("symbol").loc["C.NS", "reasons"]


def test_incomplete_download_excludes():
    result = _select(complete_downloads={"A.NS", "B.NS"})
    assert "C.NS" not in result.symbols
    assert "incomplete_download" in result.excluded.set_index("symbol").loc["C.NS", "reasons"]


def test_quality_fatal_excludes():
    result = _select(quality_fatal={"B.NS"})
    assert "B.NS" not in result.symbols
    assert "quality_fatal" in result.excluded.set_index("symbol").loc["B.NS", "reasons"]


def test_never_eligible_excludes():
    close, turnover, volume = make_panels(n_days=2000)
    turnover["C.NS"] = 1e6  # never clears the floor on any date
    result = select_universe(
        close=close, turnover=turnover, volume=volume,
        sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"},
        criteria=criteria(), dates=[close.index[1200], close.index[1900]],
    )
    assert "C.NS" not in result.symbols
    assert "never_eligible" in result.excluded.set_index("symbol").loc["C.NS", "reasons"]


def test_every_exclusion_reason_is_documented():
    """A report describing a reason that never fires, or omitting one that
    does, is worse than no report."""
    fired = set()
    for result in (
        _select(sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": ""}),
        _select(complete_downloads={"A.NS"}),
        _select(quality_fatal={"B.NS"}),
    ):
        for reasons in result.excluded.get("reasons", []):
            fired |= {c for c in str(reasons).split(";") if c}
    assert fired <= set(REASON_CODES), f"undocumented reasons: {fired - set(REASON_CODES)}"


def test_size_cap_is_recorded_as_a_reason():
    close, turnover, volume = make_panels(n_days=2000)
    turnover["A.NS"], turnover["B.NS"], turnover["C.NS"] = 1e9, 5e8, 3e8
    result = select_universe(
        close=close, turnover=turnover, volume=volume,
        sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"},
        criteria=criteria(), dates=[close.index[1900]], max_size=2,
    )
    assert len(result.symbols) == 2
    assert "size_cap" in result.excluded.set_index("symbol").loc["C.NS", "reasons"]


# ===========================================================================
# hashing and versioning
# ===========================================================================


def test_hash_is_order_independent():
    a = universe_hash(["B.NS", "A.NS"], {"A.NS": "IT", "B.NS": "IT"})
    b = universe_hash(["A.NS", "B.NS"], {"B.NS": "IT", "A.NS": "IT"})
    assert a == b


def test_hash_changes_when_a_member_changes():
    base = universe_hash(["A.NS", "B.NS"], {"A.NS": "IT", "B.NS": "IT"})
    added = universe_hash(["A.NS", "B.NS", "C.NS"], {"A.NS": "IT", "B.NS": "IT", "C.NS": "IT"})
    removed = universe_hash(["A.NS"], {"A.NS": "IT"})
    assert base != added != removed
    assert base != removed


def test_hash_changes_when_a_sector_is_reclassified():
    """A reclassification changes which pairs the economic prior allows, so it
    is a different universe even though the ticker list is identical."""
    a = universe_hash(["A.NS", "B.NS"], {"A.NS": "IT", "B.NS": "IT"})
    b = universe_hash(["A.NS", "B.NS"], {"A.NS": "IT", "B.NS": "BANK"})
    assert a != b


def test_hash_is_twelve_hex_characters():
    h = universe_hash(["A.NS"], {"A.NS": "IT"})
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_version_does_not_bump_when_the_universe_is_unchanged(tmp_path: Path):
    """Otherwise the version counts script runs rather than real changes."""
    import yaml

    path = tmp_path / "universe.yaml"
    symbols, sectors = ["A.NS", "B.NS"], {"A.NS": "IT", "B.NS": "IT"}
    path.write_text(
        yaml.safe_dump({"version": 3, "tickers": symbols, "sectors": sectors}), encoding="utf-8"
    )
    assert next_version(symbols, sectors, path) == 3


def test_version_bumps_when_the_universe_changes(tmp_path: Path):
    import yaml

    path = tmp_path / "universe.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": 3, "tickers": ["A.NS"], "sectors": {"A.NS": "IT"}}
        ),
        encoding="utf-8",
    )
    assert next_version(["A.NS", "B.NS"], {"A.NS": "IT", "B.NS": "IT"}, path) == 4


def test_first_version_is_one(tmp_path: Path):
    assert next_version(["A.NS"], {"A.NS": "IT"}, tmp_path / "nothing.yaml") == 1


# ===========================================================================
# report
# ===========================================================================


def test_report_contains_the_numbers_it_claims(tmp_path: Path):
    result = _select()
    text = build_report(
        result,
        universe_hash="abc123abc123",
        version=1,
        config_hash="cfg123456789",
        criteria_source={"min_median_turnover_inr": 2.5e8},
    )
    assert "abc123abc123" in text
    assert str(result.n_selected) in text
    assert "Eligible names over time" in text
    assert "Bias declaration" in text
    assert "optimistic" in text


def test_report_files_are_written(tmp_path: Path):
    result = _select()
    paths = write_report(
        result,
        universe_hash="abc123abc123",
        version=1,
        config_hash="cfg",
        criteria_source={"min_price_inr": 20.0},
        directory=tmp_path,
    )
    for path in paths.values():
        assert path.is_file(), f"{path} not written"
    eligibility = pd.read_csv(paths["eligibility"])
    assert {"symbol", "as_of", "eligible", "reasons"} <= set(eligibility.columns)


def test_eligible_over_time_is_reported_per_date():
    result = _select()
    series = result.eligible_over_time()
    assert len(series) == len(result.decision_dates)
    assert (series >= 0).all()


def test_frozen_universe_pair_counts():
    frozen = FrozenUniverse(
        name="x", version=1, as_of="2026-01-01",
        symbols=("A.NS", "B.NS", "C.NS", "D.NS"),
        sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK", "D.NS": "BANK"},
        hash="h", criteria={}, provenance={},
    )
    assert frozen.n_pairs == 6
    assert frozen.same_sector_pairs() == 2
    assert frozen.sector_counts() == {"BANK": 2, "IT": 2}


# ===========================================================================
# REAL DATA
# ===========================================================================


@pytest.mark.realdata
def test_real_universe_is_internally_consistent(cfg):
    u = cfg.universe
    if not u.frozen:
        pytest.skip("universe not frozen")
    assert u.is_populated
    assert set(u.tickers) <= set(u.sectors)
    assert len(set(u.tickers)) == len(u.tickers)
    assert 100 <= len(u.tickers) <= 260, f"{len(u.tickers)} names is outside the plausible band"


@pytest.mark.realdata
def test_real_universe_hash_matches_its_contents(cfg):
    """The hash in the file must describe the list in the file."""
    import yaml

    from dsa.paths import config_dir

    text = (config_dir() / "universe.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not data.get("frozen"):
        pytest.skip("universe not frozen")
    recomputed = universe_hash(data["tickers"], data["sectors"])
    assert f"universe hash : {recomputed}" in text, (
        "the hash recorded in universe.yaml does not match its own ticker list; the file "
        "was edited by hand or written by a stale builder"
    )


@pytest.mark.realdata
def test_real_eligibility_is_causal(cfg):
    """The lookahead guard, on the actual NSE panel."""
    from dsa.data.clean import load_panel
    from dsa.data.quality import infer_sessions

    try:
        close = load_panel("close")
        turnover = load_panel("turnover")
        volume = load_panel("volume")
    except FileNotFoundError:
        pytest.skip("no clean panel; run scripts/build_dataset.py")

    sessions, _, _ = infer_sessions(close, volume, quorum=cfg.quality.session_quorum)
    crit = EligibilityCriteria.from_config(cfg.selection)
    as_of = sessions[len(sessions) // 2]

    full = eligibility_at(
        as_of, close=close, turnover=turnover, volume=volume, criteria=crit, sessions=sessions
    )
    truncated = eligibility_at(
        as_of,
        close=close.loc[:as_of],
        turnover=turnover.loc[:as_of],
        volume=volume.loc[:as_of],
        criteria=crit,
        sessions=sessions[sessions <= as_of],
    )
    pd.testing.assert_frame_equal(full, truncated)


@pytest.mark.realdata
def test_real_eligible_count_is_never_larger_than_the_frozen_universe(cfg):
    report = Path("reports/universe/eligibility.csv")
    if not report.is_file():
        pytest.skip("no universe report; run scripts/build_universe.py")
    schedule = pd.read_csv(report)
    per_date = schedule.groupby("as_of")["eligible"].sum()
    assert (per_date <= len(cfg.universe.tickers)).all()
    assert (per_date > 0).all(), "some decision date has nobody eligible"
