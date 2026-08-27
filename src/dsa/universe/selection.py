"""Point-in-time universe selection --- spec Sec 11.4.

The bug this module exists to fix
---------------------------------
The first version of the universe filter computed median turnover over the
*entire* 2015-2026 sample and used it to declare a name eligible for that whole
period. Two leaks, both flattering in the same direction:

* **Liquidity lookahead.** A stock that was thin through 2015-2018 and became
  liquid in 2022 passes a Rs 25cr filter on its full-sample median, and is then
  traded in 2015. In 2015 you could not have known it would become liquid.
* **History lookahead.** ``min_history_years >= 5`` measured to 2026 admits a
  2021 listing into the 2015 universe, where it did not exist.

Both preferentially admit names that *turned out* to be tradeable. That is the
universe-construction analogue of the full-sample z-score (spec Sec 2.5), and
it is just as invisible in a backtest, because it also makes the equity curve
better rather than worse.

The two-level design
--------------------
There is an apparent conflict between spec Sec 11.4 ("freeze this list and
version it") and the rule against lookahead: freezing one list for the whole
sample is precisely what leaks. The resolution is to separate two different
things that the word "universe" is doing:

============================  ==========================  ===================
Level                         What it is                  Frozen?
============================  ==========================  ===================
Research universe             names ever considered       yes: versioned+hashed
Point-in-time eligibility     who qualifies at date T,    recomputed per window
                              using only data <= T
============================  ==========================  ===================

The membership *list* is frozen so an experiment is reproducible and its hash
is stable. *Eligibility at a date* is computed causally, and it is eligibility
that the walk-forward loop consults. A name in the frozen list that was not
liquid in 2016 is simply not eligible in 2016.

What is still biased, and cannot be fixed here
----------------------------------------------
The candidate pool comes from **today's** NSE F&O list. Names dropped from F&O,
delisted, or shrunk out of the segment are absent, and their absence flatters
everything. Direction: optimistic. Point-in-time F&O constituent history is not
available free. This is declared rather than fixed, and it is recorded in the
frozen universe file so the caveat travels with the data.

Determinism
-----------
No random number is drawn anywhere in this module. Ranking uses a
point-in-time metric, ties break alphabetically by symbol, and the output is
sorted. Running twice on the same panel gives byte-identical output, and the
universe hash proves it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dsa.logging_utils import get_logger

__all__ = [
    "EligibilityCriteria",
    "SelectionError",
    "decision_dates",
    "liquidity_stats_at",
    "eligibility_at",
    "eligibility_schedule",
    "select_universe",
    "REASON_CODES",
]

_log = get_logger(__name__)

# Exclusion reasons, as stable codes so the report can group them and a test
# can assert on them without matching prose.
REASON_CODES: dict[str, str] = {
    "no_data": "no observations on or before the decision date",
    "short_history": "listed for less than the required number of years",
    "illiquid": "trailing median daily turnover below the floor",
    "low_price": "price below the penny-stock floor",
    "poor_coverage": "too many missing or untradeable sessions in the window",
    "no_sector": "no NSE industry classification",
    "incomplete_download": "no complete raw download recorded in the manifest",
    "quality_fatal": "a FATAL data-quality finding",
    "never_eligible": "never cleared the point-in-time bar on any decision date",
    "size_cap": "eligible, but ranked below the size cap",
    "below_daily_rank": "eligible, but outside the top N by turnover on this date",
}


class SelectionError(RuntimeError):
    """Universe selection could not proceed."""


@dataclass(frozen=True)
class EligibilityCriteria:
    """Point-in-time criteria. Every one uses data on or before the date."""

    min_history_years: float
    min_median_turnover_inr: float
    max_missing_frac: float
    min_price_inr: float
    lookback_sessions: int

    @classmethod
    def from_config(cls, selection_cfg, trading_days_per_year: int = 252) -> EligibilityCriteria:
        return cls(
            min_history_years=selection_cfg.min_history_years,
            min_median_turnover_inr=selection_cfg.min_median_turnover_inr,
            max_missing_frac=selection_cfg.max_missing_frac,
            min_price_inr=selection_cfg.min_price_inr,
            lookback_sessions=selection_cfg.lookback_sessions(trading_days_per_year),
        )


# ---------------------------------------------------------------------------
# decision dates
# ---------------------------------------------------------------------------


def decision_dates(
    sessions: pd.DatetimeIndex,
    *,
    formation_months: int,
    trading_months: int,
    step_months: int,
) -> list[pd.Timestamp]:
    """Formation-window END dates --- the moments eligibility is decided.

    These come from the walk-forward schedule (spec Sec 4.2) rather than being
    listed anywhere, so the decision dates and the backtest windows cannot
    drift apart. Each returned date is the last session on or before a
    formation boundary, so it is a date on which the market was actually open.
    """
    if len(sessions) == 0:
        raise SelectionError("no sessions supplied; cannot derive decision dates")

    sessions = pd.DatetimeIndex(sessions).sort_values()
    start, end = sessions[0], sessions[-1]

    out: list[pd.Timestamp] = []
    boundary = start + pd.DateOffset(months=formation_months)
    while boundary + pd.DateOffset(months=trading_months) <= end:
        # Snap back to the last real session at or before the boundary, so
        # every decision date is a day the market was actually open.
        prior = sessions[sessions <= boundary]
        if len(prior):
            candidate = prior[-1]
            if not out or candidate != out[-1]:
                out.append(candidate)
        boundary = boundary + pd.DateOffset(months=step_months)
    return out


# ---------------------------------------------------------------------------
# point-in-time statistics
# ---------------------------------------------------------------------------


def liquidity_stats_at(
    as_of: pd.Timestamp,
    *,
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    lookback_sessions: int = 252,
) -> pd.DataFrame:
    """Per-symbol statistics computed from data on or before ``as_of``.

    THE CAUSALITY CONTRACT. Every panel is truncated at ``as_of`` before any
    statistic is taken. Appending future rows to the inputs must not change a
    single number returned here, and ``tests/test_universe.py`` asserts exactly
    that by re-running against a truncated panel.

    Returns one row per symbol with:
      first_obs, last_obs, history_years, n_sessions_in_window,
      n_traded_in_window, missing_frac, median_turnover_inr, last_price
    """
    as_of = pd.Timestamp(as_of)

    # --- the truncation. Nothing below may see beyond this line. ------------
    close = close.loc[:as_of]
    turnover = turnover.loc[:as_of]
    volume = None if volume is None else volume.loc[:as_of]
    if sessions is not None:
        sessions = pd.DatetimeIndex(sessions)
        sessions = sessions[sessions <= as_of]
    else:
        sessions = close.index

    if close.empty:
        raise SelectionError(f"no data at or before {as_of.date()}")

    window = sessions[-lookback_sessions:]

    rows: list[dict] = []
    for symbol in close.columns:
        series = close[symbol]
        obs = series.dropna()
        if obs.empty:
            rows.append(
                dict(
                    symbol=symbol,
                    first_obs=None,
                    last_obs=None,
                    history_years=0.0,
                    n_sessions_in_window=0,
                    n_traded_in_window=0,
                    missing_frac=1.0,
                    median_turnover_inr=0.0,
                    last_price=np.nan,
                )
            )
            continue

        first_obs, last_obs = obs.index.min(), obs.index.max()
        # History is measured to the DECISION DATE, not to the end of the
        # sample. That is what makes a 2021 listing ineligible in 2018.
        history_years = (as_of - first_obs).days / 365.25

        # The window is bounded below by the listing date, so a name that
        # listed six months ago is judged on six months, not blamed for the
        # six before it existed.
        win = window[window >= first_obs]
        n_win = len(win)

        if n_win == 0:
            rows.append(
                dict(
                    symbol=symbol,
                    first_obs=str(first_obs.date()),
                    last_obs=str(last_obs.date()),
                    history_years=round(history_years, 3),
                    n_sessions_in_window=0,
                    n_traded_in_window=0,
                    missing_frac=1.0,
                    median_turnover_inr=0.0,
                    last_price=float(obs.iloc[-1]),
                )
            )
            continue

        present = series.reindex(win).notna()
        if volume is not None and symbol in volume.columns:
            # "Traded", not merely "supplied": a zero-volume halt is not a
            # session you could have traded, whatever the price column says.
            traded = present & (volume[symbol].reindex(win).fillna(0) > 0)
        else:
            traded = present
        n_traded = int(traded.sum())

        turn_win = turnover[symbol].reindex(win)
        median_turnover = float(turn_win[traded].median()) if n_traded else 0.0

        rows.append(
            dict(
                symbol=symbol,
                first_obs=str(first_obs.date()),
                last_obs=str(last_obs.date()),
                history_years=round(history_years, 3),
                n_sessions_in_window=n_win,
                n_traded_in_window=n_traded,
                missing_frac=round(1.0 - n_traded / n_win, 4),
                median_turnover_inr=0.0 if np.isnan(median_turnover) else median_turnover,
                last_price=float(obs.iloc[-1]),
            )
        )

    return pd.DataFrame(rows).set_index("symbol").sort_index()


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------


def eligibility_at(
    as_of: pd.Timestamp,
    *,
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    criteria: EligibilityCriteria,
    volume: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    max_eligible: int | None = None,
) -> pd.DataFrame:
    """Who is eligible at ``as_of``, with a reason for everyone who is not.

    Returns the stats frame plus ``eligible`` and ``reasons`` columns. Reasons
    are stable codes from :data:`REASON_CODES`, joined by ``;`` --- prose lives
    in the report, so a test can assert on the code without matching wording.
    """
    stats = liquidity_stats_at(
        as_of,
        close=close,
        turnover=turnover,
        volume=volume,
        sessions=sessions,
        lookback_sessions=criteria.lookback_sessions,
    )

    reasons: list[str] = []
    for _, row in stats.iterrows():
        why: list[str] = []
        if row["n_sessions_in_window"] == 0 or row["first_obs"] is None:
            why.append("no_data")
        else:
            if row["history_years"] < criteria.min_history_years:
                why.append("short_history")
            if row["median_turnover_inr"] < criteria.min_median_turnover_inr:
                why.append("illiquid")
            if not np.isnan(row["last_price"]) and row["last_price"] < criteria.min_price_inr:
                why.append("low_price")
            if row["missing_frac"] > criteria.max_missing_frac:
                why.append("poor_coverage")
        reasons.append(";".join(why))

    stats = stats.copy()
    stats["reasons"] = reasons
    stats["eligible"] = stats["reasons"] == ""

    # --- optional relative rule -------------------------------------------
    # A fixed rupee floor is not scale-invariant across eleven years: measured
    # on this panel it admits 105 names in 2018 and 199 in 2026, because market
    # turnover roughly tripled. Keeping the top N by trailing turnover instead
    # holds the universe size constant, so an era comparison is like-for-like
    # rather than confounded by the universe doubling underneath it.
    #
    # The ranking metric is itself point-in-time, and the tie-break is
    # alphabetical, so this stays deterministic and free of lookahead.
    if max_eligible is not None:
        eligible_now = stats.index[stats["eligible"]]
        if len(eligible_now) > max_eligible:
            ranked = (
                stats.loc[eligible_now]
                .sort_index()
                .sort_values("median_turnover_inr", ascending=False, kind="stable")
            )
            dropped = ranked.index[max_eligible:]
            stats.loc[dropped, "eligible"] = False
            stats.loc[dropped, "reasons"] = "below_daily_rank"

    stats["as_of"] = str(pd.Timestamp(as_of).date())
    return stats


def eligibility_schedule(
    dates: Sequence[pd.Timestamp],
    *,
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    criteria: EligibilityCriteria,
    volume: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    max_eligible: int | None = None,
) -> pd.DataFrame:
    """Eligibility at every decision date, stacked.

    This is the object the walk-forward loop consults: for formation window
    ending T, trade only the names with ``eligible`` True at T.
    """
    frames = []
    for as_of in dates:
        frame = eligibility_at(
            as_of,
            close=close,
            turnover=turnover,
            criteria=criteria,
            volume=volume,
            sessions=sessions,
            max_eligible=max_eligible,
        )
        frames.append(frame.reset_index())
    if not frames:
        raise SelectionError("no decision dates supplied")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# the frozen research universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionResult:
    """The frozen list plus everything needed to explain and audit it."""

    symbols: tuple[str, ...]
    sectors: Mapping[str, str]
    excluded: pd.DataFrame  # symbol, reasons, plus stats at the last decision date
    schedule: pd.DataFrame  # eligibility at every decision date
    decision_dates: tuple[pd.Timestamp, ...]
    criteria: EligibilityCriteria
    n_candidates: int

    @property
    def n_selected(self) -> int:
        return len(self.symbols)

    @property
    def n_pairs(self) -> int:
        n = self.n_selected
        return n * (n - 1) // 2

    def same_sector_pairs(self) -> int:
        counts = pd.Series([self.sectors[s] for s in self.symbols]).value_counts()
        return int((counts * (counts - 1) // 2).sum())

    def eligible_over_time(self) -> pd.Series:
        """Count of eligible names at each decision date --- the honest size."""
        frame = self.schedule[self.schedule["symbol"].isin(self.symbols)]
        return frame.groupby("as_of")["eligible"].sum().astype(int)


def select_universe(
    *,
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    sectors: Mapping[str, str],
    criteria: EligibilityCriteria,
    dates: Sequence[pd.Timestamp],
    volume: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    complete_downloads: set[str] | None = None,
    quality_fatal: set[str] | None = None,
    require_sector: bool = True,
    require_ever_eligible: bool = True,
    max_size: int | None = None,
    max_eligible_per_date: int | None = None,
    rank_by: str = "median_turnover_inr",
) -> SelectionResult:
    """Build the frozen research universe. Deterministic; draws no randomness.

    Structural gates (sector, complete download, no FATAL quality finding) are
    properties of the data, so they are not point-in-time. The liquidity and
    history gates are, and a name enters the frozen list if it clears them on
    at least one decision date --- which is a superset condition, so it cannot
    leak into the backtest: the per-window eligibility still excludes it on
    every date where it did not qualify.
    """
    schedule = eligibility_schedule(
        dates,
        close=close,
        turnover=turnover,
        criteria=criteria,
        volume=volume,
        sessions=sessions,
        max_eligible=max_eligible_per_date,
    )

    candidates = list(close.columns)
    ever = set(schedule.loc[schedule["eligible"], "symbol"])
    last_date = str(pd.Timestamp(dates[-1]).date())
    latest = schedule[schedule["as_of"] == last_date].set_index("symbol")

    rows: list[dict] = []
    keep: list[str] = []

    for symbol in sorted(candidates):
        why: list[str] = []
        if require_sector and not sectors.get(symbol):
            why.append("no_sector")
        if complete_downloads is not None and symbol not in complete_downloads:
            why.append("incomplete_download")
        if quality_fatal and symbol in quality_fatal:
            why.append("quality_fatal")
        if require_ever_eligible and symbol not in ever:
            why.append("never_eligible")

        if why:
            row = dict(symbol=symbol, sector=sectors.get(symbol, ""), reasons=";".join(why))
            if symbol in latest.index:
                for col in (
                    "history_years",
                    "median_turnover_inr",
                    "missing_frac",
                    "last_price",
                ):
                    row[col] = latest.loc[symbol, col]
            rows.append(row)
        else:
            keep.append(symbol)

    # --- deterministic size cap --------------------------------------------
    if max_size is not None and len(keep) > max_size:
        if rank_by not in latest.columns:
            raise SelectionError(f"cannot rank by {rank_by!r}; not in the eligibility stats")
        # Rank by a point-in-time metric, break ties alphabetically. Sorting by
        # symbol first and using a stable sort makes the tie-break explicit
        # rather than an accident of the sort implementation.
        ranked = (
            latest.loc[[s for s in keep if s in latest.index]]
            .sort_index()
            .sort_values(rank_by, ascending=False, kind="stable")
        )
        chosen = list(ranked.index[:max_size])
        for symbol in keep:
            if symbol not in chosen:
                row = dict(symbol=symbol, sector=sectors.get(symbol, ""), reasons="size_cap")
                if symbol in latest.index:
                    row["median_turnover_inr"] = latest.loc[symbol, "median_turnover_inr"]
                rows.append(row)
        keep = sorted(chosen)

    excluded = pd.DataFrame(rows)
    if not excluded.empty:
        excluded = excluded.sort_values("symbol").reset_index(drop=True)

    _log.info(
        "universe selection: %d candidates -> %d selected (%d excluded)",
        len(candidates),
        len(keep),
        len(candidates) - len(keep),
    )

    return SelectionResult(
        symbols=tuple(sorted(keep)),
        sectors={s: sectors.get(s, "") for s in sorted(keep)},
        excluded=excluded,
        schedule=schedule,
        decision_dates=tuple(pd.Timestamp(d) for d in dates),
        criteria=criteria,
        n_candidates=len(candidates),
    )


def as_of_today() -> date:
    return date.today()
