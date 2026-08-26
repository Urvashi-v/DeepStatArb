"""The data-quality engine --- spec Sec 11.2.

The governing rule
------------------
**This module flags. It never deletes.** Every check emits ``Issue`` records
carrying a location and a reason; nothing here filters, repairs, interpolates
or drops a row. That is not squeamishness, it is method: silently discarding
"suspicious" observations is itself a form of lookahead. You would be choosing
which days to keep using knowledge of what the price did on them, and the days
that look wrong are disproportionately the days something real happened ---
results, a block deal, a crisis. Delete them and the backtest gets smoother and
less true.

So the output is a report a person reads, plus boolean masks a downstream stage
can consult if it decides a particular check should exclude a particular bar
from a particular computation. The decision stays explicit and stays out of
the loader.

The three checks that carry the most weight
-------------------------------------------
**Session inference.** There is no authoritative NSE holiday calendar
available offline. Business-day ranges are wrong --- roughly fifteen holidays a
year would show as fake gaps across all 203 names simultaneously. Taking the
union of observed dates is worse: one ticker with a spurious print invents a
session for everyone, and every other ticker then shows a phantom missing day.
So a date is a session when at least ``session_quorum`` of the tickers *listed
at that time* **traded** on it.

The word "traded" is doing real work there, and getting it wrong was a bug
this project's own data caught. Participation was first measured as "has a
non-NaN close", which passes an exchange holiday that the provider padded with
a carried-forward close and zero volume --- and the panel contains five such
dates. Measuring participation by volume instead separates them out. See
``infer_sessions`` for why an undetected padded day is worse here than
elsewhere.

**Extreme moves, and what this feed can actually tell you.** Spec Sec 11.2
asks for a >20% single-day screen to catch corporate-action artefacts. In this
feed it cannot do that directly: Day 2 established that Yahoo's ``Close`` for
NSE names arrives already split-adjusted (the 2:1 RELIANCE split of 2017-09-07
shows as -0.56%, not -50%). A screen that reports nothing and is assumed to be
passing is worse than no screen. So every flagged move is cross-referenced
against the ``Dividends`` and ``Stock Splits`` columns within a few sessions,
splitting the flags into two populations that mean different things: moves with
no nearby action (genuine market events, or bad prints) and moves adjacent to a
reported action (candidates for an adjustment the provider got wrong).

**Stale prices.** An unchanged close for several sessions is the quiet way a
suspended stock manufactures a false result: a flat stretch reads to an ADF
test as an exceptionally well-behaved spread, which is exactly the direction
that invents cointegration where there is none. Not in the usual checklist,
and worth more than most things that are.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsa.logging_utils import get_logger
from dsa.paths import reports_dir

__all__ = [
    "Severity",
    "Issue",
    "QualityReport",
    "infer_sessions",
    "listed_span",
    "check_ticker",
    "check_panel",
    "run_quality_checks",
    "CHECKS",
]

_log = get_logger(__name__)


class Severity(str, Enum):
    """How much a finding should stop you.

    FATAL   structurally impossible. Something downstream will be wrong if this
            is ignored --- a duplicated bar double-counts a return, a negative
            price breaks every log.
    WARN    plausible but needs a human. Most real findings land here.
    INFO    expected and benign. Recorded so the count is visible rather than
            silently absorbed.
    """

    FATAL = "FATAL"
    WARN = "WARN"
    INFO = "INFO"


# Registry of every check, with the one-line rationale that appears in the
# report. Keeping it here means the report can never describe a check that
# does not run, or omit one that does.
CHECKS: dict[str, str] = {
    "duplicate_timestamps": "the same trading date appears more than once",
    "chronological_order": "the index is not sorted ascending",
    "missing_values": "NaN in a price or volume field within the listed life",
    "invalid_prices": "non-positive price, or OHLC that cannot be a real bar",
    "zero_volume": "bars with no shares traded",
    "stale_price": "an unchanged close across several sessions",
    "extreme_move": "single-day move beyond the inspection threshold",
    "missing_sessions": "no bar on a date the market was open",
    "gaps": "consecutive missing sessions within the listed life",
    "coverage": "observed fraction of the ticker's own listed life",
    "phantom_session": "a date carried by too few tickers to be a real session",
    "padded_session": "a non-trading date the provider filled with a carried-forward close",
    "ticker_consistency": "universe, panel and manifest disagree on membership",
}


@dataclass(frozen=True)
class Issue:
    """One finding. Never an instruction to delete anything."""

    check: str
    severity: Severity
    symbol: str | None = None
    date: pd.Timestamp | None = None
    detail: str = ""
    value: float | None = None
    n_affected: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "date": None if self.date is None else str(pd.Timestamp(self.date).date()),
            "detail": self.detail,
            "value": self.value,
            "n_affected": self.n_affected,
        }


@dataclass
class QualityReport:
    """Everything the engine found, plus enough provenance to reproduce it."""

    issues: list[Issue] = field(default_factory=list)
    per_ticker: pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    padded_sessions: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    n_symbols: int = 0
    generated_at: str = ""
    config_hash: str | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)

    # -- queries ------------------------------------------------------------

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def of_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def fatal(self) -> list[Issue]:
        return self.of_severity(Severity.FATAL)

    @property
    def warnings(self) -> list[Issue]:
        return self.of_severity(Severity.WARN)

    @property
    def is_usable(self) -> bool:
        """False when something structurally impossible was found."""
        return not self.fatal

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["check", "severity", "symbol", "date", "detail", "value", "n_affected"]
            )
        return pd.DataFrame([i.to_dict() for i in self.issues])

    def counts(self) -> pd.DataFrame:
        """Issue counts by check and severity --- the top of the report."""
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["check", "FATAL", "WARN", "INFO", "total"])
        pivot = (
            frame.pivot_table(index="check", columns="severity", values="n_affected", aggfunc="sum")
            .reindex(columns=["FATAL", "WARN", "INFO"])
            .fillna(0)
            .astype(int)
        )
        pivot["total"] = pivot.sum(axis=1)
        return pivot.sort_values("total", ascending=False).reset_index()

    # -- output -------------------------------------------------------------

    def summary_text(self, max_per_check: int = 25) -> str:
        lines: list[str] = []
        add = lines.append

        add("# Data quality report")
        add("")
        add(f"generated   : {self.generated_at}")
        add(f"config hash : {self.config_hash}")
        add(f"symbols     : {self.n_symbols}")
        if len(self.sessions):
            add(
                f"sessions    : {len(self.sessions)} "
                f"({self.sessions.min().date()} .. {self.sessions.max().date()})"
            )
        add("")
        add("**Nothing below has been deleted, repaired or interpolated.** Every")
        add("finding is a flag for inspection. Dropping suspicious observations")
        add("would itself be a form of lookahead: the days that look wrong are")
        add("disproportionately the days something real happened.")
        add("")

        add("## Verdict")
        add("")
        if self.fatal:
            add(f"**{len(self.fatal)} FATAL finding(s).** Something structurally impossible is")
            add("present. Resolve before building anything on this panel.")
        else:
            add("**No FATAL findings.** No duplicated bars, no non-monotonic indexes,")
            add("no non-positive prices, no impossible OHLC. The panel is structurally sound.")
        add("")
        if len(self.padded_sessions):
            add(f"**{len(self.padded_sessions)} padded non-trading day(s)** were identified "
                "and excluded from the session calendar. They are listed under "
                "`padded_session` below and must be dropped before returns are computed.")
            add("")
        n_warn = sum(i.n_affected for i in self.warnings)
        add(f"{n_warn} observation(s) flagged for inspection across "
            f"{len(self.warnings)} warning group(s).")
        add("")

        add("## Counts by check")
        add("")
        counts = self.counts()
        if counts.empty:
            add("_no issues recorded_")
        else:
            add("| check | FATAL | WARN | INFO | total | what it means |")
            add("|---|---:|---:|---:|---:|---|")
            for _, row in counts.iterrows():
                add(
                    f"| `{row['check']}` | {row['FATAL']} | {row['WARN']} | {row['INFO']} "
                    f"| {row['total']} | {CHECKS.get(row['check'], '')} |"
                )
        add("")

        for severity in (Severity.FATAL, Severity.WARN, Severity.INFO):
            group = self.of_severity(severity)
            if not group:
                continue
            add(f"## {severity.value}")
            add("")
            by_check: dict[str, list[Issue]] = {}
            for issue in group:
                by_check.setdefault(issue.check, []).append(issue)
            for check, items in sorted(by_check.items()):
                add(f"### `{check}` --- {CHECKS.get(check, '')}")
                add("")
                for issue in items[:max_per_check]:
                    where = issue.symbol or "-"
                    when = "" if issue.date is None else f" {pd.Timestamp(issue.date).date()}"
                    add(f"- **{where}**{when}: {issue.detail}")
                if len(items) > max_per_check:
                    add(f"- _... and {len(items) - max_per_check} more (see issues.csv)_")
                add("")

        add("## Per-ticker coverage")
        add("")
        if not self.per_ticker.empty:
            add("`coverage` asks whether a bar was supplied; `tradeable` asks whether")
            add("anything actually changed hands. A name can show 100% coverage and")
            add("still have been untradeable for weeks, so the two are shown together.")
            add("")
            column = (
                "tradeable_coverage" if "tradeable_coverage" in self.per_ticker else "coverage"
            )
            worst = self.per_ticker.nsmallest(10, column)
            add("Ten lowest-tradeability names (over their own listed life):")
            add("")
            add("| symbol | first | last | sessions | observed | coverage | tradeable | zero-vol | max gap |")
            add("|---|---|---|---:|---:|---:|---:|---:|---:|")
            for _, r in worst.iterrows():
                add(
                    f"| {r['symbol']} | {r['first']} | {r['last']} | {r['n_sessions']} "
                    f"| {r['n_observed']} | {r['coverage']:.1%} "
                    f"| {r.get('tradeable_coverage', float('nan')):.1%} "
                    f"| {r.get('n_zero_volume', 0)} | {r['max_gap']} |"
                )
        add("")
        return "\n".join(lines)

    def write(self, directory: Path | None = None) -> dict[str, Path]:
        """Write the reproducible report. Returns the paths written."""
        out = directory or (reports_dir() / "data_quality")
        out.mkdir(parents=True, exist_ok=True)

        paths = {
            "summary": out / "summary.md",
            "issues": out / "issues.csv",
            "per_ticker": out / "per_ticker.csv",
            "meta": out / "meta.json",
        }
        paths["summary"].write_text(
            self.summary_text(self.thresholds.get("max_issues_listed_per_check", 25)),
            encoding="utf-8",
        )
        self.to_frame().to_csv(paths["issues"], index=False)
        self.per_ticker.to_csv(paths["per_ticker"], index=False)
        paths["meta"].write_text(
            json.dumps(
                {
                    "generated_at": self.generated_at,
                    "config_hash": self.config_hash,
                    "n_symbols": self.n_symbols,
                    "n_sessions": len(self.sessions),
                    "padded_sessions": [str(d.date()) for d in self.padded_sessions],
                    "thresholds": self.thresholds,
                    "counts": self.counts().to_dict(orient="records"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        _log.info("quality report written to %s", out)
        return paths


# ---------------------------------------------------------------------------
# session calendar
# ---------------------------------------------------------------------------


def listed_span(series: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """First and last observed date --- the ticker's own listed life."""
    observed = series.dropna()
    if observed.empty:
        return None, None
    return observed.index.min(), observed.index.max()


def infer_sessions(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    quorum: float = 0.5,
    padding_frac: float = 0.90,
) -> tuple[pd.DatetimeIndex, pd.Series, pd.DatetimeIndex]:
    """Infer the trading calendar from the panel.

    Returns ``(sessions, participation, padded_dates)``.

    A date is a session when at least ``quorum`` of the tickers listed at that
    time actually **traded** on it. Taking every date that appears anywhere
    would let a single bad print create a session for the whole universe, after
    which every other ticker shows a spurious missing day on it.

    Why participation is measured by VOLUME, not by the presence of a price
    ----------------------------------------------------------------------
    This was a real bug, caught by this project's own data. Yahoo pads NSE
    market holidays with a row carrying the previous close and zero volume. In
    the downloaded panel five such dates exist, on which 99-100% of listed
    tickers show zero volume and *exactly* zero price change.

    Those rows have a non-NaN close, so a ``notna()``-based quorum passes them
    and they enter the calendar as real sessions. They are not. Each one
    inserts a zero return into every series, which biases volatility down,
    inflates lag-1 autocorrelation, and --- the reason it matters most here ---
    makes every spread look flatter and therefore more mean-reverting than it
    is. That is precisely the direction that manufactures cointegration where
    there is none.

    So when a volume panel is supplied, a ticker "participates" only if it
    traded. Dates where at least ``padding_frac`` of listed tickers show no
    volume are returned separately as ``padded_dates`` and excluded from the
    session calendar.
    """
    observed = close.notna()

    # "Listed on date d" = d lies between the ticker's first and last bar.
    # Without this, a 2024 IPO would drag the participation ratio down for
    # every date from 2015 onward and the quorum would never be met.
    first = observed.idxmax()  # first True per column
    has_any = observed.any()
    last = observed[::-1].idxmax()

    index = close.index
    listed = pd.DataFrame(False, index=index, columns=close.columns)
    for symbol in close.columns:
        if not has_any[symbol]:
            continue
        listed.loc[first[symbol] : last[symbol], symbol] = True

    n_listed = listed.sum(axis=1)

    # Participation means "actually traded", wherever volume is available to
    # say so. See the docstring: a padded holiday has a price but no trade.
    traded = observed & (volume.reindex_like(close).fillna(0) > 0) if volume is not None else observed

    n_traded = (traded & listed).sum(axis=1)
    participation = (n_traded / n_listed.replace(0, np.nan)).fillna(0.0)

    padded = pd.DatetimeIndex(
        index[(n_listed > 0) & (participation <= 1.0 - padding_frac)], name="date"
    )
    sessions = pd.DatetimeIndex(
        index[(participation >= quorum) & (~index.isin(padded))], name="date"
    )
    return sessions, participation, padded


# ---------------------------------------------------------------------------
# per-ticker checks (run against RAW frames --- the provider's own output)
# ---------------------------------------------------------------------------


def _runs_of(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Consecutive True runs as (start, end, length)."""
    if not mask.any():
        return []
    values = mask.to_numpy()
    idx = mask.index
    runs = []
    start = None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((idx[start], idx[i - 1], i - start))
            start = None
    if start is not None:
        runs.append((idx[start], idx[len(values) - 1], len(values) - start))
    return runs


def check_ticker(
    symbol: str,
    raw: pd.DataFrame,
    *,
    extreme_move_pct: float = 0.20,
    severe_move_pct: float = 0.50,
    corporate_action_window_days: int = 3,
    max_zero_volume_run: int = 3,
    zero_volume_frac_warn: float = 0.02,
    max_stale_price_run: int = 5,
    market_stress: pd.Series | None = None,
    market_stress_frac: float = 0.10,
) -> list[Issue]:
    """Structural and value checks on one ticker's raw bars.

    Runs against the RAW frame rather than the clean panel, because the raw
    frame is the provider's own output --- checking the cleaned version would
    only tell you whether cleaning succeeded, not whether the source was sound.
    """
    issues: list[Issue] = []

    def flag(check: str, severity: Severity, detail: str, **kw: Any) -> None:
        issues.append(Issue(check=check, severity=severity, symbol=symbol, detail=detail, **kw))

    if raw.empty:
        flag("missing_values", Severity.FATAL, "the frame is empty")
        return issues

    df = raw.copy()
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()

    # --- duplicate timestamps -----------------------------------------------
    dupes = df.index[df.index.duplicated(keep=False)]
    if len(dupes):
        unique_dupes = sorted(set(dupes))
        flag(
            "duplicate_timestamps",
            Severity.FATAL,
            f"{len(unique_dupes)} date(s) appear more than once, e.g. "
            f"{[str(d.date()) for d in unique_dupes[:3]]}. A duplicated bar double-counts "
            "its return and inflates any variance estimated from the series.",
            n_affected=len(dupes) - len(unique_dupes),
            date=unique_dupes[0],
        )

    # --- chronological ordering ---------------------------------------------
    if not df.index.is_monotonic_increasing:
        flag(
            "chronological_order",
            Severity.FATAL,
            "the index is not sorted ascending. Every rolling window, every "
            "shift(1) and every return would be computed across the wrong pairs of bars.",
        )
        df = df.sort_index()

    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in df.columns]

    # --- missing values ------------------------------------------------------
    for col in [*price_cols, "Volume"]:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            flag(
                "missing_values",
                Severity.WARN,
                f"{n_null} NaN in `{col}` inside the downloaded range",
                n_affected=n_null,
            )

    # --- invalid prices ------------------------------------------------------
    for col in price_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((values <= 0).sum())
        if n_bad:
            flag(
                "invalid_prices",
                Severity.FATAL,
                f"{n_bad} non-positive value(s) in `{col}`. A price of zero or less is not a "
                "price; every log return built on it is undefined.",
                n_affected=n_bad,
            )

    if {"High", "Low"} <= set(df.columns):
        hl = df["High"] < df["Low"]
        if hl.any():
            first = df.index[hl][0]
            flag(
                "invalid_prices",
                Severity.FATAL,
                f"{int(hl.sum())} bar(s) with High < Low, first on {first.date()}. "
                "This cannot happen in a real session.",
                n_affected=int(hl.sum()),
                date=first,
            )
        if "Close" in df.columns:
            outside = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
            if outside.any():
                first = df.index[outside][0]
                flag(
                    "invalid_prices",
                    Severity.WARN,
                    f"{int(outside.sum())} bar(s) where Close sits outside [Low, High], first "
                    f"on {first.date()}. Usually a bad print in one of the three fields.",
                    n_affected=int(outside.sum()),
                    date=first,
                )
        if "Open" in df.columns:
            outside = (df["Open"] > df["High"]) | (df["Open"] < df["Low"])
            if outside.any():
                first = df.index[outside][0]
                flag(
                    "invalid_prices",
                    Severity.WARN,
                    f"{int(outside.sum())} bar(s) where Open sits outside [Low, High], first on "
                    f"{first.date()}. The execution model fills at the open, so these bars "
                    "matter more than their count suggests.",
                    n_affected=int(outside.sum()),
                    date=first,
                )

    # --- zero volume ---------------------------------------------------------
    if "Volume" in df.columns:
        volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        zero = volume <= 0
        n_zero = int(zero.sum())
        if n_zero:
            frac = n_zero / len(df)
            severity = Severity.WARN if frac > zero_volume_frac_warn else Severity.INFO
            flag(
                "zero_volume",
                severity,
                f"{n_zero} bar(s) with zero volume ({frac:.2%} of the series). A single one is "
                "usually a halt; a large fraction means the name was not reliably tradeable.",
                n_affected=n_zero,
            )
            for start, end, length in _runs_of(zero):
                if length > max_zero_volume_run:
                    flag(
                        "zero_volume",
                        Severity.WARN,
                        f"{length} consecutive zero-volume sessions {start.date()}..{end.date()}. "
                        "Any spread built on this window is fiction regardless of what the "
                        "price column says.",
                        n_affected=length,
                        date=start,
                    )
        if (volume < 0).any():
            flag(
                "invalid_prices",
                Severity.FATAL,
                f"{int((volume < 0).sum())} bar(s) with negative volume",
                n_affected=int((volume < 0).sum()),
            )

    # --- stale prices --------------------------------------------------------
    if "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
        unchanged = close.diff() == 0
        for start, end, length in _runs_of(unchanged):
            if length >= max_stale_price_run:
                flag(
                    "stale_price",
                    Severity.WARN,
                    f"close unchanged at {close.loc[start]:.2f} for {length + 1} consecutive "
                    f"sessions {start.date()}..{end.date()}. A flat stretch reads to an ADF "
                    "test as an exceptionally well-behaved spread, which is how a suspended "
                    "stock manufactures a false cointegration result.",
                    n_affected=length + 1,
                    date=start,
                    value=float(close.loc[start]),
                )

    # --- extreme single-day moves -------------------------------------------
    if "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
        # fill_method=None: a NaN bar must NOT be padded before differencing,
        # or a gap silently becomes a zero return.
        returns = close.pct_change(fill_method=None)

        dividends = pd.to_numeric(df.get("Dividends", 0.0), errors="coerce").fillna(0.0)
        splits = pd.to_numeric(df.get("Stock Splits", 0.0), errors="coerce").fillna(0.0)
        action = (dividends != 0) | (splits != 0)
        # A corporate action within +/- w sessions is a candidate explanation.
        w = corporate_action_window_days
        near_action = (
            action.rolling(2 * w + 1, center=True, min_periods=1).max().astype(bool)
            if w > 0
            else action
        )
        action_dates = df.index[action]

        def _nearest_action(date: pd.Timestamp) -> str:
            """Describe the action that made this move 'explained'.

            Reporting the dividend on the flagged bar itself is useless --- it is
            usually zero, because the action sits a session or two away. Naming
            the actual nearby event is what lets someone go and check it.
            """
            if len(action_dates) == 0:
                return "none reported"
            offsets = np.abs(
                df.index.get_indexer(action_dates) - int(df.index.get_loc(date))
            )
            nearest = action_dates[int(np.argmin(offsets))]
            return (
                f"{nearest.date()} (dividend {float(dividends.loc[nearest]):.2f}, "
                f"split {float(splits.loc[nearest]):.2f}), {int(offsets.min())} session(s) away"
            )

        extreme = returns.abs() > extreme_move_pct
        for date in df.index[extreme.fillna(False)]:
            move = float(returns.loc[date])
            severe = abs(move) > severe_move_pct
            explained = bool(near_action.loc[date])

            # Market context: was this a day the whole market moved, or did
            # this one name move alone? A 30% drop on a day a third of the
            # universe fell sharply is a crash. The same drop on a quiet day is
            # a name-specific event or a bad print. That distinction turns a
            # long list of moves to eyeball into a short list worth eyeballing.
            stress = float("nan")
            if market_stress is not None and date in market_stress.index:
                stress = float(market_stress.loc[date])
            known = stress == stress  # False for NaN
            wide = known and stress >= market_stress_frac

            if wide:
                context = (
                    f" {stress:.0%} of the universe also moved more than 5% that day, so this "
                    "is a market-wide move rather than a name-specific one."
                )
            elif known:
                context = (
                    f" Isolated --- only {stress:.0%} of the universe moved more than 5% that "
                    "day, which makes a name-specific event or a bad print more likely."
                )
            else:
                context = ""

            if explained:
                detail = (
                    f"{move:+.1%} single-day move; nearest corporate action "
                    f"{_nearest_action(date)}. Yahoo pre-adjusts NSE splits, so a move this "
                    "large next to a reported action suggests the adjustment was applied on "
                    f"the wrong date or not at all.{context}"
                )
                severity = Severity.WARN
            else:
                detail = (
                    f"{move:+.1%} single-day move with no corporate action reported nearby."
                    f"{context}"
                )
                severity = Severity.WARN if (severe or not wide) else Severity.INFO

            flag("extreme_move", severity, detail, date=date, value=move)

    return issues


# ---------------------------------------------------------------------------
# panel-level checks
# ---------------------------------------------------------------------------


def check_panel(
    close: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    participation: pd.Series,
    padded: pd.DatetimeIndex | None = None,
    volume: pd.DataFrame | None = None,
    universe: Sequence[str] | None = None,
    manifest_symbols: Iterable[str] | None = None,
    max_gap_sessions: int = 5,
    min_coverage_over_life: float = 0.95,
    phantom_date_max_tickers: int = 5,
) -> tuple[list[Issue], pd.DataFrame]:
    """Alignment, coverage and membership checks across the whole panel."""
    issues: list[Issue] = []
    rows: list[dict[str, Any]] = []

    observed = close.notna()

    # --- padded non-trading days ---------------------------------------------
    # Rows the provider invented: a carried-forward close with no volume, on a
    # date the exchange was shut. They are excluded from the session calendar
    # and reported here so downstream stages can drop them explicitly.
    for date in padded if padded is not None else []:
        row_returns = close.loc[date] / close.shift(1).loc[date] - 1.0
        n_moved = int((row_returns.abs() > 1e-12).sum())
        n_listed_here = int(observed.loc[date].sum())
        n_traded_here = int(round(participation.loc[date] * n_listed_here))

        if n_traded_here == 0:
            cause = (
                "no listed ticker traded and none moved: an exchange holiday the provider "
                "filled with a carried-forward close and zero volume."
            )
        else:
            # Worse than a clean holiday. A handful of real rows makes the date
            # look like a thin but genuine session, so it survives casual
            # inspection while the padded majority quietly contributes zeros.
            cause = (
                f"only {n_traded_here} of {n_listed_here} listed tickers traded and only "
                f"{n_moved} moved at all. This is a PARTIAL provider failure rather than a "
                "clean holiday: most names got a padded row while a few got real data, so "
                "the date looks like a thin session instead of an absent one."
            )

        issues.append(
            Issue(
                check="padded_session",
                severity=Severity.WARN,
                date=date,
                detail=(
                    f"{cause} NOT DELETED --- the date is excluded from the inferred session "
                    "calendar, and downstream stages should drop it before computing returns. "
                    "Left in place it inserts a zero return into nearly every series, which "
                    "biases volatility down and makes every spread look flatter, and therefore "
                    "more mean-reverting, than it is."
                ),
                value=float(participation.loc[date]),
                n_affected=n_listed_here,
            )
        )

    # --- phantom sessions ----------------------------------------------------
    n_present = observed.sum(axis=1)
    phantom = close.index[(n_present > 0) & (n_present <= phantom_date_max_tickers)]
    for date in phantom:
        issues.append(
            Issue(
                check="phantom_session",
                severity=Severity.WARN,
                date=date,
                detail=(
                    f"only {int(n_present.loc[date])} ticker(s) have a bar on this date, and "
                    f"{participation.loc[date]:.1%} of listed names participated. Almost "
                    "certainly a stray print rather than a session --- if it were treated as a "
                    "trading day, every other ticker would show a spurious missing bar on it."
                ),
                value=float(participation.loc[date]),
            )
        )

    # --- per ticker: missing sessions, gaps, coverage -------------------------
    for symbol in close.columns:
        series = close[symbol]
        first, last = listed_span(series)
        if first is None:
            issues.append(
                Issue(
                    check="coverage",
                    severity=Severity.FATAL,
                    symbol=symbol,
                    detail="no observations at all in the panel",
                )
            )
            rows.append(
                dict(
                    symbol=symbol,
                    first=None,
                    last=None,
                    n_sessions=0,
                    n_observed=0,
                    coverage=0.0,
                    n_traded=0,
                    n_zero_volume=0,
                    tradeable_coverage=0.0,
                    n_missing=0,
                    max_gap=0,
                    n_gaps=0,
                )
            )
            continue

        # Coverage is measured over the ticker's OWN listed life. A 2024 IPO is
        # not "60% missing" over a 2015-2026 panel; it is young.
        own_sessions = sessions[(sessions >= first) & (sessions <= last)]
        present = series.reindex(own_sessions).notna()
        n_sessions = len(own_sessions)
        n_observed = int(present.sum())
        coverage = n_observed / n_sessions if n_sessions else 0.0
        missing_mask = ~present
        n_missing = int(missing_mask.sum())

        gap_runs = _runs_of(missing_mask)
        max_gap = max((length for _, _, length in gap_runs), default=0)

        # `coverage` asks "did the provider give us a bar", so a zero-volume
        # halt counts as covered. That is the right definition for a coverage
        # metric but a misleading one on its own: a name can show 100% coverage
        # and still have been untradeable for weeks. `tradeable_coverage`
        # carries the other half, so the two are never read apart.
        if volume is not None and symbol in volume.columns:
            traded = (volume[symbol].reindex(own_sessions).fillna(0) > 0) & present
            n_traded = int(traded.sum())
            n_zero_volume = n_observed - n_traded
        else:
            n_traded, n_zero_volume = n_observed, 0

        rows.append(
            dict(
                symbol=symbol,
                first=str(first.date()),
                last=str(last.date()),
                n_sessions=n_sessions,
                n_observed=n_observed,
                coverage=coverage,
                n_traded=n_traded,
                n_zero_volume=n_zero_volume,
                tradeable_coverage=n_traded / n_sessions if n_sessions else 0.0,
                n_missing=n_missing,
                max_gap=max_gap,
                n_gaps=len(gap_runs),
            )
        )

        if n_missing:
            issues.append(
                Issue(
                    check="missing_sessions",
                    severity=Severity.INFO if coverage >= min_coverage_over_life else Severity.WARN,
                    symbol=symbol,
                    detail=(
                        f"{n_missing} session(s) with no bar inside its listed life "
                        f"({first.date()}..{last.date()}), coverage {coverage:.2%}"
                    ),
                    n_affected=n_missing,
                )
            )

        for start, end, length in gap_runs:
            if length > max_gap_sessions:
                issues.append(
                    Issue(
                        check="gaps",
                        severity=Severity.WARN,
                        symbol=symbol,
                        date=start,
                        detail=(
                            f"{length} consecutive sessions with no bar, {start.date()}.."
                            f"{end.date()}. The panel leaves these as NaN --- forward-filling "
                            "would invent a flat price and fabricate mean reversion."
                        ),
                        n_affected=length,
                    )
                )

        if coverage < min_coverage_over_life:
            issues.append(
                Issue(
                    check="coverage",
                    severity=Severity.WARN,
                    symbol=symbol,
                    detail=(
                        f"coverage {coverage:.2%} over its listed life is below the "
                        f"{min_coverage_over_life:.0%} threshold"
                    ),
                    value=coverage,
                )
            )

    per_ticker = pd.DataFrame(rows)

    # --- ticker consistency ---------------------------------------------------
    panel_symbols = set(close.columns)
    if universe is not None:
        missing_from_panel = sorted(set(universe) - panel_symbols)
        if missing_from_panel:
            issues.append(
                Issue(
                    check="ticker_consistency",
                    severity=Severity.FATAL,
                    detail=(
                        f"{len(missing_from_panel)} ticker(s) in the frozen universe have no "
                        f"column in the panel: {missing_from_panel[:10]}. The universe claims "
                        "names the data cannot supply."
                    ),
                    n_affected=len(missing_from_panel),
                )
            )
        extra = sorted(panel_symbols - set(universe))
        if extra:
            issues.append(
                Issue(
                    check="ticker_consistency",
                    severity=Severity.INFO,
                    detail=(
                        f"{len(extra)} ticker(s) in the panel are not in the frozen universe "
                        "(downloaded as candidates, then filtered out). Expected; recorded so "
                        "the difference is visible rather than assumed."
                    ),
                    n_affected=len(extra),
                )
            )

    if manifest_symbols is not None:
        manifest_set = set(manifest_symbols)
        orphans = sorted(panel_symbols - manifest_set)
        if orphans:
            issues.append(
                Issue(
                    check="ticker_consistency",
                    severity=Severity.WARN,
                    detail=(
                        f"{len(orphans)} panel column(s) have no manifest entry: {orphans[:10]}. "
                        "Their provenance is unknown --- they were not produced by a recorded "
                        "download."
                    ),
                    n_affected=len(orphans),
                )
            )

    return issues, per_ticker


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_quality_checks(
    raw_frames: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    quality_cfg: Any,
    volume: pd.DataFrame | None = None,
    universe: Sequence[str] | None = None,
    manifest_symbols: Iterable[str] | None = None,
    config_hash: str | None = None,
) -> QualityReport:
    """Run every check and assemble the report."""
    q = quality_cfg

    sessions, participation, padded = infer_sessions(
        close, volume=volume, quorum=q.session_quorum
    )
    _log.info(
        "inferred %d trading sessions from %d panel dates "
        "(quorum %.0f%%, %d padded non-trading day(s) excluded)",
        len(sessions),
        len(close.index),
        q.session_quorum * 100,
        len(padded),
    )

    # Market-wide stress per date: the fraction of listed names moving >5%.
    # Used to tell a crash apart from a name-specific move or a bad print.
    returns = close.pct_change(fill_method=None)
    big = (returns.abs() > 0.05).sum(axis=1)
    n_live = close.notna().sum(axis=1).replace(0, np.nan)
    market_stress = (big / n_live).fillna(0.0)

    report = QualityReport(
        sessions=sessions,
        padded_sessions=padded,
        n_symbols=len(close.columns),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_hash=config_hash,
        thresholds={
            "session_quorum": q.session_quorum,
            "extreme_move_pct": q.extreme_move_pct,
            "severe_move_pct": q.severe_move_pct,
            "corporate_action_window_days": q.corporate_action_window_days,
            "max_gap_sessions": q.max_gap_sessions,
            "min_coverage_over_life": q.min_coverage_over_life,
            "max_zero_volume_run": q.max_zero_volume_run,
            "zero_volume_frac_warn": q.zero_volume_frac_warn,
            "max_stale_price_run": q.max_stale_price_run,
            "phantom_date_max_tickers": q.phantom_date_max_tickers,
            "max_issues_listed_per_check": q.max_issues_listed_per_check,
        },
    )

    for symbol, frame in raw_frames.items():
        for issue in check_ticker(
            symbol,
            frame,
            extreme_move_pct=q.extreme_move_pct,
            severe_move_pct=q.severe_move_pct,
            corporate_action_window_days=q.corporate_action_window_days,
            max_zero_volume_run=q.max_zero_volume_run,
            zero_volume_frac_warn=q.zero_volume_frac_warn,
            max_stale_price_run=q.max_stale_price_run,
            market_stress=market_stress,
        ):
            report.add(issue)

    panel_issues, per_ticker = check_panel(
        close,
        sessions=sessions,
        participation=participation,
        padded=padded,
        volume=volume,
        universe=universe,
        manifest_symbols=manifest_symbols,
        max_gap_sessions=q.max_gap_sessions,
        min_coverage_over_life=q.min_coverage_over_life,
        phantom_date_max_tickers=q.phantom_date_max_tickers,
    )
    for issue in panel_issues:
        report.add(issue)
    report.per_ticker = per_ticker

    _log.info(
        "quality checks complete: %d issue group(s) --- %d FATAL, %d WARN",
        len(report.issues),
        len(report.fatal),
        len(report.warnings),
    )
    return report
