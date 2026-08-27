"""The universe report.

Answers, from an actual run and never by hand: how big is the universe, what
is in it, who was excluded and why, how liquid is it, and how well is it
covered.

The number worth arguing about is not the headline size. It is the
**eligible-per-decision-date series**: the frozen list is one number, but the
set of names you could actually have traded changes over the sample, and a
report that shows only the frozen count invites the reader to assume all of
them were tradeable in 2015. They were not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dsa.logging_utils import get_logger
from dsa.paths import reports_dir
from dsa.universe.selection import REASON_CODES, SelectionResult

__all__ = ["build_report", "write_report"]

_log = get_logger(__name__)


def _fmt_cr(value: float) -> str:
    """Rupees as crore, the unit an Indian desk actually speaks."""
    return f"{value / 1e7:,.1f}"


def build_report(
    result: SelectionResult,
    *,
    universe_hash: str,
    version: int,
    config_hash: str | None,
    criteria_source: Mapping[str, Any],
    generated_at: str | None = None,
) -> str:
    """Render the report as markdown."""
    lines: list[str] = []
    add = lines.append

    symbols = list(result.symbols)
    counts = pd.Series([result.sectors[s] for s in symbols]).value_counts()
    eligible_ts = result.eligible_over_time()
    last_date = str(result.decision_dates[-1].date())
    latest = result.schedule[result.schedule["as_of"] == last_date].set_index("symbol")
    latest_sel = latest.loc[[s for s in symbols if s in latest.index]]

    add("# Research universe")
    add("")
    add(f"generated     : {generated_at or datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"universe hash : `{universe_hash}`")
    add(f"version       : {version}")
    add(f"config hash   : `{config_hash}`")
    add("")

    # ---------------------------------------------------------------- size
    add("## Size")
    add("")
    add(f"- candidates with price data: **{result.n_candidates}**")
    add(f"- frozen research universe: **{len(symbols)}** names across **{counts.size}** sectors")
    add(f"- unrestricted pairs: **{result.n_pairs:,}**")
    add(f"- same-sector pairs: **{result.same_sector_pairs():,}** "
        f"({1 - result.same_sector_pairs() / max(result.n_pairs, 1):.1%} of the search "
        "space removed before a single test is run)")
    add("")
    add("At a 5% test level, the same-sector pair count alone would yield roughly "
        f"**{0.05 * result.same_sector_pairs():.0f}** 'cointegrated' pairs in a world where "
        "none are. That is what the BH FDR control exists for (spec Sec 4.1).")
    add("")

    # ------------------------------------------------- point-in-time size
    add("## Eligible names over time")
    add("")
    add("The frozen list is a single number; the set you could actually have traded")
    add("is not. Eligibility is re-evaluated at each formation date using only data")
    add("available then, so a name that became liquid in 2022 is ineligible before it.")
    add("")
    add("| formation date | eligible | share of frozen universe |")
    add("|---|---:|---:|")
    for as_of, n in eligible_ts.items():
        add(f"| {as_of} | {n} | {n / max(len(symbols), 1):.0%} |")
    add("")
    if len(eligible_ts):
        add(f"Range: **{eligible_ts.min()}** to **{eligible_ts.max()}** eligible names "
            f"(median {eligible_ts.median():.0f}).")
        add("")

    # ------------------------------------------------------------- sectors
    add("## Sectors")
    add("")
    add("Classification is NSE's own `Industry` field, not one written here. That")
    add("matters: the same-sector prior (Sec 4.1) and the survival stratification")
    add("(Sec 7.2) are only defensible if the grouping came from somewhere other")
    add("than the person hoping to find pairs in it.")
    add("")
    add("| sector | names | same-sector pairs |")
    add("|---|---:|---:|")
    for sector, n in counts.items():
        add(f"| {sector} | {n} | {n * (n - 1) // 2:,} |")
    add(f"| **total** | **{counts.sum()}** | **{result.same_sector_pairs():,}** |")
    add("")
    singletons = [s for s, n in counts.items() if n < 2]
    if singletons:
        add(f"{len(singletons)} sector(s) hold a single name and can therefore form no "
            f"same-sector pair at all: {', '.join(singletons)}.")
        add("")

    # ----------------------------------------------------------- exclusions
    add("## Exclusions")
    add("")
    excluded = result.excluded
    if excluded.empty:
        add("_no candidate was excluded_")
    else:
        add(f"**{len(excluded)}** of {result.n_candidates} candidates excluded.")
        add("")
        tally: dict[str, int] = {}
        for reasons in excluded["reasons"]:
            for code in str(reasons).split(";"):
                if code:
                    tally[code] = tally.get(code, 0) + 1
        add("| reason | names | what it means |")
        add("|---|---:|---|")
        for code, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            add(f"| `{code}` | {n} | {REASON_CODES.get(code, '')} |")
        add("")
        add("Every excluded name, with the statistics behind the decision, is in")
        add("`exclusions.csv`. Nothing is dropped without a recorded reason.")
        add("")

    # ---------------------------------------------------- liquidity stats
    add("## Liquidity")
    add("")
    add(f"Trailing median daily turnover at the last decision date ({last_date}), in Rs crore.")
    add("Measured over traded sessions only --- a zero-volume halt is not a session")
    add("you could have traded, whatever the price column says.")
    add("")
    if not latest_sel.empty:
        turn = latest_sel["median_turnover_inr"].sort_values(ascending=False)
        quantiles = [0.0, 0.25, 0.50, 0.75, 1.0]
        add("| statistic | Rs crore |")
        add("|---|---:|")
        add(f"| minimum | {_fmt_cr(turn.min())} |")
        for q in quantiles[1:-1]:
            add(f"| {int(q * 100)}th percentile | {_fmt_cr(turn.quantile(q))} |")
        add(f"| maximum | {_fmt_cr(turn.max())} |")
        add(f"| **median** | **{_fmt_cr(turn.median())}** |")
        add("")
        add("Ten most liquid, and ten least:")
        add("")
        add("| most liquid | Rs cr | least liquid | Rs cr |")
        add("|---|---:|---|---:|")
        top, bottom = turn.head(10), turn.tail(10).iloc[::-1]
        for (ts, tv), (bs, bv) in zip(top.items(), bottom.items(), strict=False):
            add(f"| {ts} | {_fmt_cr(tv)} | {bs} | {_fmt_cr(bv)} |")
        add("")

    # ------------------------------------------------------------ coverage
    add("## Coverage")
    add("")
    if not latest_sel.empty:
        hist = latest_sel["history_years"]
        miss = latest_sel["missing_frac"]
        add(f"- listed history at {last_date}: median **{hist.median():.1f}y**, "
            f"minimum **{hist.min():.1f}y**, maximum **{hist.max():.1f}y**")
        add(f"- missing/untradeable sessions in the trailing window: median "
            f"**{miss.median():.2%}**, worst **{miss.max():.2%}**")
        add(f"- names with a full trailing window: "
            f"**{int((miss == 0).sum())}/{len(latest_sel)}**")
        add("")
        worst = latest_sel.nlargest(min(8, len(latest_sel)), "missing_frac")
        if float(worst["missing_frac"].max()) > 0:
            add("Least complete names in the trailing window:")
            add("")
            add("| symbol | history (y) | missing | median turnover (Rs cr) |")
            add("|---|---:|---:|---:|")
            for symbol, row in worst.iterrows():
                if row["missing_frac"] <= 0:
                    continue
                add(f"| {symbol} | {row['history_years']:.1f} | {row['missing_frac']:.2%} "
                    f"| {_fmt_cr(row['median_turnover_inr'])} |")
            add("")

    # ------------------------------------------------------------ criteria
    add("## Criteria applied")
    add("")
    add("From `config/selection.yaml`. Every point-in-time criterion is evaluated")
    add("using only data available on or before the decision date.")
    add("")
    add("| parameter | value |")
    add("|---|---|")
    for key in sorted(criteria_source):
        add(f"| `{key}` | {criteria_source[key]} |")
    add("")

    # --------------------------------------------------------------- bias
    add("## Bias declaration")
    add("")
    add("**Survivorship: not eliminated.** The candidate pool is today's NSE F&O")
    add("list. Names dropped from F&O, delisted, or shrunk out of the segment are")
    add("absent, and their absence flatters every downstream result. Direction:")
    add("**optimistic**. Point-in-time F&O constituent history is not available")
    add("free, so this is declared rather than fixed.")
    add("")
    add("**Lookahead in selection: eliminated.** Liquidity, history, price and")
    add("coverage are computed from a trailing window ending at the decision date.")
    add("Appending future data to the panel does not change any eligibility")
    add("decision already made --- asserted by a test that re-runs the selection")
    add("against a truncated panel and requires identical output.")
    add("")
    return "\n".join(lines)


def write_report(
    result: SelectionResult,
    *,
    universe_hash: str,
    version: int,
    config_hash: str | None,
    criteria_source: Mapping[str, Any],
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write the report and its backing CSVs."""
    out = directory or (reports_dir() / "universe")
    out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    paths = {
        "summary": out / "summary.md",
        "universe": out / "universe.csv",
        "exclusions": out / "exclusions.csv",
        "eligibility": out / "eligibility.csv",
        "meta": out / "meta.json",
    }

    paths["summary"].write_text(
        build_report(
            result,
            universe_hash=universe_hash,
            version=version,
            config_hash=config_hash,
            criteria_source=criteria_source,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )

    last_date = str(result.decision_dates[-1].date())
    latest = result.schedule[result.schedule["as_of"] == last_date].set_index("symbol")
    selected = pd.DataFrame(
        {
            "symbol": list(result.symbols),
            "sector": [result.sectors[s] for s in result.symbols],
        }
    )
    for col in ("history_years", "median_turnover_inr", "missing_frac", "last_price"):
        selected[col] = [
            latest.loc[s, col] if s in latest.index else None for s in result.symbols
        ]
    selected.to_csv(paths["universe"], index=False)
    result.excluded.to_csv(paths["exclusions"], index=False)
    result.schedule.to_csv(paths["eligibility"], index=False)

    eligible_ts = result.eligible_over_time()
    paths["meta"].write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "universe_hash": universe_hash,
                "version": version,
                "config_hash": config_hash,
                "n_candidates": result.n_candidates,
                "n_selected": result.n_selected,
                "n_pairs": result.n_pairs,
                "n_same_sector_pairs": result.same_sector_pairs(),
                "decision_dates": [str(d.date()) for d in result.decision_dates],
                "eligible_over_time": {str(k): int(v) for k, v in eligible_ts.items()},
                "criteria": dict(criteria_source),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _log.info("universe report written to %s", out)
    return paths


def sector_table(symbols: Sequence[str], sectors: Mapping[str, str]) -> pd.DataFrame:
    counts = pd.Series([sectors[s] for s in symbols]).value_counts()
    return pd.DataFrame(
        {
            "sector": counts.index,
            "names": counts.to_numpy(),
            "same_sector_pairs": (counts * (counts - 1) // 2).to_numpy(),
        }
    )
