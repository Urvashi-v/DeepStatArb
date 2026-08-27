#!/usr/bin/env python
"""Build, freeze and report the research universe.

    python scripts/build_universe.py            # rebuild and report
    python scripts/build_universe.py --dry-run  # report without writing config

Reads the clean panels and the candidate list, evaluates point-in-time
eligibility at every walk-forward formation date, freezes the surviving names
into config/universe.yaml, and writes reports/universe/.

Deterministic: no random number is drawn. Running twice on an unchanged panel
produces the same universe hash and does not bump the version.

Exit codes: 0 success, 1 blocked (no data, or nothing survives selection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from dsa.config import load_config  # noqa: E402
from dsa.data.clean import load_panel  # noqa: E402
from dsa.data.quality import Severity, infer_sessions, run_quality_checks  # noqa: E402
from dsa.data.store import Manifest, raw_ohlcv_path, raw_reference_dir, read_parquet  # noqa: E402
from dsa.logging_utils import setup_logging  # noqa: E402
from dsa.provenance import capture_run_context  # noqa: E402
from dsa.universe import (  # noqa: E402
    EligibilityCriteria,
    FrozenUniverse,
    decision_dates,
    next_version,
    render_universe_yaml,
    select_universe,
    universe_hash,
    write_report,
    write_universe_yaml,
)

CANDIDATES_JSON = "candidates.json"


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing config")
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="skip the FATAL quality gate (faster; use only when iterating)",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    pd.set_option("display.width", 140)

    cfg = load_config()
    sel = cfg.selection
    ctx = capture_run_context(cfg)

    _banner("DeepStatArb --- research universe")
    print(f"run         : {ctx.run_id}")
    print(f"config hash : {cfg.hash}")

    # ------------------------------------------------------------- load data
    try:
        close = load_panel("close")
        turnover = load_panel("turnover")
        volume = load_panel("volume")
    except FileNotFoundError:
        print(
            "\nBLOCKED --- no clean panel on disk.\n"
            "Run `python scripts/build_dataset.py --stage all` first.\n"
            "No substitute dataset will be generated.",
            file=sys.stderr,
        )
        return 1

    sessions, _, padded = infer_sessions(close, volume, quorum=cfg.quality.session_quorum)
    print(f"panel       : {close.shape[0]} dates x {close.shape[1]} symbols")
    print(f"sessions    : {len(sessions)} ({len(padded)} padded non-trading day(s) excluded)")

    # --------------------------------------------------------- sector map
    candidates_path = raw_reference_dir() / CANDIDATES_JSON
    if not candidates_path.is_file():
        print(
            f"\nBLOCKED --- no {candidates_path}.\n"
            "Run `python scripts/build_dataset.py --stage candidates` first.",
            file=sys.stderr,
        )
        return 1
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    sectors = {c["symbol"]: c["sector"] for c in payload["candidates"]}
    provenance = payload["provenance"]

    # ----------------------------------------------------- structural gates
    manifest = Manifest.load()
    complete = set(manifest.completed()) if sel.require_complete_download else None

    quality_fatal: set[str] = set()
    if sel.exclude_quality_fatal and not args.skip_quality:
        print("\nrunning FATAL quality gate ...", flush=True)
        frames = {}
        for symbol in close.columns:
            path = raw_ohlcv_path(symbol)
            if path.is_file():
                frames[symbol] = read_parquet(path)
        report = run_quality_checks(
            frames, close, volume=volume, quality_cfg=cfg.quality, config_hash=cfg.hash
        )
        quality_fatal = {i.symbol for i in report.of_severity(Severity.FATAL) if i.symbol}
        print(f"  {len(quality_fatal)} symbol(s) with a FATAL finding")

    # --------------------------------------------------------- decision dates
    dates = decision_dates(
        sessions,
        formation_months=cfg.backtest.formation_months,
        trading_months=cfg.backtest.trading_months,
        step_months=cfg.backtest.step_months,
    )
    if not dates:
        print("\nBLOCKED --- the sample is too short for even one formation window.",
              file=sys.stderr)
        return 1
    print(f"\ndecision dates: {len(dates)} "
          f"({dates[0].date()} .. {dates[-1].date()}), "
          f"{cfg.backtest.formation_months}m formation / "
          f"{cfg.backtest.trading_months}m trading / {cfg.backtest.step_months}m step")

    # ------------------------------------------------------------- selection
    criteria = EligibilityCriteria.from_config(sel, cfg.backtest.metrics.trading_days_per_year)
    print(f"criteria      : turnover >= Rs{sel.min_median_turnover_inr / 1e7:.0f}cr over "
          f"{sel.liquidity_lookback_months}m ({criteria.lookback_sessions} sessions), "
          f"history >= {sel.min_history_years}y, price >= Rs{sel.min_price_inr:.0f}, "
          f"missing <= {sel.max_missing_frac:.0%}")
    print("\nevaluating point-in-time eligibility ...", flush=True)

    result = select_universe(
        close=close,
        turnover=turnover,
        volume=volume,
        sessions=sessions,
        sectors=sectors,
        criteria=criteria,
        dates=dates,
        complete_downloads=complete,
        quality_fatal=quality_fatal,
        require_sector=sel.require_sector,
        require_ever_eligible=sel.require_ever_eligible,
        max_size=sel.max_size,
        max_eligible_per_date=sel.max_eligible_per_date,
        rank_by=sel.rank_by,
    )

    if not result.symbols:
        print("\nBLOCKED --- no candidate survived selection.", file=sys.stderr)
        return 1

    uhash = universe_hash(result.symbols, result.sectors)
    version = next_version(result.symbols, result.sectors)

    # ---------------------------------------------------------------- output
    _banner("RESULT")
    print(f"  candidates                  : {result.n_candidates}")
    print(f"  frozen research universe    : {result.n_selected}")
    print(f"  excluded                    : {len(result.excluded)}")
    print(f"  universe hash               : {uhash}")
    print(f"  version                     : {version}")
    print(f"\n  unrestricted pairs          : {result.n_pairs:,}")
    print(f"  same-sector pairs           : {result.same_sector_pairs():,}")

    if not result.excluded.empty:
        tally: dict[str, int] = {}
        for reasons in result.excluded["reasons"]:
            for code in str(reasons).split(";"):
                if code:
                    tally[code] = tally.get(code, 0) + 1
        print("\n  exclusion reasons:")
        for code, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>4}  {code}")

    eligible_ts = result.eligible_over_time()
    print("\n  ELIGIBLE NAMES OVER TIME (the honest universe size)")
    for as_of, n in eligible_ts.items():
        bar = "#" * int(40 * n / max(eligible_ts.max(), 1))
        print(f"    {as_of}  {n:>4}  {bar}")
    print(f"\n    range {eligible_ts.min()}..{eligible_ts.max()}, "
          f"median {eligible_ts.median():.0f}")

    criteria_source = {
        "liquidity_lookback_months": sel.liquidity_lookback_months,
        "min_median_turnover_inr": sel.min_median_turnover_inr,
        "min_history_years": sel.min_history_years,
        "max_missing_frac": sel.max_missing_frac,
        "min_price_inr": sel.min_price_inr,
        "max_size": sel.max_size,
        "max_eligible_per_date": sel.max_eligible_per_date,
        "rank_by": sel.rank_by,
        "tie_break": sel.tie_break,
        "require_sector": sel.require_sector,
        "require_complete_download": sel.require_complete_download,
        "exclude_quality_fatal": sel.exclude_quality_fatal,
        "require_ever_eligible": sel.require_ever_eligible,
    }

    paths = write_report(
        result,
        universe_hash=uhash,
        version=version,
        config_hash=cfg.hash,
        criteria_source=criteria_source,
    )

    if args.dry_run:
        print("\n  --dry-run: config/universe.yaml NOT written")
    else:
        frozen = FrozenUniverse(
            name=cfg.universe.name,
            version=version,
            as_of=str(pd.Timestamp(dates[-1]).date()),
            symbols=result.symbols,
            sectors=result.sectors,
            hash=uhash,
            criteria=criteria_source,
            provenance={**provenance, "generated_at": ctx.started_at[:10]},
        )
        text = render_universe_yaml(
            frozen,
            start_date=str(cfg.universe.start_date),
            end_date=str(cfg.universe.end_date),
            smoke_test_tickers=cfg.universe.smoke_test_tickers,
            n_candidates=result.n_candidates,
            n_decision_dates=len(dates),
            first_decision=str(dates[0].date()),
            last_decision=str(dates[-1].date()),
            source_urls=provenance.get("source_urls", []),
            mitigation=cfg.universe.survivorship_bias.mitigation,
            target_size=sel.max_size,
        )
        written = write_universe_yaml(text)
        print(f"\n  wrote {written}")
        reloaded = load_config()
        print(f"  re-validated: {reloaded.summary()}")

    print("\n  reports:")
    for name, path in paths.items():
        print(f"    {name:<12} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
