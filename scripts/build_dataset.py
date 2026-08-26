#!/usr/bin/env python
"""Build the market-data layer, in four resumable stages.

    python scripts/build_dataset.py --stage all
    python scripts/build_dataset.py --stage download      # resume a part-done pull

Stages, in dependency order:

  candidates  fetch the NSE F&O list and sector map, intersect them
  download    pull daily OHLCV for every candidate into data/raw  (resumable)
  panels      derive adjusted, aligned panels into data/clean
  universe    apply history/turnover filters, freeze config/universe.yaml

``download`` is the only slow stage and the only one that touches the network
in bulk. It is safe to interrupt: progress is recorded in the manifest after
every ticker, and re-running skips whatever is already complete.

No credentials are required at any stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from dsa.config import load_config  # noqa: E402
from dsa.data.clean import build_panels  # noqa: E402
from dsa.data.download import DownloadConfig, download_universe  # noqa: E402
from dsa.data.store import Manifest, raw_reference_dir  # noqa: E402
from dsa.data.universe import (  # noqa: E402
    Candidate,
    build_candidates,
    finalise_universe,
    render_universe_yaml,
    same_sector_pair_count,
    sector_counts,
    write_universe_yaml,
)
from dsa.logging_utils import setup_logging  # noqa: E402
from dsa.paths import reports_dir  # noqa: E402
from dsa.provenance import capture_run_context  # noqa: E402

CANDIDATES_JSON = "candidates.json"

START = "2015-01-01"
END = "2026-08-01"


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def stage_candidates() -> list[Candidate]:
    _banner("STAGE 1/4  candidates --- NSE F&O list x sector map")
    candidates, provenance = build_candidates()

    path = raw_reference_dir() / CANDIDATES_JSON
    path.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "candidates": [c.__dict__ for c in candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    counts = sector_counts(candidates)
    n = len(candidates)
    print(f"\n  F&O individual securities   : {provenance['n_fo_symbols']}")
    print(f"  with an NSE sector          : {n}")
    print(f"  dropped (no sector)         : {len(provenance['unmapped'])}")
    print(f"  sectors                     : {counts.size}")
    print(f"\n  unrestricted candidate pairs: {n * (n - 1) // 2:,}")
    print(f"  same-sector candidate pairs : {same_sector_pair_count(candidates):,}")
    print(f"\n  sector distribution:\n{counts.to_string()}")
    print(f"\n  saved -> {path}")
    return candidates


def load_candidates() -> list[Candidate]:
    path = raw_reference_dir() / CANDIDATES_JSON
    if not path.is_file():
        raise SystemExit(f"no {path}. Run --stage candidates first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Candidate(**c) for c in payload["candidates"]]


# ---------------------------------------------------------------------------


def stage_download(candidates: list[Candidate], *, refresh: bool = False) -> None:
    _banner("STAGE 2/4  download --- daily OHLCV into data/raw")
    symbols = [c.symbol for c in candidates]
    cfg = DownloadConfig(start=START, end=END, min_interval_s=0.30, jitter_s=0.20)

    started = time.monotonic()

    def progress(i: int, total: int, symbol: str) -> None:
        elapsed = time.monotonic() - started
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        print(
            f"  [{i:>4}/{total}] {symbol:<18} {elapsed:6.0f}s elapsed, ~{eta:.0f}s left",
            flush=True,
        )

    result = download_universe(symbols, config=cfg, resume=not refresh, refresh=refresh,
                              progress=progress)

    print(f"\n  {result.summary()}")
    if result.failed:
        print(f"\n  FAILED ({len(result.failed)}): {result.failed}")
    if result.empty:
        print(f"\n  EMPTY ({len(result.empty)}): {result.empty}")
    print(f"\n  {result.manifest.summary()}")


# ---------------------------------------------------------------------------


def stage_panels(candidates: list[Candidate]) -> None:
    _banner("STAGE 3/4  panels --- adjusted, aligned panels into data/clean")
    available = set(Manifest.load().completed())
    symbols = [c.symbol for c in candidates if c.symbol in available]
    print(f"  {len(symbols)} symbols with a complete raw download")

    panels, summary = build_panels(symbols, start=START, end=END, write=True)
    print(f"\n  {summary.summary()}")
    close = panels["close"]
    coverage = close.notna().sum().sum() / (close.shape[0] * close.shape[1])
    print(f"  panel shape                 : {close.shape[0]} dates x {close.shape[1]} symbols")
    print(f"  observed cells              : {coverage:.1%}")
    if summary.dropped:
        print(f"\n  dropped {len(summary.dropped)} symbol(s):")
        for sym, why in list(summary.dropped.items())[:15]:
            print(f"    {sym:<18} {why}")


# ---------------------------------------------------------------------------


def stage_universe(candidates: list[Candidate]) -> None:
    _banner("STAGE 4/4  universe --- filter and freeze")
    cfg = load_config()
    u = cfg.universe

    survivors, diagnostics = finalise_universe(
        candidates,
        min_history_years=u.min_history_years,
        min_median_turnover_inr=u.min_median_turnover_inr,
        max_missing_frac=u.max_missing_frac,
        start=START,
        end=END,
    )

    report = reports_dir() / "universe_selection.csv"
    diagnostics.to_csv(report, index=False)

    payload = json.loads((raw_reference_dir() / CANDIDATES_JSON).read_text(encoding="utf-8"))
    provenance = payload["provenance"]

    text = render_universe_yaml(
        survivors,
        provenance,
        name=u.name,
        version=u.version + 1,
        start_date=START,
        end_date=END,
        target_size=u.target_size,
        min_history_years=u.min_history_years,
        min_median_turnover_inr=u.min_median_turnover_inr,
        max_missing_frac=u.max_missing_frac,
        mitigation=u.survivorship_bias.mitigation,
        smoke_test_tickers=u.smoke_test_tickers,
        n_candidates=len(candidates),
    )
    path = write_universe_yaml(text)

    n = len(survivors)
    counts = sector_counts(survivors)
    print(f"\n  candidates                  : {len(candidates)}")
    print(f"  passing all filters         : {n}")
    print(f"  dropped                     : {len(candidates) - n}")

    if "reason" in diagnostics.columns:
        dropped = diagnostics[diagnostics.get("passes", False) != True]  # noqa: E712
        if len(dropped):
            print("\n  drop reasons:")
            for reason, cnt in dropped["reason"].value_counts().head(12).items():
                print(f"    {cnt:>4}  {str(reason)[:70]}")

    print("\n  THE FUNNEL SO FAR (spec Sec 4.1)")
    print(f"    unrestricted candidate pairs : {n * (n - 1) // 2:,}")
    print(f"    same-sector candidate pairs  : {same_sector_pair_count(survivors):,}")
    print("    -> cointegration screening is Day 4-6")
    print(f"\n  sectors ({counts.size}):\n{counts.to_string()}")
    print(f"\n  wrote {path}")
    print(f"  wrote {report}")

    reloaded = load_config()
    print(f"\n  re-validated: {reloaded.summary()}")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["candidates", "download", "panels", "universe", "all"],
        default="all",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even completed tickers (the only path that overwrites data/raw)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, run_id=None)
    ctx = capture_run_context(load_config())
    print(f"run: {ctx.run_id}")

    pd.set_option("display.width", 140)

    candidates = (
        stage_candidates() if args.stage in {"candidates", "all"} else load_candidates()
    )

    if args.stage in {"download", "all"}:
        stage_download(candidates, refresh=args.refresh)
    if args.stage in {"panels", "all"}:
        stage_panels(candidates)
    if args.stage in {"universe", "all"}:
        stage_universe(candidates)

    return 0


if __name__ == "__main__":
    sys.exit(main())
