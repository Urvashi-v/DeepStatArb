#!/usr/bin/env python
"""Generate the reproducible data-quality report.

    python scripts/run_quality_report.py

Reads data/raw and data/clean, runs every check in dsa.data.quality, and
writes reports/data_quality/{summary.md, issues.csv, per_ticker.csv, meta.json}.

Nothing is deleted, repaired or interpolated. The report is a list of things
for a person to look at.

Exit codes: 0 no FATAL findings, 1 at least one FATAL finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from dsa.config import load_config  # noqa: E402
from dsa.data.clean import load_panel  # noqa: E402
from dsa.data.quality import Severity, run_quality_checks  # noqa: E402
from dsa.data.store import Manifest, raw_ohlcv_path, read_parquet  # noqa: E402
from dsa.logging_utils import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe-only",
        action="store_true",
        help="check only the frozen universe rather than every downloaded candidate",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    pd.set_option("display.width", 140)

    cfg = load_config()
    manifest = Manifest.load()

    try:
        close = load_panel("close")
        volume = load_panel("volume")
    except FileNotFoundError:
        print(
            "BLOCKED --- no clean panel on disk.\n"
            "Run `python scripts/build_dataset.py --stage all` first.\n"
            "No substitute dataset will be generated.",
            file=sys.stderr,
        )
        return 1

    symbols = list(cfg.universe.tickers) if args.universe_only else list(close.columns)
    symbols = [s for s in symbols if s in close.columns]

    print("=" * 78)
    print("DeepStatArb --- data quality report")
    print("=" * 78)
    print(f"config      : {cfg.hash}")
    print(f"scope       : {'frozen universe' if args.universe_only else 'all downloaded'}")
    print(f"symbols     : {len(symbols)}")
    print(f"panel       : {close.shape[0]} dates x {close.shape[1]} columns")
    print("\nreading raw frames ...", flush=True)

    raw_frames = {}
    for symbol in symbols:
        path = raw_ohlcv_path(symbol)
        if path.is_file():
            raw_frames[symbol] = read_parquet(path)
    print(f"  {len(raw_frames)} raw frames loaded")

    print("running checks ...", flush=True)
    report = run_quality_checks(
        raw_frames,
        close[symbols],
        volume=volume[symbols],
        quality_cfg=cfg.quality,
        universe=cfg.universe.tickers if cfg.universe.frozen else None,
        manifest_symbols=manifest.completed(),
        config_hash=cfg.hash,
    )

    paths = report.write()

    print("\n" + "-" * 78)
    print("COUNTS BY CHECK")
    print("-" * 78)
    counts = report.counts()
    print(counts.to_string(index=False) if not counts.empty else "  (no issues)")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    n_flagged = sum(i.n_affected for i in report.warnings)
    if report.fatal:
        print(f"  {len(report.fatal)} FATAL finding(s) --- structurally impossible data present:")
        for issue in report.fatal[:10]:
            print(f"    [{issue.symbol or '-'}] {issue.detail[:110]}")
    else:
        print("  No FATAL findings. No duplicated bars, no non-monotonic indexes,")
        print("  no non-positive prices, no impossible OHLC.")
    print(f"  {n_flagged} observation(s) flagged for inspection "
          f"across {len(report.warnings)} warning group(s).")
    print(f"  {len(report.of_severity(Severity.INFO))} informational finding(s).")
    if len(report.padded_sessions):
        print("")
        print(f"  {len(report.padded_sessions)} PADDED NON-TRADING DAY(S) identified and")
        print("  excluded from the inferred session calendar:")
        for d in report.padded_sessions:
            print(f"    {d.date()}")
    print("")
    print(
        f"  inferred trading sessions: {len(report.sessions)} "
        f"(from {close.shape[0]} panel dates)"
    )
    print("\n  Nothing was deleted, repaired or interpolated.")

    print("\n" + "-" * 78)
    print("WRITTEN")
    print("-" * 78)
    for name, path in paths.items():
        print(f"  {name:<12} {path}")

    return 1 if report.fatal else 0


if __name__ == "__main__":
    sys.exit(main())
