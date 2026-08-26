#!/usr/bin/env python
"""Connection and data-availability check.

    python scripts/check_data_connection.py

Answers one question before any download is attempted: **can this machine
actually reach the real data, and does it come back looking like NSE data?**

It makes a small number of live requests --- no bulk download --- and checks
the shape and plausibility of what returns. It never falls back to cached or
synthetic data: if a source is unreachable, this script says so and exits
non-zero. That is the whole point of it.

Exit codes: 0 all sources healthy, 1 at least one source unusable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dsa.config import load_config  # noqa: E402
from dsa.data.download import DownloadConfig, download_ticker  # noqa: E402
from dsa.data.sources import (  # noqa: E402
    NSE_FO_LOTS_URL,
    NSE_NIFTY500_URL,
    RateLimiter,
    fetch_fo_underlyings,
    fetch_index_constituents,
)
from dsa.logging_utils import setup_logging  # noqa: E402

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<28} {detail}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    setup_logging("WARNING")
    failures = 0
    warnings = 0

    print("=" * 78)
    print("DeepStatArb --- data connection and availability check")
    print("=" * 78)

    section("Credentials")
    line(OK, "authentication", "NONE REQUIRED --- all sources are public")
    line(OK, "api keys", "none read, none needed")

    # ---------------------------------------------------------------- NSE F&O
    section("NSE F&O eligibility list (universe source)")
    fo = None
    try:
        fo = fetch_fo_underlyings()
        line(OK, "fo_mktlots.csv", f"{len(fo)} individual securities")
        if not 150 <= len(fo) <= 260:
            line(WARN, "count", f"{len(fo)} outside the expected 180-220 band")
            warnings += 1
        line(OK, "sample", ", ".join(fo["symbol"].head(5).tolist()))
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "fo_mktlots.csv", f"{type(exc).__name__}: {str(exc)[:150]}")
        line(FAIL, "url", NSE_FO_LOTS_URL)
        failures += 1

    # ------------------------------------------------------------ NSE sectors
    section("NSE sector classification (economic prior source)")
    idx = None
    try:
        idx = fetch_index_constituents()
        line(OK, "ind_nifty500list.csv", f"{len(idx)} symbols, {idx['sector'].nunique()} sectors")
        line(OK, "sample sector", f"{idx['symbol'].iloc[0]} -> {idx['sector'].iloc[0]}")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "ind_nifty500list.csv", f"{type(exc).__name__}: {str(exc)[:150]}")
        line(FAIL, "url", NSE_NIFTY500_URL)
        failures += 1

    if fo is not None and idx is not None:
        section("Coverage")
        sectors = set(idx["symbol"])
        covered = [s for s in fo["symbol"] if s in sectors]
        pct = 100.0 * len(covered) / max(len(fo), 1)
        status = OK if pct >= 90 else WARN
        line(status, "F&O names with a sector", f"{len(covered)}/{len(fo)} ({pct:.1f}%)")
        if pct < 90:
            warnings += 1

    # ------------------------------------------------------------ price feed
    section("Price feed (Yahoo Finance, NSE tickers)")
    cfg = load_config()
    probes = list(cfg.universe.smoke_test_tickers)[:3]
    dl = DownloadConfig(start="2024-01-01", end="2024-03-01")
    limiter = RateLimiter(min_interval_s=0.5, jitter_s=0.3)

    for symbol in probes:
        try:
            df, attempts = download_ticker(symbol, config=dl, limiter=limiter)
        except Exception as exc:  # noqa: BLE001
            line(FAIL, symbol, f"{type(exc).__name__}: {str(exc)[:120]}")
            failures += 1
            continue

        issues = []
        if df.empty:
            issues.append("no rows")
        if not (df["Close"] > 0).all():
            issues.append("non-positive close")
        if (df["High"] < df["Low"]).any():
            issues.append("high < low")
        factor = df["Adj Close"] / df["Close"]
        if not ((factor > 0) & (factor <= 1.0 + 1e-9)).all():
            issues.append(f"adj factor out of (0,1]: [{factor.min():.4f}, {factor.max():.4f}]")

        if issues:
            line(FAIL, symbol, "; ".join(issues))
            failures += 1
        else:
            line(
                OK,
                symbol,
                f"{len(df)} bars {df.index.min().date()}..{df.index.max().date()}, "
                f"close {df['Close'].iloc[-1]:.2f} INR, {attempts} attempt(s)",
            )

    # ------------------------------------------------------ full-history span
    section("Full-history availability")
    if probes:
        try:
            probe = probes[0]
            full = DownloadConfig(
                start=str(cfg.universe.start_date), end=str(cfg.universe.end_date)
            )
            df, _ = download_ticker(probe, config=full, limiter=limiter)
            years = (df.index.max() - df.index.min()).days / 365.25
            per_year = len(df) / max(years, 1e-9)
            line(
                OK,
                f"{probe} span",
                f"{len(df)} bars, {years:.1f}y ({df.index.min().date()}..{df.index.max().date()})",
            )
            status = OK if 200 <= per_year <= 270 else WARN
            line(status, "bars per year", f"{per_year:.0f} (NSE trades ~250 days/year)")
            if status is WARN:
                warnings += 1
            n_actions = int(((df["Dividends"] != 0) | (df["Stock Splits"] != 0)).sum())
            line(OK, "corporate actions", f"{n_actions} dividend/split events recorded")
        except Exception as exc:  # noqa: BLE001
            line(FAIL, "full history", f"{type(exc).__name__}: {str(exc)[:150]}")
            failures += 1

    # ------------------------------------------------------------------- done
    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {failures} source(s) UNUSABLE, {warnings} warning(s).")
        print("Do not proceed to download until these are resolved.")
        print("No synthetic or cached substitute will be used.")
    elif warnings:
        print(f"RESULT: all sources reachable, {warnings} warning(s) above.")
    else:
        print("RESULT: all sources reachable and returning plausible NSE data.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
