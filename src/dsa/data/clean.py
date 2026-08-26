"""Raw -> clean: adjusted, aligned price panels.

This module never writes to ``data/raw``. It reads raw parquet, derives
adjusted series, and writes panels to ``data/clean``. If a decision here turns
out to be wrong, you change it and rebuild; the input is still on disk.

The adjustment, and why it is derived rather than requested
-----------------------------------------------------------
Yahoo will hand you adjusted prices if you pass ``auto_adjust=True``. This
module does not do that. It stores ``Close`` and ``Adj Close`` separately and
computes

    factor(t) = Adj Close(t) / Close(t)

then applies that one factor to Open, High and Low as well. Two reasons:

* **Auditable.** The factor is a number you can plot, difference, and check
  against the ``Dividends`` and ``Stock Splits`` columns. A flag passed to a
  library is not.
* **Consistent across the bar.** The execution model fills at the *open* of
  t+1 (spec Sec 6.3). Open and Close must therefore carry the same adjustment,
  or the overnight return is contaminated by an adjustment mismatch on any day
  following a corporate action.

Verified on RELIANCE.NS over 2015-2026: reconstructing ``Adj Close`` from
``Close`` and the reported dividends alone (no split term) reproduces Yahoo's
figure to a maximum relative error of 3.2e-07, which is float32 storage
precision. So the factor is exactly the cumulative dividend adjustment, and
splits are already inside ``Close``.

Two caveats that travel with every number computed downstream
-------------------------------------------------------------
1. **``Close`` is already split-adjusted at source.** The 2:1 RELIANCE split
   on 2017-09-07 appears as -0.56%, not -50%. Consequence for spec Sec 11.2:
   the ">20% single-day move" screen will *not* surface splits in this feed,
   because the provider already handled them. What it will surface is genuine
   moves and splits the provider handled *badly* --- which is the case worth
   finding anyway.
2. **Back-adjustment puts future information into past levels.** The adjusted
   2015 price is the split-adjusted close times the product of every dividend
   factor from 2015 to today. Strictly, the 2015 series could not have been
   computed in 2015. The spec mandates adjusted prices (Sec 11.2) and the
   alternative --- dividend jumps in the spread --- is worse, so this is
   accepted and declared rather than fixed. It is worth knowing that the
   hedge ratio estimated on back-adjusted prices is not exactly the one a
   trader would have estimated in real time.

Volume is split-adjusted at source too (checked across the same split), so
``Close x Volume`` is a consistent turnover measure through corporate actions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dsa.data.store import clean_path, raw_ohlcv_path, read_parquet, write_parquet
from dsa.logging_utils import get_logger

__all__ = [
    "CleanError",
    "adjust_ohlcv",
    "build_panels",
    "load_panel",
    "PANEL_FIELDS",
    "PanelSummary",
]

_log = get_logger(__name__)

PANEL_FIELDS = ("open", "high", "low", "close", "volume", "turnover")

# Above this, the adjustment factor is doing something other than compounding
# dividends and the series needs looking at rather than using.
_MAX_PLAUSIBLE_FACTOR = 1.0 + 1e-9
_MIN_PLAUSIBLE_FACTOR = 0.05


class CleanError(RuntimeError):
    """Raw data failed a structural check that cleaning cannot paper over."""


@dataclass(frozen=True)
class PanelSummary:
    """What was built, for the run log."""

    n_symbols: int
    n_rows: int
    start: str
    end: str
    dropped: dict[str, str]

    def summary(self) -> str:
        return (
            f"panel: {self.n_symbols} symbols x {self.n_rows} trading days "
            f"({self.start} .. {self.end}), {len(self.dropped)} symbol(s) dropped"
        )


def _to_naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Normalise a tz-aware intraday-stamped index to plain trading dates.

    yfinance returns ``2015-01-01 00:00:00+05:30``. Two tickers whose bars
    carry different tz offsets would not align on a join, and the time part
    carries no information for a daily bar. Dropping to a naive date makes
    alignment exact and comparison across tickers trivially correct.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx.normalize(), name="date")


def adjust_ohlcv(raw: pd.DataFrame, symbol: str = "?") -> pd.DataFrame:
    """Apply the total-return adjustment factor to a raw OHLCV frame.

    Returns a frame indexed by trading date with columns ``open, high, low,
    close, volume, turnover, adj_factor, dividends, splits``.
    """
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(raw.columns)
    if missing:
        raise CleanError(f"{symbol}: raw frame missing {sorted(missing)}")

    df = raw.copy()
    df.index = _to_naive_dates(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    close = pd.to_numeric(df["Close"], errors="coerce")
    adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")

    # A zero or negative close is not a price. Drop rather than divide by it.
    bad = ~(close > 0) | ~(adj_close > 0)
    if bad.any():
        _log.warning("%s: dropping %d row(s) with non-positive close", symbol, int(bad.sum()))
        df, close, adj_close = df[~bad], close[~bad], adj_close[~bad]

    if df.empty:
        raise CleanError(f"{symbol}: no usable rows after removing non-positive prices")

    factor = adj_close / close

    if not np.isfinite(factor).all():
        raise CleanError(f"{symbol}: adjustment factor has non-finite values")
    if (factor < _MIN_PLAUSIBLE_FACTOR).any() or (factor > _MAX_PLAUSIBLE_FACTOR).any():
        lo, hi = float(factor.min()), float(factor.max())
        raise CleanError(
            f"{symbol}: adjustment factor outside [{_MIN_PLAUSIBLE_FACTOR}, 1.0] "
            f"(min={lo:.6f}, max={hi:.6f}). A total-return back-adjustment factor is a "
            "product of (1 - dividend/price) terms, so it must lie in (0, 1] and rise "
            "monotonically to 1 at the last bar. Something else is going on --- inspect "
            "the raw file before cleaning around it."
        )

    out = pd.DataFrame(index=df.index)
    for src, dst in (("Open", "open"), ("High", "high"), ("Low", "low")):
        out[dst] = pd.to_numeric(df[src], errors="coerce") * factor
    out["close"] = adj_close
    out["volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    # Turnover in unadjusted rupees: the rupees that actually changed hands, which is
    # what a liquidity filter must be based on. Uses the split-adjusted close
    # and split-adjusted volume, whose product is invariant to splits.
    out["turnover"] = close * out["volume"]
    out["adj_factor"] = factor
    out["dividends"] = pd.to_numeric(df.get("Dividends", 0.0), errors="coerce").fillna(0.0)
    out["splits"] = pd.to_numeric(df.get("Stock Splits", 0.0), errors="coerce").fillna(0.0)

    # OHLC coherence. A high below the low is a bad print, not a price.
    incoherent = (out["high"] < out["low"]) | (out["high"] < out["close"]) | (
        out["low"] > out["close"]
    )
    n_bad = int(incoherent.sum())
    if n_bad:
        _log.warning("%s: %d bar(s) with incoherent OHLC; flagged, not altered", symbol, n_bad)

    return out


def build_panels(
    symbols: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    min_rows: int = 500,
    write: bool = True,
) -> tuple[dict[str, pd.DataFrame], PanelSummary]:
    """Build aligned wide panels (dates x symbols) for each price field.

    Alignment policy
    ----------------
    The panel index is the union of trading dates across symbols. Missing bars
    are left as **NaN** and are not forward-filled here. Forward-filling a
    halted or suspended stock invents a flat price, which reads to a
    cointegration test as an unusually well-behaved spread --- the exact
    direction that manufactures false pairs. Handling gaps is a Day 3 decision
    made with the gap statistics in front of us, not a default buried in a
    loader.
    """
    frames: dict[str, pd.DataFrame] = {}
    dropped: dict[str, str] = {}

    for symbol in symbols:
        path = raw_ohlcv_path(symbol)
        if not path.is_file():
            dropped[symbol] = "no raw file"
            continue
        try:
            adjusted = adjust_ohlcv(read_parquet(path), symbol)
        except CleanError as exc:
            dropped[symbol] = str(exc)[:200]
            _log.warning("dropping %s: %s", symbol, exc)
            continue
        if len(adjusted) < min_rows:
            dropped[symbol] = f"only {len(adjusted)} rows (< {min_rows})"
            continue
        frames[symbol] = adjusted

    if not frames:
        raise CleanError(
            "no symbols survived cleaning. Nothing can be built on top of this --- "
            "check data/raw and the download manifest before going further."
        )

    if start is not None or end is not None:
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        frames = {s: f.loc[lo:hi] for s, f in frames.items()}

    panels: dict[str, pd.DataFrame] = {}
    for field in PANEL_FIELDS:
        panels[field] = pd.DataFrame({s: f[field] for s, f in frames.items()}).sort_index()

    # Carry the corporate-action columns through: Day 3's quality gate needs
    # them, and re-reading every raw file to get them would be wasteful.
    panels["adj_factor"] = pd.DataFrame({s: f["adj_factor"] for s, f in frames.items()}).sort_index()

    index = panels["close"].index
    summary = PanelSummary(
        n_symbols=len(frames),
        n_rows=len(index),
        start=str(index.min().date()) if len(index) else "-",
        end=str(index.max().date()) if len(index) else "-",
        dropped=dropped,
    )

    if write:
        for name, panel in panels.items():
            write_parquet(panel, clean_path(f"panel_{name}"))
        _log.info("wrote %d clean panels to %s", len(panels), clean_path("panel_close").parent)

    _log.info("%s", summary.summary())
    return panels, summary


def load_panel(field: str = "close") -> pd.DataFrame:
    """Load one clean panel. The single entry point for analysis code."""
    path = clean_path(f"panel_{field}")
    if not path.is_file():
        raise FileNotFoundError(
            f"no clean panel at {path}. Run `python scripts/download_data.py` then "
            "`python scripts/build_panels.py` first."
        )
    return read_parquet(path)


def median_turnover(symbols: Iterable[str] | None = None) -> pd.Series:
    """Median daily traded value per symbol, from the clean turnover panel.

    The liquidity criterion for universe construction (spec Sec 11.4).
    """
    panel = load_panel("turnover")
    if symbols is not None:
        keep = [s for s in symbols if s in panel.columns]
        panel = panel[keep]
    return panel.median(axis=0, skipna=True).sort_values(ascending=False)
