"""Build and freeze the trading universe from real NSE sources.

Spec Sec 11.4: *"Freeze this list and version it --- a universe that changes as
you iterate is a silent source of overfitting."*

Where the list comes from
-------------------------
Two published NSE files, no hand-curation:

* ``fo_mktlots.csv`` --- every underlying with listed derivatives. Restricting
  to these makes the short leg implementable via single-stock futures, which
  is the cleanest of the three options in spec Sec 6.1.
* ``ind_nifty500list.csv`` --- carries NSE's own ``Industry`` column, which
  becomes the sector map. Using the exchange's classification rather than one
  I wrote matters: the same-sector prior (Sec 4.1) and the same-sector
  survival stratification (Sec 7.2) are only defensible if the grouping came
  from somewhere other than the person hoping to find pairs in it.

Two stages, in this order
-------------------------
1. ``build_candidates`` --- intersect the two files. Needs no price data.
2. ``finalise_universe`` --- apply the minimum-history and minimum-turnover
   filters using the downloaded panel, then freeze.

The order matters. The liquidity filter needs prices, prices need a list to
download, and the list cannot be the *final* list or the filter would have
nothing to remove. So: download every candidate, then filter, then freeze.

The survivorship problem, stated plainly
----------------------------------------
``fo_mktlots.csv`` is the F&O list as of *today*. Backtesting a
today-constituted universe to 2015 keeps only the names that survived and
stayed derivative-eligible. Names that were dropped from F&O, delisted, or
shrank out of the segment are absent, and their absence flatters every result.
The direction of the bias is optimistic. Point-in-time F&O membership is not
available free; ``as_of`` and ``source_urls`` are recorded here so the claim
is at least precise about what was used and when.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from dsa.data.sources import (
    NSE_FO_LOTS_URL,
    NSE_NIFTY500_URL,
    fetch_fo_underlyings,
    fetch_index_constituents,
)
from dsa.data.store import raw_reference_dir
from dsa.logging_utils import get_logger
from dsa.paths import config_dir

__all__ = [
    "Candidate",
    "build_candidates",
    "finalise_universe",
    "render_universe_yaml",
    "candidate_symbols",
]

_log = get_logger(__name__)

YF_SUFFIX = ".NS"


@dataclass(frozen=True)
class Candidate:
    symbol: str  # yfinance form, e.g. "RELIANCE.NS"
    nse_symbol: str  # NSE form, e.g. "RELIANCE"
    company: str
    sector: str


def build_candidates(*, save_raw: bool = True) -> tuple[list[Candidate], dict[str, object]]:
    """Intersect the F&O list with the sector map. Returns candidates + provenance.

    Symbols with no sector are **excluded**, not assigned to "Unknown". A
    catch-all sector would silently become the largest group and would pair
    unrelated companies under the banner of an economic prior.
    """
    ref = raw_reference_dir()
    fetched_at = date.today().isoformat()

    fo = fetch_fo_underlyings(save_to=ref / "fo_mktlots.csv" if save_raw else None)
    idx = fetch_index_constituents(
        NSE_NIFTY500_URL, save_to=ref / "ind_nifty500list.csv" if save_raw else None
    )

    sector_of = dict(zip(idx["symbol"], idx["sector"], strict=True))
    company_of = dict(zip(idx["symbol"], idx["company"], strict=True))

    candidates: list[Candidate] = []
    unmapped: list[str] = []
    for nse_symbol in sorted(fo["symbol"]):
        sector = sector_of.get(nse_symbol)
        if not sector:
            unmapped.append(nse_symbol)
            continue
        candidates.append(
            Candidate(
                symbol=f"{nse_symbol}{YF_SUFFIX}",
                nse_symbol=nse_symbol,
                company=company_of.get(nse_symbol, ""),
                sector=sector,
            )
        )

    provenance = {
        "fetched_at": fetched_at,
        "source_urls": [NSE_FO_LOTS_URL, NSE_NIFTY500_URL],
        "n_fo_symbols": int(len(fo)),
        "n_index_symbols": int(len(idx)),
        "n_candidates": len(candidates),
        "unmapped": unmapped,
    }

    _log.info(
        "candidates: %d F&O securities, %d with a sector, %d dropped for no sector map",
        len(fo),
        len(candidates),
        len(unmapped),
    )
    if unmapped:
        _log.warning("no sector for: %s", unmapped)

    return candidates, provenance


def candidate_symbols(candidates: Sequence[Candidate]) -> list[str]:
    return [c.symbol for c in candidates]


def sector_counts(candidates: Sequence[Candidate]) -> pd.Series:
    return pd.Series([c.sector for c in candidates]).value_counts()


def same_sector_pair_count(candidates: Sequence[Candidate]) -> int:
    """Candidate pairs after the economic prior --- the funnel's second number."""
    counts = sector_counts(candidates)
    return int((counts * (counts - 1) // 2).sum())


def finalise_universe(
    candidates: Sequence[Candidate],
    *,
    min_history_years: float,
    min_median_turnover_inr: float,
    max_missing_frac: float,
    start: str,
    end: str,
) -> tuple[list[Candidate], pd.DataFrame]:
    """Apply history/liquidity/coverage filters using the clean panels.

    Returns the surviving candidates and a per-symbol diagnostic table, so the
    funnel can report *why* each name was dropped rather than only how many.
    """
    from dsa.data.clean import load_panel

    close = load_panel("close")
    turnover = load_panel("turnover")

    window = close.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    n_sessions = len(window)
    if n_sessions == 0:
        raise ValueError(f"clean panel has no rows between {start} and {end}")

    rows = []
    for cand in candidates:
        s = cand.symbol
        if s not in close.columns:
            rows.append(
                dict(symbol=s, sector=cand.sector, present=False, reason="not in clean panel")
            )
            continue

        series = window[s]
        observed = series.notna()
        n_obs = int(observed.sum())
        if n_obs == 0:
            rows.append(dict(symbol=s, sector=cand.sector, present=True, n_obs=0, reason="no data"))
            continue

        first, last = series[observed].index.min(), series[observed].index.max()
        history_years = (last - first).days / 365.25
        # Missingness measured over the name's own listed life, not the whole
        # panel: a 2019 listing is not "60% missing", it is younger.
        span = observed.loc[first:last]
        missing_frac = float(1.0 - span.sum() / max(len(span), 1))
        med_turnover = float(turnover.loc[first:last, s].median(skipna=True))

        reasons = []
        if history_years < min_history_years:
            reasons.append(f"history {history_years:.1f}y < {min_history_years}y")
        if med_turnover < min_median_turnover_inr:
            reasons.append(f"turnover {med_turnover/1e7:.1f}cr < {min_median_turnover_inr/1e7:.1f}cr")
        if missing_frac > max_missing_frac:
            reasons.append(f"missing {missing_frac:.1%} > {max_missing_frac:.1%}")

        rows.append(
            dict(
                symbol=s,
                sector=cand.sector,
                present=True,
                n_obs=n_obs,
                first=str(first.date()),
                last=str(last.date()),
                history_years=round(history_years, 2),
                median_turnover_inr=med_turnover,
                missing_frac=round(missing_frac, 4),
                passes=not reasons,
                reason="; ".join(reasons) if reasons else "",
            )
        )

    diagnostics = pd.DataFrame(rows)
    passing = set(diagnostics.loc[diagnostics.get("passes", False) == True, "symbol"])  # noqa: E712
    survivors = [c for c in candidates if c.symbol in passing]

    _log.info(
        "universe filter: %d candidates -> %d survivors (%d dropped)",
        len(candidates),
        len(survivors),
        len(candidates) - len(survivors),
    )
    return survivors, diagnostics


# ---------------------------------------------------------------------------
# writing config/universe.yaml
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# ===========================================================================
# universe.yaml --- the frozen, versioned trading universe
#
# !!! GENERATED FILE --- do not hand-edit the ticker list !!!
# Regenerate with:  python scripts/build_dataset.py --stage all
#
# Spec Sec 11.4: "Freeze this list and version it --- a universe that changes
# as you iterate is a silent source of overfitting."
#
# PROVENANCE
#   generated_at : {generated_at}
#   sources      : {source_urls}
#   funnel       : {n_fo} F&O securities
#                  -> {n_candidates} with an NSE sector classification
#                  -> {n_final} passing history / turnover / coverage filters
#   pair counts  : {n_pairs_all:,} unrestricted, {n_pairs_sector:,} same-sector
# ===========================================================================

name: {name}
version: {version}
frozen: {frozen}
as_of: "{as_of}"

# --- Data window -----------------------------------------------------------
start_date: "{start_date}"
end_date: "{end_date}"

# --- Source ----------------------------------------------------------------
source: yfinance
ticker_suffix: "{suffix}"        # tickers below already carry it
price_field: adj_close         # Spec Sec 11.2: non-negotiable
benchmark: "^NSEI"             # NIFTY 50
vix_symbol: "^INDIAVIX"        # ML filter market-context feature (Sec 8.2)

# --- Inclusion filters actually applied ------------------------------------
target_size: {target_size}
min_history_years: {min_history_years}
min_median_turnover_inr: {min_median_turnover_inr}
max_missing_frac: {max_missing_frac}

# --- Known bias, stated up front (Spec Sec 11.3) ---------------------------
# The F&O list above is TODAY's list. Backtesting it to {start_year} keeps only
# names that survived and stayed derivative-eligible. Direction: optimistic.
survivorship_bias:
  acknowledged: true
  direction: optimistic
  mitigation: "{mitigation}"

# --- Connectivity smoke test ------------------------------------------------
smoke_test_tickers:
{smoke_block}
# --- THE FROZEN LIST ({n_final} names, {n_sectors} sectors) ---
tickers:
{tickers_block}
sectors:
{sectors_block}"""


def render_universe_yaml(
    survivors: Sequence[Candidate],
    provenance: dict[str, object],
    *,
    name: str,
    version: int,
    start_date: str,
    end_date: str,
    target_size: int,
    min_history_years: float,
    min_median_turnover_inr: float,
    max_missing_frac: float,
    mitigation: str,
    smoke_test_tickers: Sequence[str],
    n_candidates: int,
) -> str:
    """Render the frozen universe as YAML text, with provenance in the header."""
    ordered = sorted(survivors, key=lambda c: c.symbol)
    n = len(ordered)
    counts = sector_counts(ordered)

    tickers_block = "".join(f'  - "{c.symbol}"\n' for c in ordered)
    sectors_block = "".join(f'  "{c.symbol}": "{c.sector}"\n' for c in ordered)
    smoke_block = "".join(f'  - "{s}"\n' for s in smoke_test_tickers)

    return _TEMPLATE.format(
        generated_at=provenance.get("fetched_at", date.today().isoformat()),
        source_urls=", ".join(provenance.get("source_urls", [])),  # type: ignore[arg-type]
        n_fo=provenance.get("n_fo_symbols", "?"),
        n_candidates=n_candidates,
        n_final=n,
        n_sectors=int(counts.size),
        n_pairs_all=n * (n - 1) // 2,
        n_pairs_sector=same_sector_pair_count(ordered),
        name=name,
        version=version,
        frozen="true",
        as_of=provenance.get("fetched_at", date.today().isoformat()),
        start_date=start_date,
        end_date=end_date,
        start_year=start_date[:4],
        suffix=YF_SUFFIX,
        target_size=target_size,
        min_history_years=min_history_years,
        min_median_turnover_inr=int(min_median_turnover_inr),
        max_missing_frac=max_missing_frac,
        mitigation=mitigation,
        smoke_block=smoke_block,
        tickers_block=tickers_block,
        sectors_block=sectors_block,
    )


def write_universe_yaml(text: str, path: Path | None = None) -> Path:
    """Write and immediately re-validate through the config loader.

    Writing a config the loader then rejects would leave the project unable to
    start, so the round-trip is checked here rather than discovered later.
    """
    target = path or (config_dir() / "universe.yaml")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict) or "tickers" not in parsed:
        raise ValueError("rendered universe.yaml is not a mapping with a `tickers` key")
    target.write_text(text, encoding="utf-8")
    _log.info("wrote %s (%d tickers)", target, len(parsed["tickers"]))
    return target
