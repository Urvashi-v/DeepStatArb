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

import pandas as pd

from dsa.data.sources import (
    NSE_FO_LOTS_URL,
    NSE_NIFTY500_URL,
    fetch_fo_underlyings,
    fetch_index_constituents,
)
from dsa.data.store import raw_reference_dir
from dsa.logging_utils import get_logger

__all__ = [
    "Candidate",
    "build_candidates",
    "candidate_symbols",
    "sector_counts",
    "same_sector_pair_count",
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


def finalise_universe(*args, **kwargs):  # pragma: no cover - removed on purpose
    """REMOVED. This computed liquidity over the FULL sample and then declared a
    name eligible for that whole sample --- so a stock that was thin until 2022
    passed on its full-sample median and would have been traded in 2015.

    Replaced by :mod:`dsa.universe.selection`, which evaluates liquidity and
    history from a trailing window ending at each decision date.

    It raises rather than being deleted silently, so any surviving call site
    fails loudly instead of quietly reintroducing the leak.
    """
    raise NotImplementedError(
        "finalise_universe used full-sample liquidity to decide membership for the whole "
        "sample, which is lookahead. Use dsa.universe.select_universe, which evaluates "
        "eligibility point-in-time at each formation date. See scripts/build_universe.py."
    )


# ---------------------------------------------------------------------------
# NOTE: universe freezing, versioning, hashing and YAML rendering now live in
# dsa.universe.freeze. They were moved out of the data layer because they are
# research decisions (which names, on what criteria, identified how) rather
# than data-acquisition concerns.
# ---------------------------------------------------------------------------
