"""Freezing, versioning and hashing the research universe --- spec Sec 11.4.

*"Freeze this list and version it --- a universe that changes as you iterate is
a silent source of overfitting."*

What the hash is for
--------------------
The universe hash is a digest of the **sorted ticker list and its sector map**,
and nothing else. Two runs with the same universe hash looked at the same
names; two with different hashes did not, whatever their configuration files
happened to say. That matters because the most damaging kind of iteration is
the invisible kind --- a name quietly added or dropped between runs, so that a
Sharpe improvement looks like a modelling win when it was a universe change.

The hash deliberately excludes ``version``, ``as_of``, the provenance block and
the criteria. Those describe *how* the list came to be; the hash answers only
*which names*. Re-running the builder on an unchanged panel with unchanged
rules must reproduce the same hash, which is what makes it usable as an
identity.

Versioning
----------
``version`` increments only when the hash changes. Re-freezing an identical
universe is a no-op, so the version number counts genuine changes rather than
how many times the script was run. A change to the criteria that does not
change the resulting list does not bump the version --- it is recorded in the
criteria block, where it belongs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from dsa.logging_utils import get_logger
from dsa.paths import config_dir

__all__ = [
    "universe_hash",
    "FrozenUniverse",
    "render_universe_yaml",
    "write_universe_yaml",
    "read_frozen_universe",
]

_log = get_logger(__name__)


def universe_hash(symbols: Sequence[str], sectors: Mapping[str, str] | None = None) -> str:
    """Stable 12-character digest of the universe's *identity*.

    Sorted, so insertion order cannot change it. Includes sectors, because a
    name reclassified from Banks to Financial Services changes which pairs the
    economic prior allows, and that is a different universe even though the
    ticker list is identical.
    """
    payload: dict[str, Any] = {"tickers": sorted(symbols)}
    if sectors is not None:
        payload["sectors"] = {k: sectors[k] for k in sorted(sectors)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class FrozenUniverse:
    """A universe as persisted: the list, its identity, and its provenance."""

    name: str
    version: int
    as_of: str
    symbols: tuple[str, ...]
    sectors: Mapping[str, str]
    hash: str
    criteria: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def n_pairs(self) -> int:
        n = len(self.symbols)
        return n * (n - 1) // 2

    def sector_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for symbol in self.symbols:
            out[self.sectors[symbol]] = out.get(self.sectors[symbol], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def same_sector_pairs(self) -> int:
        return sum(n * (n - 1) // 2 for n in self.sector_counts().values())


_TEMPLATE = """\
# ===========================================================================
# universe.yaml --- the FROZEN research universe
#
# !!! GENERATED FILE --- do not hand-edit !!!
# Regenerate with:  python scripts/build_universe.py
#
# This file is OUTPUT. The rules that produced it are hand-written in
# config/selection.yaml; a copy of them is recorded below so the list and the
# criteria that chose it always travel together.
#
# Spec Sec 11.4: "Freeze this list and version it --- a universe that changes
# as you iterate is a silent source of overfitting."
#
# IDENTITY
#   universe hash : {universe_hash}
#   version       : {version}   (increments only when the hash changes)
#   generated     : {generated_at}
#
# HOW THIS LIST WAS BUILT
#   sources   : {source_urls}
#   funnel    : {n_candidates} candidates with price data
#               -> {n_selected} in the frozen universe
#   pairs     : {n_pairs:,} unrestricted, {n_pairs_sector:,} same-sector
#   decisions : eligibility evaluated at {n_decision_dates} formation dates
#               ({first_decision} .. {last_decision})
#
# LOOKAHEAD
#   Membership in THIS list is frozen for reproducibility. It is NOT a claim
#   that every name was tradeable throughout the sample. Liquidity and history
#   are re-evaluated at each formation date using only data available then;
#   see reports/universe/eligibility.csv and dsa.universe.selection.
# ===========================================================================

name: {name}
version: {version}
frozen: true
as_of: "{as_of}"

# --- Data window -----------------------------------------------------------
start_date: "{start_date}"
end_date: "{end_date}"

# --- Source ----------------------------------------------------------------
source: yfinance
ticker_suffix: ".NS"           # tickers below already carry it
price_field: adj_close         # Spec Sec 11.2: non-negotiable
benchmark: "^NSEI"             # NIFTY 50
vix_symbol: "^INDIAVIX"        # ML filter market-context feature (Sec 8.2)

# --- Criteria that produced this list (copied from selection.yaml) ---------
# Recorded, not read. selection.yaml is authoritative.
target_size: {target_size}
min_history_years: {min_history_years}
min_median_turnover_inr: {min_median_turnover_inr}
max_missing_frac: {max_missing_frac}

# --- Known bias, stated up front (Spec Sec 11.3) ---------------------------
# The F&O list used is TODAY's list. Backtesting it to {start_year} keeps only
# names that survived and stayed derivative-eligible. Direction: optimistic.
# Point-in-time F&O membership is not available free.
survivorship_bias:
  acknowledged: true
  direction: optimistic
  mitigation: "{mitigation}"

# --- Connectivity smoke test ------------------------------------------------
smoke_test_tickers:
{smoke_block}
# --- THE FROZEN LIST ({n_selected} names, {n_sectors} sectors) ---
tickers:
{tickers_block}
sectors:
{sectors_block}"""


def render_universe_yaml(
    frozen: FrozenUniverse,
    *,
    start_date: str,
    end_date: str,
    smoke_test_tickers: Sequence[str],
    n_candidates: int,
    n_decision_dates: int,
    first_decision: str,
    last_decision: str,
    source_urls: Sequence[str],
    mitigation: str,
    target_size: Any,
) -> str:
    """Render the frozen universe as YAML text."""
    symbols = sorted(frozen.symbols)
    counts = frozen.sector_counts()

    return _TEMPLATE.format(
        universe_hash=frozen.hash,
        version=frozen.version,
        generated_at=frozen.provenance.get("generated_at", date.today().isoformat()),
        source_urls=", ".join(source_urls),
        n_candidates=n_candidates,
        n_selected=len(symbols),
        n_pairs=frozen.n_pairs,
        n_pairs_sector=frozen.same_sector_pairs(),
        n_decision_dates=n_decision_dates,
        first_decision=first_decision,
        last_decision=last_decision,
        name=frozen.name,
        as_of=frozen.as_of,
        start_date=start_date,
        end_date=end_date,
        start_year=start_date[:4],
        target_size="null" if target_size is None else target_size,
        min_history_years=frozen.criteria.get("min_history_years"),
        min_median_turnover_inr=int(frozen.criteria.get("min_median_turnover_inr", 0)),
        max_missing_frac=frozen.criteria.get("max_missing_frac"),
        mitigation=mitigation,
        smoke_block="".join(f'  - "{s}"\n' for s in smoke_test_tickers),
        n_sectors=len(counts),
        tickers_block="".join(f'  - "{s}"\n' for s in symbols),
        sectors_block="".join(f'  "{s}": "{frozen.sectors[s]}"\n' for s in symbols),
    )


def read_frozen_universe(path: Path | None = None) -> dict[str, Any]:
    """Read the current frozen universe file, if any."""
    target = path or (config_dir() / "universe.yaml")
    if not target.is_file():
        return {}
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def next_version(symbols: Sequence[str], sectors: Mapping[str, str], path: Path | None = None) -> int:
    """Bump the version only when the universe identity actually changed.

    Re-running the builder on an unchanged panel is a no-op. Without this the
    version number would count script invocations rather than real changes,
    and would stop being a useful thing to cite.
    """
    existing = read_frozen_universe(path)
    if not existing:
        return 1
    old_symbols = existing.get("tickers") or []
    old_sectors = existing.get("sectors") or {}
    if universe_hash(old_symbols, old_sectors) == universe_hash(symbols, sectors):
        return int(existing.get("version", 1))
    return int(existing.get("version", 0)) + 1


def write_universe_yaml(text: str, path: Path | None = None) -> Path:
    """Write and immediately re-validate. A config the loader rejects would
    leave the project unable to start, so the round-trip is checked here."""
    target = path or (config_dir() / "universe.yaml")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict) or "tickers" not in parsed:
        raise ValueError("rendered universe.yaml is not a mapping with a `tickers` key")
    missing = [t for t in parsed["tickers"] if t not in (parsed.get("sectors") or {})]
    if missing:
        raise ValueError(f"{len(missing)} ticker(s) have no sector, e.g. {missing[:5]}")
    target.write_text(text, encoding="utf-8")
    _log.info("wrote %s (%d tickers)", target, len(parsed["tickers"]))
    return target
