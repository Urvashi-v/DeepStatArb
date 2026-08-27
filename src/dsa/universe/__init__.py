"""The research universe --- spec Sec 11.4.

Two levels, deliberately separated:

* **The frozen research universe** --- the names ever considered. Versioned and
  hashed, so an experiment is reproducible and a silent change of membership
  between runs is impossible to miss.
* **Point-in-time eligibility** --- who qualifies at a given formation date,
  computed from a trailing window ending at that date. This is what the
  walk-forward loop consults, and it is what keeps universe construction free
  of lookahead.

Freezing one list for the whole sample would leak: liquidity and listing
history measured over 2015-2026 admit names that only *became* tradeable
later. Recomputing eligibility per window without freezing anything would make
results irreproducible. Doing both is the point.

``dsa.data.universe`` builds the *candidate pool* from NSE's published files;
this package decides which of those candidates make the cut, and when.
"""

from dsa.universe.freeze import (
    FrozenUniverse,
    next_version,
    read_frozen_universe,
    render_universe_yaml,
    universe_hash,
    write_universe_yaml,
)
from dsa.universe.report import build_report, write_report
from dsa.universe.selection import (
    REASON_CODES,
    EligibilityCriteria,
    SelectionError,
    SelectionResult,
    decision_dates,
    eligibility_at,
    eligibility_schedule,
    liquidity_stats_at,
    select_universe,
)

__all__ = [
    "EligibilityCriteria",
    "FrozenUniverse",
    "REASON_CODES",
    "SelectionError",
    "SelectionResult",
    "build_report",
    "decision_dates",
    "eligibility_at",
    "eligibility_schedule",
    "liquidity_stats_at",
    "next_version",
    "read_frozen_universe",
    "render_universe_yaml",
    "select_universe",
    "universe_hash",
    "write_report",
    "write_universe_yaml",
]
