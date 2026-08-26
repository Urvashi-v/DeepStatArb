"""The leak detector.

Spec Sec 14: *"Write a test that computes every feature on data[:n] and on
data[:n+50] and asserts the first n values are identical. Any feature that
fails is reading the future. ... Build the leak detector on day one, not day
twenty-five."*

The idea in one line
--------------------
A function is **causal** if its value at time *t* depends only on data up to
time *t*. So compute it twice --- once on the full history, once on a truncated
copy --- and check the overlapping values agree. If a feature computed on
``data[:100]`` differs at position 99 from the same feature computed on
``data[:200]``, then the extra 100 future rows changed the past. That is a
leak, by definition, and no amount of reasoning about the code is needed to
establish it.

This catches, automatically and on the day it is introduced:

* full-sample ``mean()`` / ``std()`` in a z-score (bug #1),
* centred rolling windows,
* a hedge ratio fitted on the whole series and applied backwards,
* ``shift(-1)``, ``bfill()``, ``interpolate(method='time')`` over gaps,
* a scaler fitted globally then applied to the training window (bug #6),
* a parameter refitted inside the trading window (bug #8).

What it deliberately does **not** catch
---------------------------------------
**Same-bar execution** (bug #4). ``spread.rolling(20).mean()`` is perfectly
causal --- at time *t* it reads *t-19 ... t*, all of which are in the past or
present. It is still wrong to trade on it at the close of *t*, because you only
know that close once *t* has finished. That is an *execution alignment*
problem, not a future-data problem, and it is enforced separately by
``backtest.execution.lag_bars >= 1`` and by the engine's own tests. Two
different bugs, two different guards; conflating them means one of them goes
unchecked.

What it can only catch *probabilistically*
------------------------------------------
The check compares the full and truncated results at a handful of truncation
points. A **pervasive** leak --- ``shift(-1)``, a full-sample mean, a centred
window --- changes values everywhere, so any checkpoint catches it with
certainty. A **sparse** leak does not. ``bfill()`` across an occasional
one-day hole only corrupts the rows adjacent to a hole, so it is visible only
when a truncation boundary happens to land on one.

The mitigation is cheap and partial: use several checkpoints (the default is
seven, spread across the sample), and when a specific sparse mechanism is
suspected, pass ``checkpoints=`` targeting the rows where it would bite. What
this tool gives you is a strong guarantee against pervasive leakage and a
useful-but-not-certain screen against sparse leakage. Claiming more than that
would be the same overconfidence the tool exists to prevent.

A note on floating point
------------------------
The comparison is tolerance-based, not exact, and that is not laziness.
pandas' rolling reductions use an online add/remove accumulator, so the running
sum at position *t* can carry a different rounding history depending on how
many elements have streamed through it. A causal ``rolling(20).mean()`` can
therefore differ in the last couple of bits between the truncated and full
runs. An exact comparison reports that as a leak, the test gets muted as
flaky, and a real leak walks through later. The default tolerances (``rtol``
1e-9, ``atol`` 1e-12) are many orders of magnitude below any leak that could
change a trading decision, and many orders above accumulator noise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "LeakageError",
    "LeakageFailure",
    "LeakageReport",
    "check_causal",
    "assert_causal",
    "default_checkpoints",
]

FeatureFn = Callable[[Any], "pd.Series | pd.DataFrame"]

# Fractions of the sample at which to truncate. Spread across the second half
# so that every checkpoint has enough history for realistic warm-up windows.
# Seven rather than three or four: each is one extra evaluation of a cheap
# function, and more boundaries means a better chance of landing on a sparse
# leak (see the module docstring).
_DEFAULT_FRACTIONS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92)

_DEFAULT_RTOL = 1e-9
_DEFAULT_ATOL = 1e-12


class LeakageError(AssertionError):
    """Raised by :func:`assert_causal` when a function reads the future.

    Inherits from ``AssertionError`` so pytest reports it as a failed
    assertion rather than an error in the test itself.
    """


@dataclass(frozen=True)
class LeakageFailure:
    """One concrete disagreement between the full and truncated computations."""

    checkpoint: int
    column: str | None
    index_label: Any
    position: int
    full_value: float
    truncated_value: float
    abs_diff: float

    def __str__(self) -> str:
        col = f"[{self.column}]" if self.column is not None else ""
        return (
            f"truncating at n={self.checkpoint} changed the value{col} at "
            f"position {self.position} (index {self.index_label!r}): "
            f"full={self.full_value!r} vs truncated={self.truncated_value!r} "
            f"(|diff|={self.abs_diff:.6g})"
        )


@dataclass(frozen=True)
class LeakageReport:
    """Outcome of a causality check."""

    name: str
    passed: bool
    n_obs: int
    checkpoints: tuple[int, ...]
    failures: tuple[LeakageFailure, ...]
    n_compared: int

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        if self.passed:
            return (
                f"CAUSAL: '{self.name}' is unchanged by future data "
                f"({self.n_compared} values compared across checkpoints "
                f"{list(self.checkpoints)} of {self.n_obs} observations)"
            )
        head = (
            f"LOOKAHEAD DETECTED in '{self.name}': {len(self.failures)} "
            f"disagreement(s) across checkpoints {list(self.checkpoints)}."
        )
        # Report the earliest failures; they are the most diagnostic.
        shown = sorted(self.failures, key=lambda f: (f.checkpoint, f.position))[:5]
        lines = [f"  - {f}" for f in shown]
        if len(self.failures) > len(shown):
            lines.append(f"  ... and {len(self.failures) - len(shown)} more")
        return "\n".join([head, *lines])


def default_checkpoints(n_obs: int, min_history: int = 30) -> tuple[int, ...]:
    """Truncation points for a sample of ``n_obs`` observations."""
    if n_obs <= min_history:
        raise ValueError(
            f"need more than min_history={min_history} observations to check causality, "
            f"got {n_obs}"
        )
    raw = {int(round(f * n_obs)) for f in _DEFAULT_FRACTIONS}
    valid = sorted(c for c in raw if min_history <= c < n_obs)
    if not valid:
        # Very short samples: fall back to a single midpoint.
        mid = max(min_history, n_obs // 2)
        valid = [mid] if mid < n_obs else [n_obs - 1]
    return tuple(valid)


def _as_frame(obj: Any, where: str) -> pd.DataFrame:
    """Normalise a feature function's output to a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name if obj.name is not None else "value")
    raise TypeError(
        f"{where}: the feature function must return a pandas Series or DataFrame indexed "
        f"like its input, got {type(obj).__name__}. A function that returns a bare array "
        "cannot be checked, because there is no index to align the truncated and full "
        "results on."
    )


def _truncate(data: Any, n: int) -> Any:
    """Take the first ``n`` rows of a Series, DataFrame, or dict of them."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[:n]
    if isinstance(data, dict):
        return {k: _truncate(v, n) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_truncate(v, n) for v in data)
    raise TypeError(
        f"cannot truncate an object of type {type(data).__name__}; pass a pandas "
        "Series/DataFrame, or a dict/tuple of them"
    )


def _n_obs(data: Any) -> int:
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return len(data)
    if isinstance(data, dict):
        lengths = {len(v) for v in data.values()}
        if len(lengths) != 1:
            raise ValueError(f"inputs have differing lengths {sorted(lengths)}; align them first")
        return lengths.pop()
    if isinstance(data, (list, tuple)):
        lengths = {len(v) for v in data}
        if len(lengths) != 1:
            raise ValueError(f"inputs have differing lengths {sorted(lengths)}; align them first")
        return lengths.pop()
    raise TypeError(f"unsupported input type {type(data).__name__}")


def _index_of(data: Any) -> pd.Index | None:
    """The index of the INPUT, used to detect invented output labels.

    Returns None when the input has no single well-defined index (in which
    case the invented-label check is skipped rather than guessed at).
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.index
    if isinstance(data, dict):
        values = list(data.values())
    elif isinstance(data, (list, tuple)):
        values = list(data)
    else:
        return None
    indexes = [v.index for v in values if isinstance(v, (pd.Series, pd.DataFrame))]
    if not indexes:
        return None
    first = indexes[0]
    return first if all(ix.equals(first) for ix in indexes[1:]) else None


def _values_equal(a: Any, b: Any, rtol: float, atol: float) -> bool:
    """Compare two scalars, treating NaN as equal to NaN."""
    a_null = pd.isna(a)
    b_null = pd.isna(b)
    if a_null or b_null:
        return bool(a_null and b_null)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return bool(a == b)


def check_causal(
    fn: FeatureFn,
    data: Any,
    *,
    name: str | None = None,
    checkpoints: Sequence[int] | None = None,
    min_history: int = 30,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> LeakageReport:
    """Check that ``fn`` never uses information from beyond the current row.

    Parameters
    ----------
    fn
        Takes the data and returns a Series or DataFrame indexed by (a subset
        of) the input index.
    data
        A Series, DataFrame, or a dict/tuple of equally-indexed ones.
    checkpoints
        Truncation lengths. Defaults to :func:`default_checkpoints`.
    min_history
        Truncations shorter than this are skipped; below it, warm-up windows
        make the comparison uninformative rather than wrong.
    rtol, atol
        Tolerances --- see the module docstring on floating point.

    Returns
    -------
    LeakageReport
        Falsy when a leak was found. Use :func:`assert_causal` in tests.
    """
    label = name or getattr(fn, "__name__", repr(fn))
    n = _n_obs(data)
    points = tuple(sorted(set(checkpoints))) if checkpoints else default_checkpoints(n, min_history)

    for c in points:
        if not 0 < c <= n:
            raise ValueError(f"checkpoint {c} is outside the sample of {n} observations")

    full = _as_frame(fn(data), f"{label}(full sample)")
    input_index = _index_of(data)

    failures: list[LeakageFailure] = []
    n_compared = 0

    for c in points:
        part = _as_frame(fn(_truncate(data, c)), f"{label}(data[:{c}])")

        if set(part.columns) != set(full.columns):
            raise AssertionError(
                f"{label}: truncating at n={c} changed the OUTPUT COLUMNS from "
                f"{sorted(map(str, full.columns))} to {sorted(map(str, part.columns))}. "
                "A feature set that depends on how much data it is given cannot be "
                "evaluated for causality --- fix the function first."
            )

        # An output label that is not among the first `c` INPUT labels means the
        # function invented a timestamp (resampling, reindexing, asfreq). Such
        # output cannot be aligned to the input clock, so causality is not even
        # well defined for it.
        if input_index is not None:
            stray = part.index.difference(input_index[:c])
            if len(stray) > 0:
                raise AssertionError(
                    f"{label}: computing on data[:{c}] produced index labels not present in "
                    f"the first {c} rows of the input, e.g. {list(stray[:3])}. The function "
                    "resamples or reindexes in a way that invents timestamps; causality "
                    "cannot be checked against an index the input never had."
                )

        common = part.index.intersection(full.index)
        if len(common) == 0:
            continue

        for col in full.columns:
            fv = full.loc[common, col]
            pv = part.loc[common, col]
            n_compared += len(common)

            # Fast path: identical arrays, no per-element work.
            try:
                if fv.equals(pv):
                    continue
            except (TypeError, ValueError):
                pass

            for label_idx, (a, b) in zip(
                common, zip(fv.to_numpy(), pv.to_numpy(), strict=True), strict=True
            ):
                if not _values_equal(a, b, rtol, atol):
                    try:
                        diff = abs(float(a) - float(b))
                    except (TypeError, ValueError):
                        diff = float("nan")
                    failures.append(
                        LeakageFailure(
                            checkpoint=c,
                            column=None if len(full.columns) == 1 else str(col),
                            index_label=label_idx,
                            position=int(full.index.get_loc(label_idx)),
                            full_value=a,
                            truncated_value=b,
                            abs_diff=diff,
                        )
                    )

    return LeakageReport(
        name=label,
        passed=not failures,
        n_obs=n,
        checkpoints=points,
        failures=tuple(failures),
        n_compared=n_compared,
    )


def assert_causal(fn: FeatureFn, data: Any, **kwargs: Any) -> LeakageReport:
    """:func:`check_causal`, raising :class:`LeakageError` on failure."""
    report = check_causal(fn, data, **kwargs)
    if not report.passed:
        raise LeakageError(str(report))
    return report
