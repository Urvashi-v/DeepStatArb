"""Tests for the leak detector itself --- spec Sec 14.

This is the most important test file in the repository, and today it is
pointed at the detector rather than at features, because there are no features
yet. The detector has to be trustworthy *before* it is trusted.

Trustworthy means two things, and both are tested here:

* **No false negatives.** Every known leakage pattern in the spec's bug list is
  planted deliberately and must be caught. A detector that misses a leak is
  worse than none, because it grants false confidence.
* **No false positives.** Legitimately causal features --- trailing rolling
  windows, expanding windows, the exact ``.rolling(w).mean().shift(1)`` idiom
  the signal path will use --- must pass cleanly. A detector that cries wolf
  gets muted, and then it catches nothing.

SYNTHETIC DATA --- METHODOLOGY/UNIT TEST ONLY. All series here come from the
fixtures in conftest.py and are generated from known processes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsa.validity.leakage import (
    LeakageError,
    assert_causal,
    check_causal,
    default_checkpoints,
)

pytestmark = [pytest.mark.leakage, pytest.mark.synthetic]


# ===========================================================================
# CAUSAL functions --- these must PASS (no false positives)
# ===========================================================================


def trailing_mean(s: pd.Series) -> pd.Series:
    """Rolling mean over a window ending at t. Causal."""
    return s.rolling(20, min_periods=20).mean()


def trailing_mean_shifted(s: pd.Series) -> pd.Series:
    """The signal-path idiom: window ends at t-1. Causal, and the one we use."""
    return s.rolling(20, min_periods=20).mean().shift(1)


def trailing_zscore(s: pd.Series) -> pd.Series:
    """The CORRECT z-score of spec Sec 2.5.

    mu and sigma come from a trailing window ending at t-1, made explicit by
    the ``.shift(1)`` rather than left to the reader's trust in intentions.
    """
    mu = s.rolling(60, min_periods=60).mean().shift(1)
    sd = s.rolling(60, min_periods=60).std().shift(1)
    return (s - mu) / sd


def expanding_mean(s: pd.Series) -> pd.Series:
    """Expanding (all history to date) mean. Causal."""
    return s.expanding(min_periods=30).mean()


def lagged_return(s: pd.Series) -> pd.Series:
    """A past return. Causal."""
    return s.pct_change(5)


def multi_feature_frame(s: pd.Series) -> pd.DataFrame:
    """Several causal features at once, to exercise the DataFrame path."""
    return pd.DataFrame(
        {
            "z": trailing_zscore(s),
            "vol20": s.rolling(20, min_periods=20).std().shift(1),
            "sign": np.sign(s),
        }
    )


@pytest.mark.parametrize(
    "fn",
    [
        trailing_mean,
        trailing_mean_shifted,
        trailing_zscore,
        expanding_mean,
        lagged_return,
        multi_feature_frame,
    ],
    ids=lambda f: f.__name__,
)
def test_causal_functions_pass(fn, ou_spread):
    """Legitimately causal features must not be flagged."""
    report = check_causal(fn, ou_spread)
    assert report.passed, str(report)
    assert report.n_compared > 0, "the check compared nothing, so it proved nothing"


def test_causal_check_is_not_defeated_by_float_noise(ou_spread):
    """Rolling accumulators can differ in the last bits; that is not a leak.

    See the module docstring in dsa.validity.leakage on why the comparison is
    tolerance-based. This test exists so that a future tightening of the
    tolerances to exact equality shows up here rather than as flakiness.
    """
    report = check_causal(trailing_mean, ou_spread, rtol=1e-9, atol=1e-12)
    assert report.passed, str(report)


# ===========================================================================
# LEAKY functions --- these must be CAUGHT (no false negatives)
# ===========================================================================


def full_sample_zscore(s: pd.Series) -> pd.Series:
    """BUG #1, spec Sec 2.5 --- the single most common bug in pairs trading.

    ``s.mean()`` and ``s.std()`` are computed over the entire sample, so the
    normalisation of 2016 uses information from 2024.
    """
    return (s - s.mean()) / s.std()


def centered_rolling(s: pd.Series) -> pd.Series:
    """A centred window straddles t and reads w/2 bars into the future."""
    return s.rolling(21, center=True, min_periods=21).mean()


def next_bar_value(s: pd.Series) -> pd.Series:
    """``shift(-1)`` --- tomorrow's value presented as today's feature."""
    return s.shift(-1)


def global_minmax_scaled(s: pd.Series) -> pd.Series:
    """A scaler fitted on the whole series --- bug #6 in miniature."""
    return (s - s.min()) / (s.max() - s.min())


def full_sample_hedge_ratio(s: pd.Series) -> pd.Series:
    """The project-specific version: beta fitted on the full sample.

    A hedge ratio estimated over all available data and then applied to the
    start of the series means the 2016 spread was built with a beta that could
    only have been known in 2024. This is how a cointegration screen leaks
    even when the z-score is computed correctly.
    """
    x = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    beta = float(np.cov(s.to_numpy(), x.to_numpy())[0, 1] / np.var(x.to_numpy()))
    return s - beta * x


def future_max_normalised(s: pd.Series) -> pd.Series:
    """Reverse-cumulative maximum: every point knows the whole future."""
    return s / s[::-1].cummax()[::-1]


@pytest.mark.parametrize(
    "fn",
    [
        full_sample_zscore,
        centered_rolling,
        next_bar_value,
        global_minmax_scaled,
        full_sample_hedge_ratio,
        future_max_normalised,
    ],
    ids=lambda f: f.__name__,
)
def test_leaky_functions_are_caught(fn, ou_spread):
    """Every planted leak must be detected."""
    report = check_causal(fn, ou_spread)
    assert not report.passed, (
        f"{fn.__name__} reads the future but the detector passed it. "
        "A detector with a false negative is worse than no detector."
    )
    assert report.failures, "check failed but reported no specific disagreement"


def _backfilled(s: pd.Series, hole_dates: pd.Index) -> pd.Series:
    """``bfill`` pulls a later observation backwards across a gap.

    Holes are addressed by DATE, not position, so that truncating the input
    truncates the holes with it --- which is how a real feed's missing bars
    behave. This is a SPARSE leak: only the rows immediately before a hole are
    corrupted, so it is invisible unless a truncation boundary lands on one.
    """
    holed = s.copy()
    holed.loc[holed.index.intersection(hole_dates)] = np.nan
    return holed.bfill()


def test_sparse_bfill_leak_is_caught_by_targeted_checkpoints(ou_spread):
    """bfill over a gap is a leak, and targeted checkpoints find it."""
    positions = [399, 549, 699]
    hole_dates = ou_spread.index[positions]

    def fn(s: pd.Series) -> pd.Series:
        return _backfilled(s, hole_dates)

    # Truncating at 400 leaves position 399 as the last row, with no later
    # value to pull from, so it stays NaN; on the full sample it borrows the
    # value at position 400. That disagreement IS the leak.
    report = check_causal(fn, ou_spread, checkpoints=[400, 550, 700])
    assert not report.passed, str(report)
    assert {f.position for f in report.failures} == set(positions)


def test_sparse_leak_can_evade_the_default_checkpoints(ou_spread):
    """A documented limitation, pinned by a test so it cannot be forgotten.

    The same bfill leak, with holes placed away from the default truncation
    boundaries, is NOT detected. This is not a bug in the detector; it is the
    boundary of what a truncation-based check can promise (see the module
    docstring). It is asserted here so that the limitation is a known, tested
    property rather than a surprise found during an interview.
    """
    hole_dates = ou_spread.index[[123, 456]]

    def fn(s: pd.Series) -> pd.Series:
        return _backfilled(s, hole_dates)

    assert check_causal(fn, ou_spread).passed, (
        "the default checkpoints unexpectedly caught this sparse leak. That is "
        "good news, but it means this test no longer documents what it claims "
        "- re-derive the limitation before deleting it."
    )


def test_leak_in_one_column_of_many_is_caught(ou_spread):
    """A single leaky column must not be masked by causal neighbours."""

    def mostly_causal(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "good_z": trailing_zscore(s),
                "good_vol": s.rolling(20, min_periods=20).std().shift(1),
                "bad_z": full_sample_zscore(s),
            }
        )

    report = check_causal(mostly_causal, ou_spread)
    assert not report.passed
    leaking_columns = {f.column for f in report.failures}
    assert leaking_columns == {"bad_z"}, (
        f"expected only 'bad_z' to be flagged, got {leaking_columns}"
    )


def test_assert_causal_raises_with_a_useful_message(ou_spread):
    """The failure message must name the function and show a concrete value."""
    with pytest.raises(LeakageError) as exc:
        assert_causal(full_sample_zscore, ou_spread)
    message = str(exc.value)
    assert "LOOKAHEAD DETECTED" in message
    assert "full_sample_zscore" in message
    assert "full=" in message and "truncated=" in message


def test_assert_causal_returns_report_on_success(ou_spread):
    report = assert_causal(trailing_zscore, ou_spread)
    assert report.passed
    assert bool(report) is True


# ===========================================================================
# The detector's own edge cases
# ===========================================================================


def test_all_nan_warmup_is_not_a_false_positive(ou_spread):
    """NaN must compare equal to NaN, or every warm-up period looks like a leak."""

    def long_warmup(s: pd.Series) -> pd.Series:
        return s.rolling(200, min_periods=200).mean().shift(1)

    report = check_causal(long_warmup, ou_spread, checkpoints=[300, 500])
    assert report.passed, str(report)


def test_dict_of_frames_input(cointegrated_pair):
    """Pair features take two aligned series; the detector must handle that."""

    def pair_spread(d: dict[str, pd.Series]) -> pd.Series:
        # A trailing-window hedge ratio: causal.
        cov = d["a"].rolling(60, min_periods=60).cov(d["b"]).shift(1)
        var = d["b"].rolling(60, min_periods=60).var().shift(1)
        return d["a"] - (cov / var) * d["b"]

    data = {"a": cointegrated_pair["A"], "b": cointegrated_pair["B"]}
    assert check_causal(pair_spread, data).passed


def test_dataframe_input(cointegrated_pair):
    def spread_from_frame(df: pd.DataFrame) -> pd.Series:
        return df["A"] - 1.5 * df["B"]

    assert check_causal(spread_from_frame, cointegrated_pair).passed


def test_leaky_pair_feature_is_caught(cointegrated_pair):
    """A full-sample beta on a real pair shape --- the case that matters most."""

    def static_full_sample_beta(df: pd.DataFrame) -> pd.Series:
        beta = float(
            np.cov(df["A"].to_numpy(), df["B"].to_numpy())[0, 1] / np.var(df["B"].to_numpy())
        )
        return df["A"] - beta * df["B"]

    report = check_causal(static_full_sample_beta, cointegrated_pair)
    assert not report.passed, (
        "a hedge ratio fitted on the full sample was not flagged; this is the "
        "leak that makes a pairs backtest look excellent and be worthless"
    )


def test_non_pandas_return_is_rejected(ou_spread):
    with pytest.raises(TypeError, match="pandas Series or DataFrame"):
        check_causal(lambda s: s.to_numpy(), ou_spread)


def test_changing_column_set_is_rejected(ou_spread):
    """A feature set that depends on sample size cannot be checked at all."""

    def unstable(s: pd.Series) -> pd.DataFrame:
        out = {"a": s}
        if len(s) > 700:
            out["b"] = s * 2
        return pd.DataFrame(out)

    with pytest.raises(AssertionError, match="OUTPUT COLUMNS"):
        check_causal(unstable, ou_spread, checkpoints=[500, 900])


def test_invented_index_labels_are_rejected(ou_spread):
    """Resampling that invents timestamps breaks the alignment contract."""

    def resampler(s: pd.Series) -> pd.Series:
        return s.resample("W").mean()

    with pytest.raises(AssertionError, match="index labels not present"):
        check_causal(resampler, ou_spread, checkpoints=[400])


def test_explicit_checkpoints_are_honoured(ou_spread):
    report = check_causal(trailing_mean, ou_spread, checkpoints=[100, 200, 300])
    assert report.checkpoints == (100, 200, 300)


def test_checkpoint_outside_sample_is_rejected(ou_spread):
    with pytest.raises(ValueError, match="outside the sample"):
        check_causal(trailing_mean, ou_spread, checkpoints=[len(ou_spread) + 1])


def test_default_checkpoints_are_inside_the_sample():
    for n in (100, 250, 1000, 2500):
        points = default_checkpoints(n)
        assert points, f"no checkpoints generated for n={n}"
        assert all(30 <= c < n for c in points), points
        assert len(points) >= 2, f"n={n} gave only {points}; one checkpoint is a weak check"


def test_too_short_a_sample_is_rejected():
    with pytest.raises(ValueError, match="min_history"):
        default_checkpoints(20, min_history=30)


def test_report_string_is_readable(ou_spread):
    passing = check_causal(trailing_zscore, ou_spread)
    assert "CAUSAL" in str(passing)
    assert "values compared" in str(passing)

    failing = check_causal(full_sample_zscore, ou_spread)
    assert "LOOKAHEAD DETECTED" in str(failing)
