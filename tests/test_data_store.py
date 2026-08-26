"""Storage contract: raw is write-once, writes are atomic, the manifest is truth.

The manifest tests matter more than they look. Resumability is not a
convenience feature here --- a 208-ticker pull that has to restart from zero
every time it is interrupted will get run fewer times, get run with a shorter
history to make it quicker, and eventually get replaced by whatever happened to
be on disk. The manifest is what stops that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa.data.store import (
    Manifest,
    StoreError,
    TickerRecord,
    read_parquet,
    write_parquet,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """SYNTHETIC — unit test only. Shape matters here, not values."""
    idx = pd.bdate_range("2020-01-01", periods=50, name="Date")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(50))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": rng.integers(1e5, 1e6, 50),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# parquet io
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path: Path, frame: pd.DataFrame):
    path = write_parquet(frame, tmp_path / "x.parquet")
    back = read_parquet(path)
    # check_freq=False: parquet stores timestamps, not the inferred `freq`
    # attribute of a DatetimeIndex. Real yfinance data carries no freq anyway
    # (NSE holidays make it irregular), so nothing downstream depends on it.
    pd.testing.assert_frame_equal(frame, back, check_freq=False)


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path, frame: pd.DataFrame):
    write_parquet(frame, tmp_path / "x.parquet")
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_failed_write_leaves_no_partial_file(tmp_path: Path):
    """A half-written parquet the manifest would call 'done' is the worst case."""

    class Unwritable:
        def to_parquet(self, *a, **k):
            raise OSError("disk full")

    with pytest.raises(OSError):
        write_parquet(Unwritable(), tmp_path / "x.parquet")  # type: ignore[arg-type]
    assert not (tmp_path / "x.parquet").exists()
    assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name]


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_path / "nope.parquet")


def test_raw_is_write_once(tmp_path: Path, frame: pd.DataFrame, monkeypatch):
    """data/raw must not be silently overwritten."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr("dsa.data.store.data_raw", lambda: raw)

    target = raw / "ohlcv" / "X.parquet"
    write_parquet(frame, target)

    with pytest.raises(StoreError, match="write-once"):
        write_parquet(frame, target)

    # The explicit refresh path is still allowed.
    write_parquet(frame, target, force=True)


def test_non_raw_paths_overwrite_freely(tmp_path: Path, frame: pd.DataFrame, monkeypatch):
    """data/clean is rebuildable, so overwriting it is normal."""
    monkeypatch.setattr("dsa.data.store.data_raw", lambda: tmp_path / "data" / "raw")
    clean = tmp_path / "data" / "clean" / "panel.parquet"
    write_parquet(frame, clean)
    write_parquet(frame, clean)  # must not raise


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path, frame: pd.DataFrame):
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest(created_at="2026-01-01T00:00:00")
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-01", attempts=1)
    m.save(tmp_path)

    back = Manifest.load(tmp_path)
    assert back.tickers["A"].rows == 50
    assert back.tickers["A"].status == "ok"
    assert back.tickers["A"].first_date == "2020-01-01"
    assert back.tickers["A"].digest is not None


def test_manifest_load_on_empty_directory(tmp_path: Path):
    m = Manifest.load(tmp_path)
    assert len(m) == 0
    assert m.created_at


def test_manifest_save_is_valid_json(tmp_path: Path, frame: pd.DataFrame):
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-01", attempts=2)
    saved = m.save(tmp_path)
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["tickers"]["A"]["attempts"] == 2


def test_failure_is_recorded_with_its_error(tmp_path: Path):
    m = Manifest()
    m.record_failure("BAD", "boom", start="2020-01-01", end="2020-03-01", attempts=4)
    assert m.failed() == ["BAD"]
    assert m.completed() == []
    assert "boom" in m.tickers["BAD"].error


def test_pending_is_the_resume_decision(tmp_path: Path, frame: pd.DataFrame):
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-01", attempts=1)
    m.record_failure("B", "network", start="2020-01-01", end="2020-03-01", attempts=4)

    pending = m.pending(["A", "B", "C"], "2020-01-01", "2020-03-01")
    assert "A" not in pending, "a completed ticker must not be re-fetched"
    assert "B" in pending, "a failed ticker must be retried on the next run"
    assert "C" in pending, "an unseen ticker must be fetched"


def test_widening_the_window_forces_a_refetch(frame: pd.DataFrame, tmp_path):
    """Asking for more history than was fetched must re-fetch."""
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-01", attempts=1)
    assert m.pending(["A"], "2020-01-01", "2026-01-01") == ["A"], "later end must re-fetch"
    assert m.pending(["A"], "2015-01-01", "2020-03-01") == ["A"], "earlier start must re-fetch"


def test_repeating_the_identical_request_is_a_no_op(frame: pd.DataFrame, tmp_path):
    """The resume fast path: same window, already answered."""
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2026-08-01", attempts=1)
    assert m.pending(["A"], "2020-01-01", "2026-08-01") == []


def test_a_delisted_name_is_not_refetched_forever(frame: pd.DataFrame, tmp_path):
    """The bug this test exists to prevent.

    A stock that stopped trading in 2020 will never have a 2026 bar. Judging
    coverage by the last stored date would put it back in the queue on every
    run, spending rate-limit budget to receive the same answer each time.
    """
    path = write_parquet(frame, tmp_path / "DEAD.parquet")
    m = Manifest()
    m.record_success(
        "DEAD", frame, path=path, start="2015-01-01", end="2026-08-01", attempts=1
    )
    # Last bar is 2020-03-09; the request ran to 2026-08-01 and was satisfied.
    assert m.tickers["DEAD"].last_date < "2026-01-01"
    assert m.pending(["DEAD"], "2015-01-01", "2026-08-01") == []


def test_coverage_tolerates_a_week_of_holidays(frame: pd.DataFrame, tmp_path):
    """Last bar 2020-03-09; asking to 2020-03-13 must not force a re-fetch."""
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-13", attempts=1)
    assert m.pending(["A"], "2020-01-01", "2020-03-13") == []


def test_empty_download_is_not_complete():
    rec = TickerRecord(symbol="X", status="empty", rows=0)
    assert not rec.is_complete
    assert not rec.covers("2020-01-01", "2020-02-01")


def test_manifest_summary_counts_every_status(tmp_path: Path, frame: pd.DataFrame):
    path = write_parquet(frame, tmp_path / "A.parquet")
    m = Manifest()
    m.record_success("A", frame, path=path, start="2020-01-01", end="2020-03-01", attempts=1)
    m.record_failure("B", "x", start="2020-01-01", end="2020-03-01", attempts=4)
    text = m.summary()
    assert "1 ok" in text and "1 failed" in text
