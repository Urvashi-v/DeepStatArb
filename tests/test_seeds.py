"""Reproducible randomness.

The property that matters for the null control (spec Sec 10.1) is not just
"the same seed gives the same numbers". It is that named streams are
independent of each other and of the order in which they are created, so that
adding a component tomorrow does not silently change the 500-run null
distribution reported yesterday.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsa.seeds import seed_report, set_global_seed, spawn_rng, temporary_seed

SEED = 20260823


def test_same_name_gives_the_same_stream():
    a = spawn_rng(SEED, "null_control", run=7).standard_normal(100)
    b = spawn_rng(SEED, "null_control", run=7).standard_normal(100)
    assert np.array_equal(a, b)


def test_different_names_give_different_streams():
    a = spawn_rng(SEED, "null_control", run=7).standard_normal(100)
    b = spawn_rng(SEED, "null_control", run=8).standard_normal(100)
    c = spawn_rng(SEED, "bootstrap", run=7).standard_normal(100)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_different_master_seeds_give_different_streams():
    a = spawn_rng(SEED, "x").standard_normal(100)
    b = spawn_rng(SEED + 1, "x").standard_normal(100)
    assert not np.array_equal(a, b)


def test_keyword_label_order_does_not_matter():
    """Call-site ordering must not change a stream, or refactors break results."""
    a = spawn_rng(SEED, "boot", pair="RELIANCE_ONGC", window=3).standard_normal(50)
    b = spawn_rng(SEED, "boot", window=3, pair="RELIANCE_ONGC").standard_normal(50)
    assert np.array_equal(a, b)


def test_adding_a_new_stream_does_not_disturb_existing_ones():
    """The property sequential ``SeedSequence.spawn()`` does NOT give you.

    With sequential spawning, the n-th child depends on how many children were
    requested before it, so introducing a new random component anywhere
    upstream shifts every stream downstream of it --- and every previously
    reported number with it. Name-derived streams do not have that coupling.
    """
    before = spawn_rng(SEED, "bootstrap").standard_normal(50)
    # Simulate a new component being added to the codebase.
    _ = spawn_rng(SEED, "brand_new_component").standard_normal(1000)
    after = spawn_rng(SEED, "bootstrap").standard_normal(50)
    assert np.array_equal(before, after)


def test_streams_are_reproducible_across_processes(project_root):
    """BLAKE2b, not ``hash()`` --- Python's str hash is randomised per process.

    If label hashing used the builtin ``hash()``, two runs of the same script
    would draw different numbers unless ``PYTHONHASHSEED`` happened to match.
    This test runs a child process under two different hash seeds and requires
    the draw to be identical both times.
    """
    import os
    import subprocess
    import sys

    expected = int(spawn_rng(SEED, "null_control", run=3).integers(0, 2**31))
    snippet = (
        "from dsa.seeds import spawn_rng; "
        f"print(spawn_rng({SEED}, 'null_control', run=3).integers(0, 2**31))"
    )

    drawn = []
    for hash_seed in ("0", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(project_root / "src")}
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        drawn.append(int(out.stdout.strip()))

    assert drawn == [expected, expected], (
        f"draws {drawn} differ from the in-process value {expected}. Label hashing is "
        "picking up Python's randomised hash(), so named streams are not reproducible "
        "between processes."
    )


def test_streams_are_statistically_independent():
    """A crude but meaningful check: near-zero correlation between streams."""
    a = spawn_rng(SEED, "stream_a").standard_normal(20_000)
    b = spawn_rng(SEED, "stream_b").standard_normal(20_000)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.03, f"streams correlate at {corr:.4f}; they are not independent"


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        spawn_rng(-1, "x")


def test_set_global_seed_makes_legacy_apis_reproducible():
    set_global_seed(SEED)
    a_np, a_py = np.random.rand(10), __import__("random").random()
    set_global_seed(SEED)
    b_np, b_py = np.random.rand(10), __import__("random").random()
    assert np.array_equal(a_np, b_np)
    assert a_py == b_py


def test_set_global_seed_reports_what_it_seeded():
    record = set_global_seed(SEED)
    assert record["seed"] == SEED
    assert "random" in record["seeded"]
    assert any("numpy" in s for s in record["seeded"])


def test_set_global_seed_rejects_bad_input():
    with pytest.raises(ValueError):
        set_global_seed(-5)


def test_temporary_seed_restores_state():
    set_global_seed(SEED)
    baseline = np.random.rand(5)

    set_global_seed(SEED)
    with temporary_seed(999):
        np.random.rand(100)  # consume the temporary stream
    restored = np.random.rand(5)

    assert np.array_equal(baseline, restored), (
        "temporary_seed leaked its state; a helper that silently advances the "
        "global RNG makes every downstream result depend on whether it ran"
    )


def test_seed_report_is_a_stable_fingerprint():
    labels = ["screening", "null_control", "bootstrap"]
    assert seed_report(SEED, labels) == seed_report(SEED, labels)
    assert seed_report(SEED, labels) != seed_report(SEED + 1, labels)
    assert set(seed_report(SEED, labels)) == set(labels)
