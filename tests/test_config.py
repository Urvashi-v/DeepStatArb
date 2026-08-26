"""Configuration: validation, immutability, and the two lookahead guards.

Most of these tests plant a bad value and assert that loading *fails*. That is
the point of the module: a research config that accepts nonsense produces a
backtest that looks fine and is not. Every rejection here corresponds to a
specific way the spec says this project silently lies.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from dsa.config import Config, ConfigError, load_config


def _edit(config_dir: Path, filename: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Apply ``mutate`` to one YAML file in a temporary config tree."""
    path = config_dir / filename
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ===========================================================================
# The happy path
# ===========================================================================


def test_repo_config_loads(cfg: Config):
    assert cfg.base.project_name == "deepstatarb"
    assert cfg.backtest.formation_months == 36
    assert cfg.backtest.trading_months == 6
    assert cfg.backtest.signal.z_entry == 2.0


def test_config_is_immutable(cfg: Config):
    """A parameter that can change mid-run makes a result unreproducible."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.backtest.signal.z_entry = 3.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.base.seed = 1  # type: ignore[misc]


def test_hash_is_stable_across_loads(repo_config_dir: Path):
    """Same YAML, same digest --- otherwise the trial count means nothing."""
    assert load_config(repo_config_dir).hash == load_config(repo_config_dir).hash


def test_hash_changes_when_any_parameter_changes(tmp_config_dir: Path):
    before = load_config(tmp_config_dir).hash
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["signal"].update(z_entry=2.5))
    after = load_config(tmp_config_dir).hash
    assert before != after, (
        "changing z_entry did not change the config hash. The deflated Sharpe "
        "counts trials by hash, so a blind hash under-counts trials and "
        "over-states significance (spec Sec 10.2)."
    )


def test_hash_ignores_where_the_yaml_lives(repo_config_dir: Path, tmp_config_dir: Path):
    """A copied config tree is the same configuration."""
    assert load_config(repo_config_dir).hash == load_config(tmp_config_dir).hash


def test_universe_hash_tracks_only_the_ticker_list(tmp_config_dir: Path):
    before = load_config(tmp_config_dir).universe_hash
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["signal"].update(z_entry=2.5))
    assert load_config(tmp_config_dir).universe_hash == before

    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(
            tickers=["A.NS", "B.NS"], sectors={"A.NS": "IT", "B.NS": "IT"}
        ),
    )
    assert load_config(tmp_config_dir).universe_hash != before


def test_n_candidate_pairs_is_the_multiple_testing_denominator(tmp_config_dir: Path):
    """N(N-1)/2 --- spec Sec 4.1. For 200 names that is 19,900."""
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(
            tickers=[f"T{i}.NS" for i in range(200)],
            sectors={f"T{i}.NS": "IT" for i in range(200)},
        ),
    )
    assert load_config(tmp_config_dir).universe.n_candidate_pairs == 19_900


def test_n_windows_matches_the_spec_scheme(cfg: Config):
    """36/6/6 over ten years gives fourteen windows --- the spec's number (Sec 4.2).

    (120 - 36 - 6) / 6 + 1 = 14. If this ever stops being 14 for a ten-year
    sample, the walk-forward scheme changed and the spec's "roughly fourteen
    non-overlapping trading windows" no longer describes what is being run.
    """
    assert cfg.backtest.n_windows(120) == 14
    assert cfg.backtest.n_windows(36) == 0  # no room for a trading window
    assert cfg.backtest.n_windows(12) == 0


# ===========================================================================
# THE LOOKAHEAD GUARDS --- spec Sec 14, bugs #1 and #4
# ===========================================================================


@pytest.mark.leakage
def test_same_bar_fill_is_rejected(tmp_config_dir: Path):
    """Bug #4: you cannot fill at the close that generated the signal."""
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["execution"].update(lag_bars=0))
    with pytest.raises(ConfigError, match="LOOKAHEAD GUARD"):
        load_config(tmp_config_dir)


@pytest.mark.leakage
def test_allow_same_bar_fill_flag_is_rejected(tmp_config_dir: Path):
    _edit(
        tmp_config_dir, "backtest.yaml", lambda d: d["execution"].update(allow_same_bar_fill=True)
    )
    with pytest.raises(ConfigError, match="LOOKAHEAD GUARD"):
        load_config(tmp_config_dir)


@pytest.mark.leakage
def test_unshifted_zscore_is_rejected(tmp_config_dir: Path):
    """Bug #1: the trailing window must end at t-1."""
    _edit(tmp_config_dir, "models.yaml", lambda d: d["ols"].update(zscore_shift=0))
    with pytest.raises(ConfigError, match="LOOKAHEAD GUARD"):
        load_config(tmp_config_dir)


@pytest.mark.leakage
def test_lookahead_demo_escape_hatch_works(tmp_config_dir: Path, monkeypatch):
    """The bug can be reproduced deliberately, but only under a loud flag.

    The report benefits from showing how much Sharpe the full-sample z-score
    manufactures. That has to be possible --- and it has to be impossible by
    accident.
    """
    _edit(tmp_config_dir, "models.yaml", lambda d: d["ols"].update(zscore_shift=0))
    monkeypatch.setenv("DSA_ALLOW_LOOKAHEAD_DEMO", "1")
    cfg = load_config(tmp_config_dir)
    assert cfg.models.ols.zscore_shift == 0


@pytest.mark.leakage
def test_random_train_test_split_is_rejected(tmp_config_dir: Path):
    """Bug #6: a random split leaks across autocorrelated, overlapping trades."""
    _edit(tmp_config_dir, "models.yaml", lambda d: d["filter"].update(split="random"))
    with pytest.raises(ConfigError, match="temporal_only"):
        load_config(tmp_config_dir)


@pytest.mark.leakage
def test_unadjusted_prices_are_rejected(tmp_config_dir: Path):
    """Bug #2: an unadjusted split shows as a 50% one-day divergence."""
    _edit(tmp_config_dir, "universe.yaml", lambda d: d.update(price_field="close"))
    with pytest.raises(ConfigError, match="UNADJUSTED"):
        load_config(tmp_config_dir)


@pytest.mark.leakage
def test_tuning_delta_outside_the_formation_window_is_rejected(tmp_config_dir: Path):
    _edit(tmp_config_dir, "models.yaml", lambda d: d["kalman"].update(tune_on="trading"))
    with pytest.raises(ConfigError, match="formation_only"):
        load_config(tmp_config_dir)


def test_overlapping_trading_windows_are_rejected(tmp_config_dir: Path):
    """Overlapping windows double-count days and shrink every interval."""
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d.update(step_months=3))
    with pytest.raises(ConfigError, match="OVERLAP"):
        load_config(tmp_config_dir)


# ===========================================================================
# Structural validation
# ===========================================================================


def test_unknown_key_is_rejected(tmp_config_dir: Path):
    """A typo'd key is a parameter that silently never took effect."""
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["signal"].update(z_entery=2.5))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(tmp_config_dir)


def test_missing_key_is_rejected(tmp_config_dir: Path):
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["signal"].pop("z_stop"))
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(tmp_config_dir)


def test_missing_file_is_rejected(tmp_config_dir: Path):
    (tmp_config_dir / "costs.yaml").unlink()
    with pytest.raises(ConfigError, match="missing configuration file"):
        load_config(tmp_config_dir)


def test_malformed_yaml_is_rejected(tmp_config_dir: Path):
    (tmp_config_dir / "base.yaml").write_text("seed: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(tmp_config_dir)


@pytest.mark.parametrize(
    ("z_entry", "z_exit", "z_stop", "expected"),
    [
        (2.0, 2.5, 3.5, "z_exit"),  # exit above entry: trade closes at birth
        (2.0, 0.5, 1.5, "z_stop"),  # stop below entry: stop trips immediately
    ],
)
def test_incoherent_thresholds_are_rejected(
    tmp_config_dir: Path, z_entry, z_exit, z_stop, expected
):
    _edit(
        tmp_config_dir,
        "backtest.yaml",
        lambda d: d["signal"].update(z_entry=z_entry, z_exit=z_exit, z_stop=z_stop),
    )
    with pytest.raises(ConfigError, match=expected):
        load_config(tmp_config_dir)


def test_half_life_band_must_be_ordered(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "backtest.yaml",
        lambda d: d["screening"].update(min_half_life_days=90, max_half_life_days=60),
    )
    with pytest.raises(ConfigError, match="min_half_life_days"):
        load_config(tmp_config_dir)


def test_fdr_alpha_must_be_a_probability(tmp_config_dir: Path):
    _edit(tmp_config_dir, "backtest.yaml", lambda d: d["screening"].update(fdr_alpha=10))
    with pytest.raises(ConfigError, match="fdr_alpha"):
        load_config(tmp_config_dir)


def test_book_that_cannot_be_deployed_is_rejected(tmp_config_dir: Path):
    """10 pairs x 5% cap = 50% of capital permanently idle, silently."""
    _edit(
        tmp_config_dir,
        "backtest.yaml",
        lambda d: d["portfolio"].update(max_weight_per_pair=0.05, max_concurrent_pairs=10),
    )
    with pytest.raises(ConfigError, match="never be fully deployed"):
        load_config(tmp_config_dir)


def test_kalman_warmup_below_sixty_is_rejected(tmp_config_dir: Path):
    """Spec Sec 3: discard the first 60-100 observations while beta converges."""
    _edit(tmp_config_dir, "models.yaml", lambda d: d["kalman"].update(warmup_obs=10))
    with pytest.raises(ConfigError, match="warmup_obs"):
        load_config(tmp_config_dir)


def test_deep_trees_are_rejected(tmp_config_dir: Path):
    """Spec Sec 8.3: a few thousand entry events. Resist deep trees."""
    _edit(tmp_config_dir, "models.yaml", lambda d: d["xgboost"].update(max_depth=12))
    with pytest.raises(ConfigError, match="max_depth"):
        load_config(tmp_config_dir)


def test_globally_normalised_lstm_inputs_are_rejected(tmp_config_dir: Path):
    """Spec Sec 9: the full-sample z-score bug wearing a different hat."""
    _edit(
        tmp_config_dir,
        "models.yaml",
        lambda d: d["lstm"].update(enabled=True, normalisation="global"),
    )
    with pytest.raises(ConfigError, match="per_window"):
        load_config(tmp_config_dir)


def test_hand_entered_trial_count_is_rejected(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "models.yaml",
        lambda d: d["validity"]["deflated_sharpe"].update(trials_from_config_log=False),
    )
    with pytest.raises(ConfigError, match="trials_from_config_log"):
        load_config(tmp_config_dir)


# ===========================================================================
# Costs: units and the short-leg decision
# ===========================================================================


def test_basis_point_unit_error_is_caught(tmp_config_dir: Path):
    """0.02% is 2 bps. Writing 0.02 as a percentage would be a 100x error."""
    _edit(tmp_config_dir, "costs.yaml", lambda d: d.update(stt_sell_bps=200))
    with pytest.raises(ConfigError, match="BASIS POINTS"):
        load_config(tmp_config_dir)


def test_gst_is_a_fraction_not_a_percentage(tmp_config_dir: Path):
    _edit(tmp_config_dir, "costs.yaml", lambda d: d.update(gst_rate=18))
    with pytest.raises(ConfigError, match="gst_rate"):
        load_config(tmp_config_dir)


def test_slippage_buckets_must_cover_every_bucket(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "costs.yaml",
        lambda d: d.update(slippage_bps_per_side={1: 2.0, 2: 4.0}),
    )
    with pytest.raises(ConfigError, match="slippage_bps_per_side"):
        load_config(tmp_config_dir)


def test_short_leg_decision_cannot_be_claimed_without_a_method(tmp_config_dir: Path):
    """Spec Sec 6.1: address the short-sale constraint or the backtest is fiction."""
    _edit(tmp_config_dir, "costs.yaml", lambda d: d["short_leg"].update(decided=True, method=None))
    with pytest.raises(ConfigError, match="method is null"):
        load_config(tmp_config_dir)


def test_short_leg_method_must_be_one_of_the_three_legitimate_options(tmp_config_dir: Path):
    _edit(tmp_config_dir, "costs.yaml", lambda d: d["short_leg"].update(method="just_short_it"))
    with pytest.raises(ConfigError, match="method must be one of"):
        load_config(tmp_config_dir)


# ===========================================================================
# The frozen-universe contract --- spec Sec 11.4
# ===========================================================================


def test_an_unfrozen_universe_may_be_empty(tmp_config_dir: Path):
    """Before the list is built, an empty universe must still load.

    Written as an invariant rather than an assertion about today's config:
    the repo universe was empty on Day 1 and frozen on Day 2, and a test that
    pins either state has to be edited every time the project moves --- which
    is how tests stop being trusted.
    """
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(frozen=False, version=0, as_of=None, tickers=[], sectors={}),
    )
    loaded = load_config(tmp_config_dir)
    assert loaded.universe.is_populated is False
    assert loaded.universe.n_candidate_pairs == 0


def test_repo_universe_is_internally_consistent(cfg: Config):
    """Whatever state the repo config is in, it must be a coherent one."""
    u = cfg.universe
    if u.frozen:
        assert u.is_populated, "frozen but empty"
        assert u.as_of is not None, "frozen but no as_of date"
        assert u.version > 0, "frozen but version 0"
        assert set(u.tickers) <= set(u.sectors), "frozen but some tickers lack a sector"
        assert len(set(u.tickers)) == len(u.tickers), "duplicate tickers"
        assert u.n_candidate_pairs == len(u.tickers) * (len(u.tickers) - 1) // 2
    else:
        assert u.version == 0 or not u.is_populated


def test_frozen_universe_must_not_be_empty(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(frozen=True, version=1, tickers=[], sectors={}),
    )
    with pytest.raises(ConfigError, match="tickers` is empty"):
        load_config(tmp_config_dir)


def test_frozen_universe_requires_a_sector_for_every_ticker(tmp_config_dir: Path):
    """Sectors drive the economic prior and the survival stratification."""
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(
            frozen=True,
            version=1,
            as_of="2026-08-23",
            tickers=["A.NS", "B.NS"],
            sectors={"A.NS": "IT"},
        ),
    )
    with pytest.raises(ConfigError, match="no sector mapping"):
        load_config(tmp_config_dir)


def test_frozen_universe_rejects_duplicates(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(
            frozen=True,
            version=1,
            as_of="2026-08-23",
            tickers=["A.NS", "A.NS"],
            sectors={"A.NS": "IT"},
        ),
    )
    with pytest.raises(ConfigError, match="duplicate tickers"):
        load_config(tmp_config_dir)


def test_a_properly_frozen_universe_loads(tmp_config_dir: Path):
    _edit(
        tmp_config_dir,
        "universe.yaml",
        lambda d: d.update(
            frozen=True,
            version=1,
            as_of="2026-08-23",
            tickers=["A.NS", "B.NS", "C.NS"],
            sectors={"A.NS": "IT", "B.NS": "IT", "C.NS": "BANK"},
        ),
    )
    loaded = load_config(tmp_config_dir)
    assert loaded.universe.frozen is True
    assert loaded.universe.n_candidate_pairs == 3


# ===========================================================================
# Serialisation
# ===========================================================================


def test_to_dict_round_trips_to_json(cfg: Config):
    import json

    payload = cfg.to_dict()
    assert json.loads(json.dumps(payload, default=str))["backtest"]["formation_months"] == 36


def test_summary_reports_the_universe_state(cfg: Config):
    text = cfg.summary()
    assert cfg.hash in text
    assert f"frozen={cfg.universe.frozen}" in text
    assert f"n={len(cfg.universe.tickers)}" in text
