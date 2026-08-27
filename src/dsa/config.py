"""Typed, immutable, hashable configuration.

Three properties matter here, and each of them exists to prevent a specific
class of research bug.

**Immutable.** Every config object is a frozen dataclass. A parameter that can
be mutated halfway through a walk-forward loop produces results that cannot be
reproduced and cannot be explained.

**Validated at load time.** Structural errors (a typo'd YAML key, a negative
window, ``z_exit`` above ``z_entry``) raise immediately rather than producing a
plausible-looking but wrong backtest six hours later. Unknown keys are a hard
error: a silently ignored ``z_entery: 2.0`` is exactly the kind of bug that
never gets found.

**Hashable.** ``Config.hash`` is a stable digest of every parameter in play.
The deflated Sharpe ratio (spec Sec 10.2) requires an honest count of how many
distinct configurations were tried; that count is only trustworthy if each run
is identified by the content of its configuration rather than by memory.

Two dangerous settings are additionally guarded here, because they are the two
most common ways this project silently lies (spec Sec 14, bugs #1 and #4):

* ``models.ols.zscore_shift`` must be >= 1 --- the trailing-window ``.shift(1)``.
* ``backtest.execution.allow_same_bar_fill`` must be ``false`` --- you cannot
  trade at the close that generated the signal.

Both can be overridden *only* by setting ``DSA_ALLOW_LOOKAHEAD_DEMO=1``, which
exists so the report can deliberately reproduce the bug to show its size. It is
never set during a real run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import yaml

from dsa.paths import config_dir as default_config_dir

__all__ = [
    "ConfigError",
    "Config",
    "BaseConfig",
    "UniverseConfig",
    "CostsConfig",
    "BacktestConfig",
    "ModelsConfig",
    "QualityConfig",
    "SelectionConfig",
    "load_config",
    "get_config",
    "clear_config_cache",
]

T = TypeVar("T")

_LOOKAHEAD_DEMO_ENV = "DSA_ALLOW_LOOKAHEAD_DEMO"
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------


def _build(cls: type[T], data: Mapping[str, Any] | None, where: str, **extra: Any) -> T:
    """Construct a dataclass from a mapping, rejecting unknown/missing keys.

    Keys beginning with ``_`` are treated as annotations and ignored, so YAML
    can carry ``_note:`` entries without them becoming parameters.
    """
    if data is None:
        raise ConfigError(f"{where}: section is missing or empty")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    supplied = {k: v for k, v in data.items() if not str(k).startswith("_")}
    supplied.update(extra)

    unknown = set(supplied) - names
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(names)}. "
            "Unknown keys are rejected because a silently ignored typo is a "
            "parameter that never took effect."
        )

    required = {
        f.name
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    missing = required - set(supplied)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {sorted(missing)}")

    return cls(**supplied)  # type: ignore[call-arg]


def _check(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ConfigError(f"{where}: {message}")


def _parse_date(value: Any, where: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{where}: expected an ISO date (YYYY-MM-DD), got {value!r}") from exc


def _lookahead_demo_enabled() -> bool:
    return os.environ.get(_LOOKAHEAD_DEMO_ENV, "").strip() in {"1", "true", "TRUE", "yes"}


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseConfig:
    """Project-wide settings that belong to no single subsystem."""

    project_name: str
    seed: int
    log_level: str
    log_to_file: bool
    figure_dpi: int
    figure_format: str

    def __post_init__(self) -> None:
        w = "base.yaml"
        _check(isinstance(self.seed, int) and self.seed >= 0, w, "seed must be a non-negative int")
        _check(
            self.log_level.upper() in _VALID_LOG_LEVELS,
            w,
            f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {self.log_level!r}",
        )
        _check(self.figure_dpi > 0, w, "figure_dpi must be positive")
        _check(
            self.figure_format in {"png", "pdf", "svg"},
            w,
            "figure_format must be png, pdf or svg",
        )


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivorshipConfig:
    acknowledged: bool
    direction: str
    mitigation: str

    def __post_init__(self) -> None:
        w = "universe.yaml:survivorship_bias"
        _check(
            self.direction in {"optimistic", "pessimistic", "unknown"},
            w,
            "direction must be optimistic, pessimistic or unknown",
        )


@dataclass(frozen=True)
class UniverseConfig:
    """The frozen, versioned trading universe (spec Sec 11.4)."""

    name: str
    version: int
    frozen: bool
    as_of: date | None
    start_date: date
    end_date: date
    source: str
    ticker_suffix: str
    price_field: str
    benchmark: str
    vix_symbol: str
    target_size: int | None
    min_history_years: float
    min_median_turnover_inr: float
    max_missing_frac: float
    tickers: tuple[str, ...]
    sectors: Mapping[str, str]
    smoke_test_tickers: tuple[str, ...]
    survivorship_bias: SurvivorshipConfig

    def __post_init__(self) -> None:
        w = "universe.yaml"
        _check(self.start_date < self.end_date, w, "start_date must precede end_date")
        # target_size mirrors selection.yaml's max_size, which is null when no
        # cap is applied. It is a RECORD of the rule that produced this list,
        # not an input --- selection.yaml is authoritative.
        if self.target_size is not None:
            _check(self.target_size > 0, w, "target_size must be positive if set")
        _check(self.min_history_years > 0, w, "min_history_years must be positive")
        _check(0.0 <= self.max_missing_frac < 1.0, w, "max_missing_frac must be in [0, 1)")
        _check(
            self.price_field in {"adj_close", "close"},
            w,
            "price_field must be adj_close or close",
        )
        # Spec Sec 11.2: adjusted prices are non-negotiable.
        if self.price_field == "close":
            raise ConfigError(
                f"{w}: price_field='close' uses UNADJUSTED prices. A 1:2 split then shows as a "
                "50% single-day divergence and the strategy takes an enormous position in an "
                "event that never happened (spec Sec 11.2, bug #2). Use 'adj_close'."
            )
        _check(self.source in {"yfinance", "jugaad_data"}, w, "unknown data source")

        # The frozen-universe contract. Only enforced once frozen is claimed,
        # so the config is loadable on Day 1 with an empty list.
        if self.frozen:
            _check(bool(self.tickers), w, "frozen=true but `tickers` is empty")
            _check(self.as_of is not None, w, "frozen=true but `as_of` is null")
            _check(self.version > 0, w, "frozen=true but version is still 0")
            missing = [t for t in self.tickers if t not in self.sectors]
            _check(
                not missing,
                w,
                f"{len(missing)} ticker(s) have no sector mapping, e.g. {missing[:5]}. "
                "Sectors drive the economic prior (Sec 4.1) and the survival "
                "stratification (Sec 7.2); every ticker needs one.",
            )
            dupes = sorted({t for t in self.tickers if self.tickers.count(t) > 1})
            _check(not dupes, w, f"duplicate tickers: {dupes}")

    @property
    def is_populated(self) -> bool:
        return bool(self.tickers)

    @property
    def n_candidate_pairs(self) -> int:
        """N(N-1)/2 --- the multiple-testing denominator (spec Sec 4.1)."""
        n = len(self.tickers)
        return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortLegConfig:
    """How the short leg is executed. Spec Sec 6.1 --- must not stay implicit."""

    method: str | None
    decided: bool
    futures_roll_cost_bps: float
    margin_fraction: float

    ALLOWED = ("single_stock_futures", "intraday_only", "cash_with_caveat")

    def __post_init__(self) -> None:
        w = "costs.yaml:short_leg"
        if self.method is not None:
            _check(self.method in self.ALLOWED, w, f"method must be one of {self.ALLOWED}")
        if self.decided:
            _check(self.method is not None, w, "decided=true but method is null")
        _check(self.futures_roll_cost_bps >= 0, w, "futures_roll_cost_bps must be >= 0")
        _check(0 < self.margin_fraction <= 1, w, "margin_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CostsConfig:
    """The Indian transaction cost stack (spec Sec 6.2).

    All rate fields are BASIS POINTS of traded notional except ``gst_rate``
    (a fraction) and ``brokerage_flat_inr`` (rupees per order).
    """

    rates_as_of: date | None
    verified: bool
    segment: str
    brokerage_flat_inr: float
    brokerage_bps: float
    exchange_txn_bps: float
    stt_sell_bps: float
    sebi_bps: float
    stamp_duty_buy_bps: float
    gst_rate: float
    slippage_bps_per_side: Mapping[int, float]
    n_liquidity_buckets: int
    scenario_bps_round_trip: tuple[float, ...]
    short_leg: ShortLegConfig

    def __post_init__(self) -> None:
        w = "costs.yaml"
        _check(
            self.segment in {"equity_futures", "equity_cash"},
            w,
            "segment must be equity_futures or equity_cash",
        )
        for name in (
            "brokerage_flat_inr",
            "brokerage_bps",
            "exchange_txn_bps",
            "stt_sell_bps",
            "sebi_bps",
            "stamp_duty_buy_bps",
        ):
            _check(getattr(self, name) >= 0, w, f"{name} must be >= 0")
        _check(0 <= self.gst_rate < 1, w, "gst_rate is a fraction and must be in [0, 1)")
        _check(self.n_liquidity_buckets >= 1, w, "n_liquidity_buckets must be >= 1")
        expected = set(range(1, self.n_liquidity_buckets + 1))
        _check(
            set(self.slippage_bps_per_side) == expected,
            w,
            f"slippage_bps_per_side must have exactly the keys {sorted(expected)}, "
            f"got {sorted(self.slippage_bps_per_side)}",
        )
        _check(
            all(v >= 0 for v in self.slippage_bps_per_side.values()),
            w,
            "slippage must be >= 0 in every bucket",
        )
        _check(bool(self.scenario_bps_round_trip), w, "scenario_bps_round_trip must not be empty")
        _check(
            all(v >= 0 for v in self.scenario_bps_round_trip),
            w,
            "scenario_bps_round_trip values must be >= 0",
        )
        # A sanity band on the rates. These are bps: 0.02% is 2, not 0.0002.
        # Catching a unit error here is worth far more than the strictness costs.
        for name in ("brokerage_bps", "exchange_txn_bps", "stt_sell_bps", "stamp_duty_buy_bps"):
            _check(
                getattr(self, name) < 100,
                w,
                f"{name}={getattr(self, name)} exceeds 100 bps (1%). These fields are in BASIS "
                "POINTS; a value this large is almost certainly a unit error.",
            )


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionConfig:
    """Signal-to-fill alignment. Spec Sec 6.3, bug #4."""

    signal_price: str
    fill_price: str
    lag_bars: int
    allow_same_bar_fill: bool

    def __post_init__(self) -> None:
        w = "backtest.yaml:execution"
        _check(self.signal_price in {"close", "open"}, w, "signal_price must be close or open")
        _check(
            self.fill_price in {"open", "close", "vwap"}, w, "fill_price must be open/close/vwap"
        )
        _check(self.lag_bars >= 0, w, "lag_bars must be >= 0")
        if (self.lag_bars < 1 or self.allow_same_bar_fill) and not _lookahead_demo_enabled():
            raise ConfigError(
                f"{w}: LOOKAHEAD GUARD. lag_bars={self.lag_bars}, "
                f"allow_same_bar_fill={self.allow_same_bar_fill}. You computed z from the close "
                "of t; you cannot also fill at the close of t. Signals on close of t execute at "
                "the open of t+1 (spec Sec 6.3, bug #4). To reproduce the bug deliberately for "
                f"the report, set {_LOOKAHEAD_DEMO_ENV}=1."
            )


@dataclass(frozen=True)
class SignalConfig:
    """Entry/exit thresholds. Spec Sec 5. Fixed a priori."""

    z_entry: float
    z_exit: float
    z_stop: float
    time_stop_halflives: float
    max_holding_days: int

    def __post_init__(self) -> None:
        w = "backtest.yaml:signal"
        _check(self.z_entry > 0, w, "z_entry must be positive")
        _check(self.z_exit >= 0, w, "z_exit must be >= 0")
        _check(
            self.z_exit < self.z_entry,
            w,
            f"z_exit ({self.z_exit}) must be below z_entry ({self.z_entry}); "
            "otherwise a trade exits the instant it opens",
        )
        _check(
            self.z_stop > self.z_entry,
            w,
            f"z_stop ({self.z_stop}) must exceed z_entry ({self.z_entry}); "
            "otherwise every entry immediately trips the stop",
        )
        _check(self.time_stop_halflives > 0, w, "time_stop_halflives must be positive")
        _check(self.max_holding_days > 0, w, "max_holding_days must be positive")


@dataclass(frozen=True)
class KillSwitchConfig:
    """Pair kill switch. Spec Sec 5."""

    enabled: bool
    rolling_coint_window: int
    pval_threshold: float
    consecutive_days: int

    def __post_init__(self) -> None:
        w = "backtest.yaml:kill_switch"
        _check(self.rolling_coint_window > 30, w, "rolling_coint_window must exceed 30 days")
        _check(0 < self.pval_threshold < 1, w, "pval_threshold must be in (0, 1)")
        _check(self.consecutive_days >= 1, w, "consecutive_days must be >= 1")


@dataclass(frozen=True)
class ScreeningConfig:
    """Pair screening and multiple-testing control. Spec Sec 4."""

    fdr_alpha: float
    fdr_method: str
    raw_pval_report_level: float
    both_directions: bool
    pval_combine: str
    same_sector_only: bool
    report_unrestricted: bool
    min_half_life_days: float
    max_half_life_days: float
    min_overlapping_obs: int

    def __post_init__(self) -> None:
        w = "backtest.yaml:screening"
        _check(0 < self.fdr_alpha < 1, w, "fdr_alpha must be in (0, 1)")
        _check(
            self.fdr_method in {"fdr_bh", "fdr_by", "bonferroni", "holm"},
            w,
            "unsupported fdr_method",
        )
        _check(0 < self.raw_pval_report_level < 1, w, "raw_pval_report_level must be in (0, 1)")
        _check(self.pval_combine in {"max", "min"}, w, "pval_combine must be max or min")
        _check(
            0 < self.min_half_life_days < self.max_half_life_days,
            w,
            "require 0 < min_half_life_days < max_half_life_days",
        )
        _check(self.min_overlapping_obs > 100, w, "min_overlapping_obs must exceed 100")


@dataclass(frozen=True)
class PortfolioConfig:
    """Capital allocation and position caps. Spec Sec 5.1."""

    initial_capital_inr: float
    sizing: str
    max_weight_per_pair: float
    max_concurrent_pairs: int
    vol_scale: bool

    def __post_init__(self) -> None:
        w = "backtest.yaml:portfolio"
        _check(self.initial_capital_inr > 0, w, "initial_capital_inr must be positive")
        _check(self.sizing in {"dollar_neutral", "equal_weight"}, w, "unknown sizing scheme")
        _check(0 < self.max_weight_per_pair <= 1, w, "max_weight_per_pair must be in (0, 1]")
        _check(self.max_concurrent_pairs >= 1, w, "max_concurrent_pairs must be >= 1")
        _check(
            self.max_weight_per_pair * self.max_concurrent_pairs >= 1.0,
            w,
            f"max_weight_per_pair ({self.max_weight_per_pair}) x max_concurrent_pairs "
            f"({self.max_concurrent_pairs}) = "
            f"{self.max_weight_per_pair * self.max_concurrent_pairs:.2f} < 1.0, so the book can "
            "never be fully deployed. Raise one of them or accept the permanent cash drag "
            "deliberately.",
        )


@dataclass(frozen=True)
class MetricsConfig:
    trading_days_per_year: int
    risk_free_annual: float

    def __post_init__(self) -> None:
        w = "backtest.yaml:metrics"
        _check(200 < self.trading_days_per_year < 300, w, "trading_days_per_year looks wrong")
        _check(-0.1 < self.risk_free_annual < 0.5, w, "risk_free_annual must be a fraction")


@dataclass(frozen=True)
class BacktestConfig:
    formation_months: int
    trading_months: int
    step_months: int
    execution: ExecutionConfig
    signal: SignalConfig
    kill_switch: KillSwitchConfig
    screening: ScreeningConfig
    portfolio: PortfolioConfig
    metrics: MetricsConfig

    def __post_init__(self) -> None:
        w = "backtest.yaml"
        _check(self.formation_months > 0, w, "formation_months must be positive")
        _check(self.trading_months > 0, w, "trading_months must be positive")
        _check(self.step_months > 0, w, "step_months must be positive")
        _check(
            self.formation_months >= 12,
            w,
            "formation_months < 12 gives too few observations for a credible cointegration test",
        )
        if self.step_months < self.trading_months:
            raise ConfigError(
                f"{w}: step_months ({self.step_months}) < trading_months "
                f"({self.trading_months}) makes consecutive trading windows OVERLAP. The same "
                "days would then be counted more than once in the aggregate return series, "
                "inflating the effective sample size and shrinking every confidence interval."
            )

    def n_windows(self, total_months: int) -> int:
        """How many walk-forward windows fit in ``total_months`` of data."""
        usable = total_months - self.formation_months - self.trading_months
        return 0 if usable < 0 else usable // self.step_months + 1


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KalmanConfig:
    """Time-varying hedge ratio. Spec Sec 3."""

    delta: float
    delta_grid: tuple[float, ...]
    observation_noise_R: float
    warmup_obs: int
    beta_init: float
    P_init: float
    tune_on: str
    report_sensitivity: bool

    def __post_init__(self) -> None:
        w = "models.yaml:kalman"
        _check(0 < self.delta < 1, w, "delta must be in (0, 1); Q = delta / (1 - delta)")
        _check(bool(self.delta_grid), w, "delta_grid must not be empty")
        _check(
            all(0 < d < 1 for d in self.delta_grid), w, "every delta in the grid must be in (0, 1)"
        )
        _check(self.observation_noise_R > 0, w, "observation_noise_R must be positive")
        _check(self.P_init > 0, w, "P_init must be positive")
        # Spec Sec 3: "Discard the first 60-100 observations of every pair."
        _check(
            self.warmup_obs >= 60,
            w,
            f"warmup_obs={self.warmup_obs} is below 60. The filter starts from an arbitrary "
            "state (beta_init) and needs time to converge; signals generated before it has "
            "are noise from the initialisation, not from the data (spec Sec 3).",
        )
        _check(
            self.tune_on == "formation_only",
            w,
            "tune_on must be 'formation_only'. Tuning delta on the trading window is "
            "selecting a parameter with data you are pretending not to have seen.",
        )

    @property
    def Q(self) -> float:
        """State-noise variance implied by ``delta``."""
        return self.delta / (1.0 - self.delta)


@dataclass(frozen=True)
class OLSConfig:
    """Static / rolling OLS baselines and the trailing z-score. Spec Sec 2.5."""

    rolling_window_days: int
    zscore_halflife_multiple: float
    zscore_min_window: int
    zscore_max_window: int
    zscore_shift: int

    def __post_init__(self) -> None:
        w = "models.yaml:ols"
        _check(self.rolling_window_days > 20, w, "rolling_window_days must exceed 20")
        _check(self.zscore_halflife_multiple > 0, w, "zscore_halflife_multiple must be positive")
        _check(
            0 < self.zscore_min_window < self.zscore_max_window,
            w,
            "require 0 < zscore_min_window < zscore_max_window",
        )
        if self.zscore_shift < 1 and not _lookahead_demo_enabled():
            raise ConfigError(
                f"{w}: LOOKAHEAD GUARD. zscore_shift={self.zscore_shift}. The rolling mean and "
                "standard deviation used to normalise the spread at time t must be computed on a "
                "window ending at t-1, written as .rolling(w).mean().shift(1). Without the shift "
                "the normalisation sees the bar it is normalising (spec Sec 2.5, bug #1). To "
                f"reproduce the bug deliberately for the report, set {_LOOKAHEAD_DEMO_ENV}=1."
            )


@dataclass(frozen=True)
class LabelConfig:
    """Forward-looking labels for the ML filter. Spec Sec 8.1."""

    horizon_halflife_multiple: float
    horizon_min_days: int
    horizon_max_days: int

    def __post_init__(self) -> None:
        w = "models.yaml:labels"
        _check(self.horizon_halflife_multiple > 0, w, "horizon_halflife_multiple must be positive")
        _check(
            0 < self.horizon_min_days < self.horizon_max_days,
            w,
            "require 0 < horizon_min_days < horizon_max_days",
        )


@dataclass(frozen=True)
class FilterConfig:
    """Train/validate protocol for the ML filter. Spec Sec 8.1, bug #6."""

    split: str
    validation_fraction: float
    threshold_grid: tuple[float, ...]
    report_base_rate_first: bool

    def __post_init__(self) -> None:
        w = "models.yaml:filter"
        if self.split != "temporal_only":
            raise ConfigError(
                f"{w}: split must be 'temporal_only', got {self.split!r}. A random split places "
                "tomorrow's trades in the training set. Spreads are autocorrelated and "
                "overlapping trades share price paths, so this leaks massively and produces a "
                "spectacular, meaningless AUC (spec Sec 8.1, bug #6)."
            )
        _check(0 < self.validation_fraction < 0.5, w, "validation_fraction must be in (0, 0.5)")
        _check(bool(self.threshold_grid), w, "threshold_grid must not be empty")
        _check(
            all(0 <= t <= 1 for t in self.threshold_grid),
            w,
            "threshold_grid values are probabilities and must be in [0, 1]",
        )


@dataclass(frozen=True)
class NullControlConfig:
    n_runs: int
    preserve_sector_structure: bool

    def __post_init__(self) -> None:
        w = "models.yaml:validity.null_control"
        _check(
            self.n_runs >= 100,
            w,
            "n_runs below 100 gives a null distribution too coarse to quote a percentile from",
        )


@dataclass(frozen=True)
class DeflatedSharpeConfig:
    trials_from_config_log: bool

    def __post_init__(self) -> None:
        w = "models.yaml:validity.deflated_sharpe"
        if not self.trials_from_config_log:
            raise ConfigError(
                f"{w}: trials_from_config_log must be true. A hand-entered trial count is a "
                "number you will under-report, and under-reporting N is what the deflated "
                "Sharpe exists to prevent (spec Sec 10.2, Sec 12)."
            )


@dataclass(frozen=True)
class BootstrapConfig:
    method: str
    n_boot: int
    block_length: int | None
    ci_level: float

    def __post_init__(self) -> None:
        w = "models.yaml:validity.bootstrap"
        _check(self.method in {"stationary", "circular", "moving"}, w, "unknown bootstrap method")
        _check(self.n_boot >= 500, w, "n_boot below 500 gives unstable interval endpoints")
        _check(0.5 < self.ci_level < 1, w, "ci_level must be in (0.5, 1)")
        if self.block_length is not None:
            _check(self.block_length >= 1, w, "block_length must be >= 1 if given")


@dataclass(frozen=True)
class ValidityConfig:
    null_control: NullControlConfig
    deflated_sharpe: DeflatedSharpeConfig
    bootstrap: BootstrapConfig


@dataclass(frozen=True)
class ModelsConfig:
    kalman: KalmanConfig
    ols: OLSConfig
    labels: LabelConfig
    xgboost: Mapping[str, Any]
    filter: FilterConfig
    lstm: Mapping[str, Any]
    validity: ValidityConfig

    def __post_init__(self) -> None:
        w = "models.yaml"
        depth = self.xgboost.get("max_depth")
        _check(depth is not None, w, "xgboost.max_depth must be set")
        # Spec Sec 8.3: a few thousand entry events. "Resist deep trees."
        _check(
            1 <= int(depth) <= 6,
            w,
            f"xgboost.max_depth={depth}. With on the order of a few thousand entry events, "
            "depth above ~5 memorises the training window (spec Sec 8.3).",
        )
        if self.lstm.get("enabled"):
            _check(
                self.lstm.get("normalisation") == "per_window",
                w,
                "lstm.normalisation must be 'per_window'. Standardising sequence inputs with "
                "statistics computed over the whole dataset is the full-sample z-score bug "
                "wearing a different hat (spec Sec 9).",
            )


# ---------------------------------------------------------------------------
# universe selection rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionConfig:
    """The rules that produce the research universe (spec Sec 11.4).

    Separate from :class:`UniverseConfig`, which is the *result*. These are the
    hand-set criteria; that is the generated, frozen list they chose.

    Every point-in-time criterion here is evaluated using only data available
    on or before the decision date.
    """

    liquidity_lookback_months: int
    min_median_turnover_inr: float
    min_history_years: float
    max_missing_frac: float
    min_price_inr: float
    max_eligible_per_date: int | None
    max_size: int | None
    rank_by: str
    tie_break: str
    require_sector: bool
    require_complete_download: bool
    exclude_quality_fatal: bool
    require_ever_eligible: bool
    decision_dates_from: str

    def __post_init__(self) -> None:
        w = "selection.yaml"
        _check(self.liquidity_lookback_months >= 1, w, "liquidity_lookback_months must be >= 1")
        _check(self.min_median_turnover_inr > 0, w, "min_median_turnover_inr must be positive")
        _check(self.min_history_years > 0, w, "min_history_years must be positive")
        _check(0 <= self.max_missing_frac < 1, w, "max_missing_frac must be in [0, 1)")
        _check(self.min_price_inr >= 0, w, "min_price_inr must be >= 0")
        if self.max_eligible_per_date is not None:
            _check(
                self.max_eligible_per_date > 1,
                w,
                "max_eligible_per_date must exceed 1 if set",
            )
        if self.max_size is not None:
            _check(self.max_size > 1, w, "max_size must exceed 1 if set")
        _check(
            self.rank_by in {"median_turnover_inr", "history_years"},
            w,
            "rank_by must be median_turnover_inr or history_years",
        )
        _check(self.tie_break == "symbol", w, "tie_break must be 'symbol' (deterministic)")
        _check(
            self.decision_dates_from == "backtest_schedule",
            w,
            "decision_dates_from must be 'backtest_schedule' so the decision dates cannot "
            "drift away from the walk-forward windows in backtest.yaml",
        )

    def lookback_sessions(self, trading_days_per_year: int = 252) -> int:
        """Trailing window in sessions."""
        return int(round(self.liquidity_lookback_months / 12.0 * trading_days_per_year))


# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the data-quality engine (spec Sec 11.2).

    Every value here decides what gets FLAGGED. None of them cause a row to be
    deleted --- the engine reports issues and a human decides.
    """

    session_quorum: float
    phantom_date_max_tickers: int
    extreme_move_pct: float
    severe_move_pct: float
    corporate_action_window_days: int
    max_gap_sessions: int
    min_coverage_over_life: float
    max_zero_volume_run: int
    zero_volume_frac_warn: float
    max_stale_price_run: int
    cross_check_n_symbols: int
    cross_check_tolerance_pct: float
    max_issues_listed_per_check: int

    def __post_init__(self) -> None:
        w = "quality.yaml"
        _check(0 < self.session_quorum <= 1, w, "session_quorum must be in (0, 1]")
        _check(self.phantom_date_max_tickers >= 1, w, "phantom_date_max_tickers must be >= 1")
        _check(0 < self.extreme_move_pct < 1, w, "extreme_move_pct must be a fraction in (0, 1)")
        _check(
            self.extreme_move_pct < self.severe_move_pct,
            w,
            f"severe_move_pct ({self.severe_move_pct}) must exceed extreme_move_pct "
            f"({self.extreme_move_pct}); otherwise the two tiers collapse into one",
        )
        _check(self.corporate_action_window_days >= 0, w, "corporate_action_window_days must be >= 0")
        _check(self.max_gap_sessions >= 1, w, "max_gap_sessions must be >= 1")
        _check(0 < self.min_coverage_over_life <= 1, w, "min_coverage_over_life must be in (0, 1]")
        _check(self.max_zero_volume_run >= 1, w, "max_zero_volume_run must be >= 1")
        _check(0 <= self.zero_volume_frac_warn < 1, w, "zero_volume_frac_warn must be in [0, 1)")
        _check(self.max_stale_price_run >= 2, w, "max_stale_price_run must be >= 2")
        _check(self.cross_check_n_symbols >= 0, w, "cross_check_n_symbols must be >= 0")
        _check(
            0 < self.cross_check_tolerance_pct < 1, w, "cross_check_tolerance_pct must be in (0, 1)"
        )
        _check(self.max_issues_listed_per_check >= 1, w, "max_issues_listed_per_check must be >= 1")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The complete, validated configuration for one run."""

    base: BaseConfig
    universe: UniverseConfig
    costs: CostsConfig
    backtest: BacktestConfig
    models: ModelsConfig
    quality: QualityConfig
    selection: SelectionConfig
    source_dir: Path = field(default_factory=default_config_dir)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict of every parameter. Deep-copied; safe to mutate."""
        return _to_plain(self)  # type: ignore[return-value]

    @property
    def hash(self) -> str:
        """Stable 12-character digest of every parameter.

        Identifies a run in the configuration log. Two runs with the same hash
        used the same parameters; two with different hashes did not. This is
        what makes the trial count behind the deflated Sharpe honest.

        ``source_dir`` is excluded --- where the YAML lived is not a parameter.
        """
        payload = self.to_dict()
        payload.pop("source_dir", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def universe_hash(self) -> str:
        """Digest of the ticker list alone, so universe drift is detectable."""
        blob = json.dumps(sorted(self.universe.tickers), separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def summary(self) -> str:
        u = self.universe
        return (
            f"Config[{self.hash}] universe={u.name}@v{u.version} "
            f"(frozen={u.frozen}, n={len(u.tickers)}, pairs={u.n_candidate_pairs}) "
            f"window={self.backtest.formation_months}/{self.backtest.trading_months}"
            f"/{self.backtest.step_months}m "
            f"z={self.backtest.signal.z_entry}/{self.backtest.signal.z_exit}"
            f"/{self.backtest.signal.z_stop} seed={self.base.seed}"
        )


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/mappings/paths to JSON-safe primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_REQUIRED_FILES = (
    "base.yaml",
    "universe.yaml",
    "costs.yaml",
    "backtest.yaml",
    "models.yaml",
    "quality.yaml",
    "selection.yaml",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing configuration file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML --- {exc}") from exc
    if data is None:
        raise ConfigError(f"{path.name}: file is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return data


def load_config(config_path: Path | str | None = None) -> Config:
    """Read, validate and freeze the configuration. Raises ``ConfigError``."""
    cdir = Path(config_path) if config_path is not None else default_config_dir()
    if not cdir.is_dir():
        raise ConfigError(f"config directory not found: {cdir}")

    raw = {name: _read_yaml(cdir / name) for name in _REQUIRED_FILES}

    base = _build(BaseConfig, raw["base.yaml"], "base.yaml")

    u = dict(raw["universe.yaml"])
    universe = _build(
        UniverseConfig,
        u,
        "universe.yaml",
        as_of=_parse_date(u["as_of"], "universe.yaml:as_of") if u.get("as_of") else None,
        start_date=_parse_date(u.get("start_date"), "universe.yaml:start_date"),
        end_date=_parse_date(u.get("end_date"), "universe.yaml:end_date"),
        tickers=tuple(u.get("tickers") or ()),
        sectors=dict(u.get("sectors") or {}),
        smoke_test_tickers=tuple(u.get("smoke_test_tickers") or ()),
        survivorship_bias=_build(
            SurvivorshipConfig, u.get("survivorship_bias"), "universe.yaml:survivorship_bias"
        ),
    )

    c = dict(raw["costs.yaml"])
    costs = _build(
        CostsConfig,
        c,
        "costs.yaml",
        rates_as_of=(
            _parse_date(c["rates_as_of"], "costs.yaml:rates_as_of") if c.get("rates_as_of") else None
        ),
        slippage_bps_per_side={
            int(k): float(v) for k, v in (c.get("slippage_bps_per_side") or {}).items()
        },
        scenario_bps_round_trip=tuple(float(v) for v in (c.get("scenario_bps_round_trip") or ())),
        short_leg=_build(ShortLegConfig, c.get("short_leg"), "costs.yaml:short_leg"),
    )

    b = dict(raw["backtest.yaml"])
    backtest = _build(
        BacktestConfig,
        b,
        "backtest.yaml",
        execution=_build(ExecutionConfig, b.get("execution"), "backtest.yaml:execution"),
        signal=_build(SignalConfig, b.get("signal"), "backtest.yaml:signal"),
        kill_switch=_build(KillSwitchConfig, b.get("kill_switch"), "backtest.yaml:kill_switch"),
        screening=_build(ScreeningConfig, b.get("screening"), "backtest.yaml:screening"),
        portfolio=_build(PortfolioConfig, b.get("portfolio"), "backtest.yaml:portfolio"),
        metrics=_build(MetricsConfig, b.get("metrics"), "backtest.yaml:metrics"),
    )

    m = dict(raw["models.yaml"])
    v = m.get("validity") or {}
    models = _build(
        ModelsConfig,
        m,
        "models.yaml",
        kalman=_build(
            KalmanConfig,
            m.get("kalman"),
            "models.yaml:kalman",
            delta_grid=tuple(float(d) for d in (m.get("kalman") or {}).get("delta_grid", ())),
        ),
        ols=_build(OLSConfig, m.get("ols"), "models.yaml:ols"),
        labels=_build(LabelConfig, m.get("labels"), "models.yaml:labels"),
        xgboost=dict(m.get("xgboost") or {}),
        filter=_build(
            FilterConfig,
            m.get("filter"),
            "models.yaml:filter",
            threshold_grid=tuple(
                float(t) for t in (m.get("filter") or {}).get("threshold_grid", ())
            ),
        ),
        lstm=dict(m.get("lstm") or {}),
        validity=ValidityConfig(
            null_control=_build(
                NullControlConfig, v.get("null_control"), "models.yaml:validity.null_control"
            ),
            deflated_sharpe=_build(
                DeflatedSharpeConfig,
                v.get("deflated_sharpe"),
                "models.yaml:validity.deflated_sharpe",
            ),
            bootstrap=_build(BootstrapConfig, v.get("bootstrap"), "models.yaml:validity.bootstrap"),
        ),
    )

    quality = _build(QualityConfig, raw["quality.yaml"], "quality.yaml")
    selection = _build(SelectionConfig, raw["selection.yaml"], "selection.yaml")

    return Config(
        base=base,
        universe=universe,
        costs=costs,
        backtest=backtest,
        models=models,
        quality=quality,
        selection=selection,
        source_dir=cdir,
    )


_CACHE: dict[str, Config] = {}


def get_config(config_path: Path | str | None = None) -> Config:
    """Cached ``load_config``. Use this everywhere except in tests."""
    key = str(config_path) if config_path is not None else "<default>"
    if key not in _CACHE:
        _CACHE[key] = load_config(config_path)
    return _CACHE[key]


def clear_config_cache() -> None:
    """Drop the cache. Used by tests that write temporary config trees."""
    _CACHE.clear()
