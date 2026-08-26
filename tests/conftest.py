"""Shared pytest fixtures.

SYNTHETIC DATA NOTICE
---------------------
Every price-like series produced in this file is SYNTHETIC and exists for
METHODOLOGY / UNIT-TEST purposes only. It is generated from a known process so
that estimators can be checked against a ground truth they cannot see.
Nothing produced here may appear in any results table, figure or report as a
trading result.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa import paths
from dsa.config import Config, clear_config_cache, load_config
from dsa.logging_utils import reset_logging
from dsa.seeds import spawn_rng

# A seed used only by tests, distinct from the project seed so that a test
# passing never implies anything about a particular research run.
TEST_SEED = 12345


@pytest.fixture(scope="session")
def project_root() -> Path:
    return paths.project_root()


@pytest.fixture(scope="session")
def repo_config_dir(project_root: Path) -> Path:
    return project_root / "config"


@pytest.fixture(scope="session")
def cfg(repo_config_dir: Path) -> Config:
    """The project's real, validated configuration."""
    return load_config(repo_config_dir)


@pytest.fixture
def tmp_config_dir(tmp_path: Path, repo_config_dir: Path) -> Path:
    """A writable copy of ``config/``, for tests that mutate settings."""
    target = tmp_path / "config"
    shutil.copytree(repo_config_dir, target)
    clear_config_cache()
    yield target
    clear_config_cache()


@pytest.fixture(autouse=True)
def _clean_logging():
    """Keep logging handlers from leaking between tests."""
    reset_logging()
    yield
    reset_logging()


# ---------------------------------------------------------------------------
# SYNTHETIC processes with known parameters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trading_index() -> pd.DatetimeIndex:
    """A business-day index of ~4 years. SYNTHETIC — unit test only."""
    return pd.bdate_range("2018-01-01", periods=1000, name="date")


@pytest.fixture
def ou_spread(trading_index: pd.DatetimeIndex) -> pd.Series:
    """An Ornstein-Uhlenbeck spread with a KNOWN half-life of 20 days.

    SYNTHETIC DATA --- METHODOLOGY/UNIT TEST ONLY.

    Discrete AR(1) form of dS = theta(mu - S)dt + sigma dW:

        S[t] = mu + phi * (S[t-1] - mu) + sigma_eps * eps[t]

    with ``phi = exp(-ln(2) / half_life)``, so that the half-life recovered by
    the AR(1) fit in ``dsa.coint.halflife`` has a ground truth to be checked
    against on Day 4-6.

    ESTIMATOR TOLERANCE --- read before writing that test. The half-life is a
    non-linear transform of an AR(1) coefficient, and it is a noisy one. A
    200-draw Monte Carlo at this exact setting (true half-life 20, n = 1000)
    gives mean 19.4 with a standard deviation of 4.6 days and a 5-95% range of
    roughly [12.5, 28.4]. This particular seed recovers about 22.3.

    So a Day 4-6 test must assert a BAND, not a point. Asserting
    ``half_life == 20 +/- 1`` would fail on a correct estimator, get loosened
    until it passed, and teach nothing. Asserting the estimate lands inside
    [12, 29] --- and that the random-walk fixture does not --- is the test that
    actually distinguishes a working estimator from a broken one.
    """
    n = len(trading_index)
    half_life = 20.0
    phi = float(np.exp(-np.log(2.0) / half_life))
    mu, sigma_eps = 0.0, 1.0

    rng = spawn_rng(TEST_SEED, "ou_spread")
    eps = rng.standard_normal(n)
    s = np.empty(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = mu + phi * (s[t - 1] - mu) + sigma_eps * eps[t]
    return pd.Series(s, index=trading_index, name="spread")


@pytest.fixture
def random_walk(trading_index: pd.DatetimeIndex) -> pd.Series:
    """A pure I(1) random walk. SYNTHETIC — unit test only.

    Used as the negative control: nothing in this series is mean-reverting, so
    any estimator that reports a tradeable half-life on it is broken.
    """
    rng = spawn_rng(TEST_SEED, "random_walk")
    return pd.Series(
        100.0 + np.cumsum(rng.standard_normal(len(trading_index))),
        index=trading_index,
        name="price",
    )


@pytest.fixture
def cointegrated_pair(trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Two price series that ARE cointegrated, with a KNOWN beta of 1.5.

    SYNTHETIC DATA --- METHODOLOGY/UNIT TEST ONLY.

        P_B is a random walk (I(1))
        P_A = beta * P_B + stationary OU noise      => A - beta*B is I(0)

    On Day 4-6 the screen must find this pair, and must NOT find the
    ``independent_pair`` fixture below. Those are the two ends of the test.

    Measured properties of this exact seed (verified, not assumed):
      OLS beta            1.4437  (generating value 1.5)
      spread half-life    22.4 days from the OLS spread, 22.5 from the true
                          residual --- so the beta-estimation effect is small
                          here and the gap from the generating value of 15 is
                          sampling error in the residual draw, about 1.6 sigma.
      spread range        [-15.4, 28.6] --- bounded, i.e. stationary
    Same tolerance warning as ``ou_spread``: assert bands, not points.
    """
    n = len(trading_index)
    beta_true = 1.5
    rng = spawn_rng(TEST_SEED, "cointegrated_pair")

    pb = 100.0 + np.cumsum(rng.standard_normal(n) * 1.0)

    half_life = 15.0
    phi = float(np.exp(-np.log(2.0) / half_life))
    resid = np.empty(n)
    resid[0] = 0.0
    eps = rng.standard_normal(n) * 2.0
    for t in range(1, n):
        resid[t] = phi * resid[t - 1] + eps[t]

    pa = beta_true * pb + resid
    return pd.DataFrame({"A": pa, "B": pb}, index=trading_index)


@pytest.fixture
def independent_pair(trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Two INDEPENDENT random walks. SYNTHETIC — unit test only.

    The spurious-regression control (spec Sec 2.1): these will often show
    apparent correlation in levels and must still be rejected by a correct
    cointegration test.

    Measured properties of this exact seed (verified, not assumed):
      level correlation   +0.174
      residual half-life  229 days --- far outside the 5-60 day tradeable band
      residual range      [55.2, 102.0] --- wide and drifting, i.e. not stationary
    """
    n = len(trading_index)
    rng = spawn_rng(TEST_SEED, "independent_pair")
    return pd.DataFrame(
        {
            "A": 100.0 + np.cumsum(rng.standard_normal(n)),
            "B": 250.0 + np.cumsum(rng.standard_normal(n) * 2.0),
        },
        index=trading_index,
    )
