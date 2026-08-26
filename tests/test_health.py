"""Project health check.

The question this file answers: *is the skeleton intact?* It runs first and
fastest, so that when something further down the suite breaks, this file has
already told you whether the cause is the project or the environment.

It also encodes the Day-1 state honestly. Several things are deliberately not
done yet --- the universe is empty, the cost rates are unverified, the short-leg
method is undecided --- and the tests below assert that those gaps are
*declared* rather than silently defaulted. A pending decision that is visible
is a task; a pending decision that is invisible is a bug in the results.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

from dsa import paths
from dsa.config import Config
from dsa.logging_utils import get_logger, reset_logging, setup_logging
from dsa.provenance import capture_run_context, git_state, package_versions

# Needed for Day 1 itself. Their absence stops everything.
DAY1_PACKAGES = ["numpy", "pandas", "yaml", "pytest"]

# Declared in pyproject and needed from Day 2 onward. Their absence is
# reported, not fatal, so that Day 1 can be completed and verified first.
PIPELINE_PACKAGES = {
    "statsmodels": "cointegration tests, OLS, BH FDR (Sec 2, 4)",
    "sklearn": "metrics and preprocessing for the ML filter (Sec 8)",
    "xgboost": "the signal filter (Sec 8)",
    "lifelines": "Kaplan-Meier and Cox for the survival curve (Sec 7)",
    "arch": "stationary block bootstrap (Sec 10.3)",
    "pyarrow": "parquet cache for the price panel (Sec 11)",
    "matplotlib": "figures",
    "yfinance": "NSE daily adjusted OHLCV (Sec 11.1)",
}


# ===========================================================================
# Environment
# ===========================================================================


def test_python_version_is_supported():
    assert sys.version_info >= (3, 10), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} is below the 3.10 "
        "required by pyproject.toml (the code uses PEP 604 `X | None` annotations)"
    )


@pytest.mark.parametrize("module", DAY1_PACKAGES)
def test_day1_dependencies_are_installed(module):
    importlib.import_module(module)


def test_pipeline_dependencies_are_installed():
    """Day 2+ dependencies. Skips with an actionable message if absent."""
    missing = {}
    for module, why in PIPELINE_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing[module] = why
    if missing:
        lines = "\n".join(f"    {m:14s} {why}" for m, why in missing.items())
        pytest.skip(
            f"{len(missing)} pipeline dependency/ies not installed yet:\n{lines}\n"
            '    Install with:  pip install -e ".[dev]"'
        )


# ===========================================================================
# Layout
# ===========================================================================


def test_every_managed_directory_exists():
    missing = [name for name, path in paths.project_dirs().items() if not path.is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_spec_repository_structure_is_present(project_root: Path):
    """The file tree of spec Sec 12."""
    for relative in [
        "src/dsa",
        "src/dsa/data",
        "src/dsa/coint",
        "src/dsa/spread",
        "src/dsa/signal",
        "src/dsa/filter",
        "src/dsa/backtest",
        "src/dsa/validity",
        "tests",
        "config",
        "data/raw",
        "data/clean",
        "notebooks",
        "models",
        "reports",
        "figures",
        "app",
        "scripts",
    ]:
        assert (project_root / relative).is_dir(), f"missing {relative}/"


def test_project_files_are_present(project_root: Path):
    for filename in ["pyproject.toml", "requirements.txt", "README.md", ".gitignore", "Makefile"]:
        assert (project_root / filename).is_file(), f"missing {filename}"


def test_every_subpackage_imports():
    for name in [
        "dsa",
        "dsa.paths",
        "dsa.config",
        "dsa.seeds",
        "dsa.logging_utils",
        "dsa.provenance",
        "dsa.data",
        "dsa.coint",
        "dsa.spread",
        "dsa.signal",
        "dsa.filter",
        "dsa.backtest",
        "dsa.validity",
        "dsa.validity.leakage",
        "dsa.utils",
    ]:
        importlib.import_module(name)


def test_signal_subpackage_does_not_shadow_the_stdlib():
    """``dsa.signal`` matches the spec's tree; ``import signal`` must still work."""
    import signal as stdlib_signal

    assert hasattr(stdlib_signal, "SIGINT")


def test_importing_dsa_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    """Importing the package must not create directories anywhere."""
    monkeypatch.setenv("DSA_PROJECT_ROOT", str(tmp_path))
    importlib.reload(importlib.import_module("dsa.paths"))
    assert list(tmp_path.iterdir()) == [], f"import created {list(tmp_path.iterdir())}"


# ===========================================================================
# Secret hygiene
# ===========================================================================


def test_gitignore_covers_secrets_and_data(project_root: Path):
    text = (project_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", "data/**", "models/**", "*.key", "__pycache__/", ".venv/"]:
        assert pattern in text, f".gitignore does not cover {pattern!r}"


def test_no_dotenv_file_is_committed(project_root: Path):
    """`.env.example` is tracked; `.env` must never be."""
    assert (project_root / ".env.example").is_file(), "missing .env.example"
    env = project_root / ".env"
    if env.is_file():
        assert ".env" in (project_root / ".gitignore").read_text(encoding="utf-8")


def test_no_credentials_in_config(project_root: Path):
    """Config is tracked in git, so nothing secret may live there."""
    banned = ("api_key", "apikey", "secret", "password", "token", "passwd")
    for path in (project_root / "config").glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            # Allow the word inside a comment line (documentation), never as a key.
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert word not in stripped, f"{path.name}: '{word}' appears in {line!r}"


# ===========================================================================
# Configuration and logging wiring
# ===========================================================================


def test_config_loads_and_hashes(cfg: Config):
    assert len(cfg.hash) == 12
    assert cfg.base.seed > 0


def test_logging_is_idempotent(tmp_path):
    """Notebooks re-run cells; handlers must not accumulate."""
    reset_logging()
    setup_logging("INFO", log_to_file=True, log_dir=tmp_path)
    first = len(logging.getLogger("dsa").handlers)
    setup_logging("INFO", log_to_file=True, log_dir=tmp_path)
    assert len(logging.getLogger("dsa").handlers) == first


def test_logging_writes_to_file(tmp_path):
    reset_logging()
    setup_logging("DEBUG", log_to_file=True, log_dir=tmp_path, force=True)
    get_logger(__name__).info("health check probe")
    for handler in logging.getLogger("dsa").handlers:
        handler.flush()
    log_file = tmp_path / "dsa.log"
    assert log_file.is_file()
    assert "health check probe" in log_file.read_text(encoding="utf-8")


def test_logger_names_are_normalised_under_dsa():
    assert get_logger("dsa.coint.screen").name == "dsa.coint.screen"
    assert get_logger("some.module").name == "dsa.some.module"
    assert get_logger("__main__").name == "dsa.main"


# ===========================================================================
# Provenance
# ===========================================================================


def test_git_state_handles_a_non_repository():
    """Day 1: this is not a git repo yet, and that must not crash anything."""
    state = git_state()
    assert set(state) >= {"is_repo", "commit", "dirty"}
    if not state["is_repo"]:
        assert state["commit"] is None


def test_package_versions_reports_missing_packages():
    versions = package_versions()
    assert "numpy" in versions
    assert all(isinstance(v, str) for v in versions.values())


def test_run_context_captures_the_config(cfg: Config, tmp_path):
    ctx = capture_run_context(cfg)
    assert ctx.config_hash == cfg.hash
    assert ctx.seed == cfg.base.seed
    assert ctx.universe_frozen is cfg.universe.frozen
    written = ctx.write(tmp_path / "ctx.json")
    assert written.is_file()
    assert cfg.hash in written.read_text(encoding="utf-8")


def test_run_context_works_without_a_config():
    ctx = capture_run_context()
    assert ctx.config_hash is None
    assert ctx.run_id


# ===========================================================================
# Day-1 pending decisions --- declared, not defaulted
# ===========================================================================


def test_universe_is_declared_unfrozen_until_built(cfg: Config):
    """Spec Sec 11.4. An empty list is fine; an empty list claiming to be
    frozen is not, and the config layer rejects that (see test_config.py)."""
    if not cfg.universe.is_populated:
        assert cfg.universe.frozen is False
        assert cfg.universe.as_of is None
        assert cfg.universe.version == 0


def test_cost_rates_are_flagged_unverified_until_checked(cfg: Config):
    """Spec Sec 6.2. STT and exchange charges change; the config must say
    whether the numbers in it were ever checked against a primary source."""
    if not cfg.costs.verified:
        assert cfg.costs.rates_as_of is None, (
            "costs.yaml carries a rates_as_of date but verified is still false. "
            "Either the rates were checked (set verified: true) or the date is "
            "decoration."
        )


def test_short_leg_decision_is_pending_not_assumed(cfg: Config):
    """Spec Sec 6.1: 'address this explicitly or your backtest is fiction'."""
    if not cfg.costs.short_leg.decided:
        assert cfg.costs.short_leg.method is None


def test_survivorship_bias_is_acknowledged(cfg: Config):
    """Spec Sec 11.3: state the bias and its direction."""
    assert cfg.universe.survivorship_bias.acknowledged is True
    assert cfg.universe.survivorship_bias.direction == "optimistic"


# Artifacts the pipeline knows how to regenerate. One entry per stage, added
# on the day that stage lands. Anything in reports/ or figures/ that is not
# here was not produced by a run.
GENERATED_ARTIFACTS = {
    "runs",  # provenance JSON, written by dsa.provenance
    "universe_selection.csv",  # Day 2, scripts/build_dataset.py --stage universe
    "data_quality",  # Day 3, scripts/run_quality_report.py
}


def test_every_result_artefact_is_one_the_pipeline_produces(project_root: Path):
    """Guard against a number that no run produced.

    The failure mode this exists to catch is a chart or table dropped into
    reports/ or figures/ by hand --- an equity curve exported from a notebook
    and never regenerated, a metric typed into a CSV. Spec Sec 12: figures are
    regenerated by make, never by hand.

    The allowlist grows one entry per stage as stages land, which makes adding
    an unexplained artefact a deliberate act rather than an accident.
    """
    for directory in ("reports", "figures"):
        unexpected = [
            p.name
            for p in (project_root / directory).iterdir()
            if p.name != ".gitkeep" and p.name not in GENERATED_ARTIFACTS
        ]
        assert not unexpected, (
            f"{directory}/ contains {unexpected}, which no pipeline stage claims to "
            "generate. Either wire it into a script and add it to "
            "GENERATED_ARTIFACTS, or delete it. Every number in the report must "
            "originate from an actual experiment."
        )


def test_no_backtest_metrics_are_reported_before_the_backtest_exists(project_root: Path):
    """The README's result slots must stay unmeasured until a run fills them.

    Checked structurally rather than by reading numbers: while
    src/dsa/backtest/engine.py does not exist, no Sharpe can have been
    computed, so the README must still say so.
    """
    if (project_root / "src" / "dsa" / "backtest" / "engine.py").exists():
        pytest.skip("backtest engine exists; this guard no longer applies")

    readme = (project_root / "README.md").read_text(encoding="utf-8")
    assert "NOT YET MEASURED" in readme, (
        "the README no longer marks its result slots as unmeasured, but no "
        "backtest engine exists to have measured them."
    )
