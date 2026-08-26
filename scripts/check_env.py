#!/usr/bin/env python
"""Environment and project health check.

Run this first when something looks wrong, and after any environment change:

    python scripts/check_env.py

Reports the Python version, every declared dependency, the project layout, the
configuration hash, the seeded-stream fingerprint, and the pending decisions
that have not been made yet. Exits non-zero only if something needed *right
now* is broken --- a missing Day-2 dependency is reported, not fatal.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# Make `src/` importable when run directly from a clone without installing.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dsa import paths  # noqa: E402
from dsa.config import ConfigError, load_config  # noqa: E402
from dsa.provenance import capture_run_context, package_versions  # noqa: E402
from dsa.seeds import seed_report  # noqa: E402

OK = "  ok  "
WARN = " warn "
FAIL = " FAIL "

REQUIRED_NOW = ["numpy", "pandas", "yaml", "pytest"]
REQUIRED_LATER = {
    "statsmodels": "Day 4-6   cointegration, OLS, BH FDR",
    "sklearn": "Day 19-22 ML filter metrics",
    "xgboost": "Day 19-22 the signal filter",
    "lifelines": "Day 16-18 Kaplan-Meier, Cox",
    "arch": "Day 26-28 block bootstrap",
    "pyarrow": "Day 2-3   parquet price cache",
    "matplotlib": "Day 7+    figures",
    "yfinance": "Day 2-3   NSE price download",
}
OPTIONAL = {
    "torch": "Day 23-25 LSTM ablation (optional arm)",
    "streamlit": "Day 29-30 dashboard",
}


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<24} {detail}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    failures = 0
    warnings = 0

    print("=" * 72)
    print("DeepStatArb --- environment health check")
    print("=" * 72)

    section("Interpreter")
    version = platform.python_version()
    # UP036: ruff knows requires-python is >=3.10 so this looks dead. It is
    # not --- this script's whole job is to diagnose a broken environment,
    # including one where someone ran it on an older interpreter.
    if sys.version_info >= (3, 10):  # noqa: UP036
        line(OK, "python", f"{version} ({sys.executable})")
    else:
        line(FAIL, "python", f"{version} --- 3.10 or newer required")
        failures += 1
    line(OK, "platform", f"{platform.system()} {platform.release()} ({platform.machine()})")

    section("Dependencies required now")
    for module in REQUIRED_NOW:
        try:
            importlib.import_module(module)
            line(OK, module)
        except ImportError:
            line(FAIL, module, "not installed")
            failures += 1

    section("Dependencies required from Day 2")
    versions = package_versions()
    for module, why in REQUIRED_LATER.items():
        try:
            importlib.import_module(module)
            dist = {"sklearn": "scikit-learn"}.get(module, module)
            line(OK, module, versions.get(dist, ""))
        except ImportError:
            line(WARN, module, f"MISSING --- {why}")
            warnings += 1

    section("Optional")
    for module, why in OPTIONAL.items():
        try:
            importlib.import_module(module)
            line(OK, module, versions.get(module, ""))
        except ImportError:
            line(WARN, module, f"not installed --- {why}")

    section("Project layout")
    for name, path in paths.project_dirs().items():
        if path.is_dir():
            line(OK, name, str(path))
        else:
            line(FAIL, name, f"missing: {path}")
            failures += 1

    section("Configuration")
    cfg = None
    try:
        cfg = load_config()
        line(OK, "config loads", cfg.summary())
        line(OK, "config hash", cfg.hash)
        line(OK, "universe hash", cfg.universe_hash)
    except ConfigError as exc:
        line(FAIL, "config", str(exc))
        failures += 1

    if cfg is not None:
        section("Reproducibility")
        line(OK, "master seed", str(cfg.base.seed))
        for label, draw in seed_report(
            cfg.base.seed, ["screening", "null_control", "bootstrap"]
        ).items():
            line(OK, f"stream:{label}", str(draw))

        section("Pending decisions")
        if not cfg.universe.is_populated:
            line(WARN, "universe", "EMPTY --- build and freeze it on Day 2 (spec Sec 11.4)")
            warnings += 1
        elif not cfg.universe.frozen:
            line(WARN, "universe", f"{len(cfg.universe.tickers)} tickers, NOT frozen")
            warnings += 1
        else:
            line(
                OK,
                "universe",
                f"{len(cfg.universe.tickers)} tickers frozen at v{cfg.universe.version}, "
                f"{cfg.universe.n_candidate_pairs} candidate pairs",
            )

        if not cfg.costs.verified:
            line(WARN, "cost rates", "UNVERIFIED --- check NSE/SEBI schedules (spec Sec 6.2)")
            warnings += 1
        else:
            line(OK, "cost rates", f"verified as of {cfg.costs.rates_as_of}")

        if not cfg.costs.short_leg.decided:
            line(WARN, "short leg", "UNDECIDED --- futures / intraday / caveat (spec Sec 6.1)")
            warnings += 1
        else:
            line(OK, "short leg", str(cfg.costs.short_leg.method))

    section("Provenance")
    ctx = capture_run_context(cfg)
    git = ctx.git
    if git.get("is_repo"):
        state = "dirty" if git.get("dirty") else "clean"
        line(OK, "git", f"{git.get('short_commit')} on {git.get('branch')} ({state})")
    else:
        line(WARN, "git", "not a git repository yet --- `git init` when ready")
    line(OK, "run id", ctx.run_id)

    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: {failures} failure(s), {warnings} warning(s). Fix the failures first.")
    elif warnings:
        print(f"RESULT: healthy, with {warnings} warning(s) --- see 'Pending decisions' above.")
    else:
        print("RESULT: healthy.")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
