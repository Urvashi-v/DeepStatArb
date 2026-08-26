"""Run provenance: what produced this number, with what, and when.

Spec Sec 12 requires that "every backtest run appends to a configuration log
automatically --- parameters, universe hash, date range, and resulting Sharpe",
because the deflated Sharpe ratio (Sec 10.2) needs an honest count of trials
and "you will not remember".

This module captures the environment half of that record. The results half is
appended by the backtest engine when it lands (Day 10-12). Keeping the capture
here means every stage of the pipeline stamps its output the same way.

The git helpers are strictly read-only --- ``rev-parse`` and ``status``. This
module never commits, never pushes, and never touches history.
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dsa.paths import project_root, reports_dir

__all__ = ["RunContext", "capture_run_context", "git_state", "package_versions", "new_run_id"]

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "xgboost",
    "lifelines",
    "arch",
    "pyarrow",
    "yfinance",
    "torch",
)


def _run_git(*args: str) -> str | None:
    """Run a read-only git command; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Current commit and whether the working tree is dirty.

    A result produced from a dirty tree cannot be reproduced from any commit,
    so this flag belongs next to every number in the report.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return {"is_repo": False, "commit": None, "dirty": None, "branch": None}
    status = _run_git("status", "--porcelain")
    return {
        "is_repo": True,
        "commit": commit,
        "short_commit": commit[:12],
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "n_dirty_files": len(status.splitlines()) if status else 0,
    }


def package_versions() -> dict[str, str]:
    """Installed version of each package the results depend on."""
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def new_run_id(config_hash: str | None = None) -> str:
    """A sortable run identifier, e.g. ``20260823T211500Z-79e89d228fe0``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash}" if config_hash else stamp


@dataclass(frozen=True)
class RunContext:
    """Everything needed to explain where a number came from."""

    run_id: str
    started_at: str
    config_hash: str | None
    universe_hash: str | None
    universe_version: int | None
    universe_frozen: bool | None
    seed: int | None
    git: dict[str, Any] = field(default_factory=dict)
    python: str = ""
    platform_: str = ""
    hostname: str = ""
    user: str = ""
    packages: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def write(self, path: Path | None = None) -> Path:
        """Persist to ``reports/runs/<run_id>.json`` (or ``path``)."""
        target = path or (reports_dir() / "runs" / f"{self.run_id}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def summary(self) -> str:
        g = self.git
        git_bit = (
            f"git={g.get('short_commit')}{'+dirty' if g.get('dirty') else ''}"
            if g.get("is_repo")
            else "git=<not a repo>"
        )
        return (
            f"run {self.run_id} | config={self.config_hash} | universe={self.universe_hash}"
            f"@v{self.universe_version} | seed={self.seed} | {git_bit} | py{self.python}"
        )


def capture_run_context(cfg: Any | None = None, run_id: str | None = None) -> RunContext:
    """Snapshot the environment for the current run.

    ``cfg`` is a :class:`dsa.config.Config` when one is available; it is
    optional so that scripts with no config (the environment health check) can
    still produce a context.
    """
    config_hash = getattr(cfg, "hash", None) if cfg is not None else None
    universe = getattr(cfg, "universe", None) if cfg is not None else None
    base = getattr(cfg, "base", None) if cfg is not None else None

    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser can fail in bare containers
        user = "unknown"

    return RunContext(
        run_id=run_id or new_run_id(config_hash),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_hash=config_hash,
        universe_hash=getattr(cfg, "universe_hash", None) if cfg is not None else None,
        universe_version=getattr(universe, "version", None),
        universe_frozen=getattr(universe, "frozen", None),
        seed=getattr(base, "seed", None),
        git=git_state(),
        python=platform.python_version(),
        platform_=f"{platform.system()} {platform.release()} ({platform.machine()})",
        hostname=socket.gethostname(),
        user=user,
        packages=package_versions(),
    )


if __name__ == "__main__":  # pragma: no cover - diagnostic entry point
    print(capture_run_context().to_json())
    sys.exit(0)
