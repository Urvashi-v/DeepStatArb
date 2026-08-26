"""Reproducible randomness.

``np.random.seed(42)`` at the top of a script is not enough for this project,
for two reasons.

**It is a single global stream.** The random-pair null control (spec Sec 10.1)
runs the whole pipeline several hundred times on randomly assigned pairs. If
every one of those runs draws from one global stream, then adding a step
anywhere upstream --- an extra shuffle, a different iteration order, a
parallel worker --- silently changes every run downstream of it, and the result
stops being reproducible even though a seed was "set".

**It couples unrelated components.** With a single stream, the numbers the
bootstrap draws depend on how many numbers the null control happened to draw
first. Two components that have nothing to do with each other become entangled.

The fix is one master seed plus *named, independently derived* streams:

    rng = spawn_rng(cfg.base.seed, "null_control", run=17)

Each named stream is a pure function of ``(master_seed, labels)``. Streams for
different names are statistically independent. Adding a new named stream does
not perturb any existing one --- which is the property ``SeedSequence.spawn()``
called sequentially does *not* give you, because there the n-th child depends
on how many children were requested before it.

``set_global_seed`` still exists, because third-party libraries (scikit-learn's
internals, XGBoost, PyTorch) read global state. Use it once at process start,
and use ``spawn_rng`` for anything of our own that draws random numbers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

__all__ = [
    "set_global_seed",
    "spawn_rng",
    "seed_sequence",
    "temporary_seed",
    "seed_report",
]

_log = logging.getLogger(__name__)


def _label_entropy(label: Any) -> int:
    """Stable 64-bit integer derived from a label.

    ``hash()`` is deliberately randomised per process in Python 3 unless
    ``PYTHONHASHSEED`` is fixed, so using it here would make streams differ
    between runs. BLAKE2b is stable across processes, machines and versions.
    """
    digest = hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def seed_sequence(master_seed: int, *labels: Any, **kwlabels: Any) -> np.random.SeedSequence:
    """A ``SeedSequence`` uniquely determined by the master seed and labels.

    Keyword labels are sorted by name so that call-site ordering is irrelevant:
    ``spawn_rng(s, "boot", pair="A_B", window=3)`` and
    ``spawn_rng(s, "boot", window=3, pair="A_B")`` give the same stream.
    """
    if not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError(f"master_seed must be a non-negative int, got {master_seed!r}")
    parts = [int(master_seed)]
    parts.extend(_label_entropy(lbl) for lbl in labels)
    parts.extend(_label_entropy(f"{k}={v}") for k, v in sorted(kwlabels.items()))
    return np.random.SeedSequence(parts)


def spawn_rng(master_seed: int, *labels: Any, **kwlabels: Any) -> np.random.Generator:
    """An independent, reproducible ``Generator`` for a named component.

    Examples
    --------
    >>> a = spawn_rng(20260823, "null_control", run=0)
    >>> b = spawn_rng(20260823, "null_control", run=1)
    >>> c = spawn_rng(20260823, "null_control", run=0)
    >>> bool((a.random(5) == c.random(5)).all())   # same name -> same stream
    True
    """
    return np.random.default_rng(seed_sequence(master_seed, *labels, **kwlabels))


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed every global RNG a third-party library might reach for.

    Call once, at process start. Returns a record of what was seeded, for the
    run log.

    Parameters
    ----------
    seed
        The master seed, normally ``cfg.base.seed``.
    deterministic_torch
        Also force deterministic cuDNN/algorithm selection in PyTorch. Slower,
        and only relevant to the LSTM ablation (spec Sec 9), so off by default.

    Caveat worth knowing
    --------------------
    This does *not* make everything deterministic. Anything that reduces
    floating-point values in a non-fixed order --- multi-threaded BLAS,
    ``n_jobs=-1`` in scikit-learn, XGBoost's ``hist`` method across differing
    thread counts --- can still produce bitwise-different results from an
    identical seed. Those differences are tiny, but if a conclusion flips
    because of one, the conclusion was never real.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    seed = int(seed)

    record: dict[str, Any] = {"seed": seed, "seeded": []}

    # Affects hash ordering of sets/dicts of str in *child* processes only;
    # setting it here is documentation as much as effect.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    record["seeded"].append("PYTHONHASHSEED")

    random.seed(seed)
    record["seeded"].append("random")

    # Legacy global numpy state. Modern code should use spawn_rng instead, but
    # pandas' .sample() and some sklearn paths still consult this.
    np.random.seed(seed % (2**32))
    record["seeded"].append("numpy.random(legacy global)")

    try:
        import torch  # noqa: PLC0415  (optional dependency, imported lazily)
    except ImportError:
        record["torch"] = "not installed"
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        record["seeded"].append("torch")
        record["torch"] = torch.__version__
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            record["seeded"].append("torch(deterministic algorithms)")

    _log.debug("global seed set: %s", record)
    return record


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Seed the global RNGs inside the block and restore state afterwards.

    For tests and for the rare third-party call that insists on global state.
    Prefer ``spawn_rng`` in library code.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(int(seed) % (2**32))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def seed_report(master_seed: int, labels: list[str]) -> dict[str, int]:
    """First draw from each named stream. A cheap fingerprint for the run log.

    If two runs claim the same seed but disagree here, the seeding discipline
    was broken somewhere between them.
    """
    return {lbl: int(spawn_rng(master_seed, lbl).integers(0, 2**31)) for lbl in labels}
