"""Statistics fixed before any results exist (see docs/DESIGN.md, Statistics).

Rows are evaluation records in evaluation order. A row is a *valid failure* when
``row["valid_failure"]`` is true. Uniqueness is by *failure cell*: the scenario's
parameter-grid cell (``row["cell"]``, from avtest.scenario.cell) plus the failure
type. Repeatedly hitting one exploit therefore counts once.
"""
from __future__ import annotations

import math

import numpy as np

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026


def failure_key(row: dict) -> tuple:
    return (*row["cell"], row["outcome"])


def discovery_curve(rows: list[dict]) -> np.ndarray:
    """Number of unique valid failure cells after each evaluation."""
    seen, curve = set(), []
    for row in rows:
        if row["valid_failure"]:
            seen.add(failure_key(row))
        curve.append(len(seen))
    return np.asarray(curve, dtype=float)


def unique_failures(rows: list[dict], budget: int | None = None) -> int:
    rows = rows if budget is None else rows[:budget]
    return len({failure_key(r) for r in rows if r["valid_failure"]})


def auc(rows: list[dict], budget: int) -> float:
    """Mean of the discovery curve over the first ``budget`` evaluations (0 if none yet)."""
    curve = discovery_curve(rows[:budget])
    if curve.size < budget:                       # unfinished run: carry the last value, never extrapolate up
        curve = np.concatenate([curve, np.full(budget - curve.size, curve[-1] if curve.size else 0.0)])
    return float(curve.mean())


def first_failure(rows: list[dict]) -> int | None:
    """1-based evaluation index of the first valid failure; None = right-censored."""
    return next((i + 1 for i, r in enumerate(rows) if r["valid_failure"]), None)


def invalid_fraction(rows: list[dict]) -> float:
    """Among SUT failures, the share no reference driver could avoid (RQ2)."""
    failures = [r for r in rows if r["sut_failed"]]
    return float("nan") if not failures else sum(not r["avoidable"] for r in failures) / len(failures)


def cell_entropy(rows: list[dict]) -> float:
    """Shannon entropy (bits) of valid failures over failure cells (RQ3)."""
    counts: dict[tuple, int] = {}
    for r in rows:
        if r["valid_failure"]:
            counts[failure_key(r)] = counts.get(failure_key(r), 0) + 1
    total = sum(counts.values())
    return 0.0 if not total else -sum(c / total * math.log2(c / total) for c in counts.values())


def bootstrap_mean(values, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling independent units (search seeds)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
