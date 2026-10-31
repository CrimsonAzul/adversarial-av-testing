"""Disjoint map-seed ranges. Every sampler names its split; nothing crosses a boundary.

A MetaDrive map seed fixes the procedural road geometry and background traffic
spawns. The ranges never overlap, so a qualification or RQ4 scenario can never
have been seen in training, validation or a search run.
"""
from __future__ import annotations

import numpy as np

SPLITS: dict[str, tuple[int, int]] = {
    "train": (0, 100_000),            # driver training
    "val": (100_000, 110_000),        # driver checkpoint selection
    "qual": (110_000, 120_000),       # SUT qualification panels (one fresh panel per attempt)
    "search": (120_000, 160_000),     # adversary search runs (RQ1-RQ3) and RL adversary training
    "heldout": (160_000, 200_000),    # frozen RL adversary vs random (RQ4)
}


def check(seed: int, split: str) -> int:
    low, high = SPLITS[split]
    if not low <= seed < high:
        raise ValueError(f"map seed {seed} is outside the {split} split [{low}, {high})")
    return int(seed)


def draw(rng: np.random.Generator, split: str, count: int | None = None):
    low, high = SPLITS[split]
    values = rng.integers(low, high, size=count)
    return int(values) if count is None else [int(v) for v in values]


def panel_seeds(split: str, count: int, panel_seed: int) -> list[int]:
    """Distinct seeds for a fixed panel (qualification, validation, RQ4)."""
    low, high = SPLITS[split]
    rng = np.random.default_rng(panel_seed)
    return sorted(int(v) for v in rng.choice(np.arange(low, high), size=count, replace=False))
