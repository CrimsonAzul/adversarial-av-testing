"""Uniform random search: exactly the qualification distribution, so its failure rate is the random-testing baseline."""
from __future__ import annotations

import numpy as np


class RandomSearch:
    name = "random"

    def __init__(self, dim: int, seed: int):
        self.dim, self.rng = dim, np.random.default_rng([seed, 11])

    def ask(self, n: int) -> np.ndarray:
        return self.rng.random((n, self.dim))

    def tell(self, thetas: np.ndarray, scores: np.ndarray) -> None:
        pass

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
