"""CMA-ES with restarts (pycma), maximising severity inside the unit box.

Population 12 and σ₀ = 0.3 are fixed in docs/DESIGN.md. A restart from a new
random mean happens when pycma signals stop, when σ falls below 0.02, or after
15 generations without a better best score. Proposals are served from the
current generation; ``tell`` waits until the whole generation is scored.
"""
from __future__ import annotations

import pickle

import cma
import numpy as np


class CMAES:
    name = "cmaes"

    def __init__(self, dim: int, seed: int, popsize: int = 12, sigma0: float = 0.3):
        self.dim, self.popsize, self.sigma0 = dim, popsize, sigma0
        self.rng = np.random.default_rng([seed, 12])
        self.es = None
        self.queue: list[np.ndarray] = []          # asked, not yet handed out
        self.pending: list[np.ndarray] = []        # handed out, awaiting scores
        self.scored: list[tuple[np.ndarray, float]] = []
        self.best, self.stale, self.restarts = -np.inf, 0, 0

    def _new_es(self) -> None:
        self.es = cma.CMAEvolutionStrategy(self.rng.random(self.dim), self.sigma0,
                                           {"bounds": [0, 1], "popsize": self.popsize, "verbose": -9,
                                            "seed": int(self.rng.integers(1, 2**31 - 1))})
        self.best, self.stale = -np.inf, 0

    def ask(self, n: int) -> np.ndarray:
        out = []
        while len(out) < n:
            if not self.queue:
                if self.pending or self.scored:      # a generation is still open
                    break
                if self.es is None or self.es.stop() or self.es.sigma < 0.02 or self.stale >= 15:
                    if self.es is not None:
                        self.restarts += 1
                    self._new_es()
                self.queue = [np.asarray(x, dtype=float) for x in self.es.ask()]
            x = self.queue.pop(0)
            self.pending.append(x)
            out.append(np.clip(x, 0, 1))
        return np.asarray(out).reshape(len(out), self.dim)

    def tell(self, thetas: np.ndarray, scores: np.ndarray) -> None:
        for _theta, score in zip(thetas, scores):
            self.scored.append((self.pending.pop(0), float(score)))
        if not self.queue and not self.pending and len(self.scored) == self.popsize:
            xs, fs = zip(*self.scored)
            self.es.tell(list(xs), [-f for f in fs])
            top = max(fs)
            self.stale = 0 if top > self.best + 1e-9 else self.stale + 1
            self.best = max(self.best, top)
            self.scored = []

    def state(self) -> dict:
        return {"pickle": pickle.dumps(self.__dict__)}

    def load(self, state: dict) -> None:
        self.__dict__.update(pickle.loads(state["pickle"]))
