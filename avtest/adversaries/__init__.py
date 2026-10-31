"""Search methods over the scenario vector θ ∈ [0,1]^d, all behind one ask/tell interface.

``ask(n)`` returns an (n, d) array of proposals; ``tell(thetas, scores)`` reports the
severity score of each (higher = closer to a valid failure). Every method sees the
same score, from avtest.severity. ``state()`` / ``load(state)`` checkpoint the method
between batches so a paused run resumes exactly.
"""
from __future__ import annotations


def make(name: str, dim: int, seed: int, **kwargs):
    if name == "random":
        from .random_search import RandomSearch
        return RandomSearch(dim, seed)
    if name == "cmaes":
        from .cmaes import CMAES
        return CMAES(dim, seed, **kwargs)
    if name == "bo":
        from .bo import BayesOpt
        return BayesOpt(dim, seed, **kwargs)
    if name == "rl":
        from .rl import RLAdversary
        return RLAdversary(dim, seed, **kwargs)
    raise ValueError(f"unknown adversary {name}")


METHODS = ("random", "cmaes", "bo", "rl")
