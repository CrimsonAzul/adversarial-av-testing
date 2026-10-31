"""Bayesian optimisation: GP (Matérn 5/2 + noise) with expected improvement.

Fixed settings (docs/DESIGN.md): 20 uniform initial points; kernel hyperparameters are
re-optimised (marginal likelihood, 2 restarts) every 10 new evaluations and kept fixed in
between; the GP is refit on at most 400 points (the 200 best plus the 200 most recent, so
cost stays bounded over a 1,000-evaluation run) before every proposal; EI is maximised over
4,000 uniform candidates plus 1,000 Gaussian perturbations (σ = 0.05) of the 10 best points;
batches use the kriging believer (each chosen point is added with its predicted mean before
choosing the next).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class BayesOpt:
    name = "bo"
    REOPTIMISE_EVERY = 10

    def __init__(self, dim: int, seed: int, initial: int = 20, max_points: int = 400):
        self.dim, self.initial, self.max_points = dim, initial, max_points
        self.rng = np.random.default_rng([seed, 13])
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.kernel = None                  # last optimised kernel, hyperparameters included
        self.optimised_at = None            # evaluation count when it was optimised

    def _training_set(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = np.asarray(self.X), np.asarray(self.y)
        if len(y) <= self.max_points:
            return X, y
        half = self.max_points // 2
        keep = set(np.argsort(-y)[:half].tolist()) | set(range(len(y) - half, len(y)))
        idx = np.asarray(sorted(keep))
        return X[idx], y[idx]

    def _fit(self, X: np.ndarray, y: np.ndarray, optimise: bool) -> GaussianProcessRegressor:
        if optimise:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(self.dim, 0.3),
                                                               length_scale_bounds=(1e-2, 1e1), nu=2.5) \
                + WhiteKernel(1e-2, (1e-6, 1e0))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(self.rng.integers(2**31 - 1)))
        else:
            gp = GaussianProcessRegressor(self.kernel, normalize_y=True, optimizer=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        if optimise:
            self.kernel = gp.kernel_
        return gp

    def ask(self, n: int) -> np.ndarray:
        if len(self.y) < self.initial:
            return self.rng.random((n, self.dim))
        X, y = self._training_set()
        optimise = self.kernel is None or len(self.y) - self.optimised_at >= self.REOPTIMISE_EVERY
        if optimise:
            self.optimised_at = len(self.y)
        chosen = []
        for i in range(n):
            gp = self._fit(X, y, optimise=optimise and i == 0)
            best = y.max()
            elites = X[np.argsort(-y)[:10]]
            local = np.clip(elites[self.rng.integers(0, len(elites), 1000)] + self.rng.normal(0, .05, (1000, self.dim)), 0, 1)
            candidates = np.vstack([self.rng.random((4000, self.dim)), local])
            mu, sd = gp.predict(candidates, return_std=True)
            sd = np.maximum(sd, 1e-9)
            z = (mu - best) / sd
            ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
            pick = candidates[int(np.argmax(ei))]
            chosen.append(pick)
            X = np.vstack([X, pick])                  # kriging believer
            y = np.append(y, gp.predict(pick[None])[0])
        return np.asarray(chosen)

    def tell(self, thetas: np.ndarray, scores: np.ndarray) -> None:
        self.X.extend(np.asarray(thetas, dtype=float))
        self.y.extend(float(s) for s in scores)

    def state(self) -> dict:
        return {"pickle": pickle.dumps(self.__dict__)}

    def load(self, state: dict) -> None:
        self.__dict__.update(pickle.loads(state["pickle"]))
