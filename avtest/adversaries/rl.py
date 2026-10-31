"""RL adversary: a PPO policy that builds a scenario one parameter at a time.

Design episode = d steps. At step k the policy sees the coordinates chosen so far,
which ones are chosen, how far the search budget has gone, and a compact summary of
the search history (per parameter and bin: share of evaluations, mean severity and
share of valid failures). Its action sets coordinate k. After the last coordinate
the scenario is evaluated and the reward is

    severity (avtest.severity, shared with CMA-ES/BO)  +  0.2 · novelty,

where novelty = 1/sqrt(1 + evaluations already in the scenario's cell). The novelty
bonus is part of this method, not of the shared objective.

W design episodes run in lockstep (``LockstepDesignEnv``), so their scenarios reach
the simulator together as one parallel batch. Training is online: the scenarios the
policy proposes while learning are the run's evaluations and are charged to the same
budget as every other method. Pauses happen only between PPO updates.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

BINS = 4
NOVELTY_WEIGHT = 0.2
ACTION_LIMIT = 3.0


class History:
    """Per-parameter, per-bin running statistics of past evaluations."""

    def __init__(self, dim: int):
        self.count = np.zeros((dim, BINS))
        self.severity = np.zeros((dim, BINS))
        self.failures = np.zeros((dim, BINS))
        self.cells: dict[tuple, int] = {}
        self.total = 0

    def add(self, theta: np.ndarray, score: float, valid_failure: bool, cell: tuple) -> None:
        bins = np.minimum((np.asarray(theta) * BINS).astype(int), BINS - 1)
        rows = np.arange(len(bins))
        self.count[rows, bins] += 1
        self.severity[rows, bins] += score
        self.failures[rows, bins] += float(valid_failure)
        self.cells[cell] = self.cells.get(cell, 0) + 1
        self.total += 1

    def novelty(self, cell: tuple) -> float:
        return 1.0 / np.sqrt(1.0 + self.cells.get(cell, 0))

    def features(self) -> np.ndarray:
        n = max(self.total, 1)
        safe = np.maximum(self.count, 1)
        return np.concatenate([(self.count / n).ravel(), np.clip(self.severity / safe, -1, 1).ravel(),
                               (self.failures / safe).ravel()])


class LockstepDesignEnv(VecEnv):
    """W parallel design episodes that always step together; terminal step evaluates the batch."""

    def __init__(self, dim: int, num_envs: int, evaluate: Callable[[np.ndarray], list[dict]],
                 budget: int, history: History, cell: Callable[[np.ndarray], tuple], used: int = 0):
        self.dim, self.evaluate, self.budget, self.history, self.cell = dim, evaluate, budget, history, cell
        self.used = used
        obs_dim = 2 + 2 * dim + 3 * dim * BINS
        super().__init__(num_envs, gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
                         gym.spaces.Box(-ACTION_LIMIT, ACTION_LIMIT, (1,), np.float32))
        self.stage = 0
        self.x = np.zeros((num_envs, dim))
        self.actions = None
        self.last_rows: list[dict] = []

    def _obs(self) -> np.ndarray:
        mask = (np.arange(self.dim) < self.stage).astype(float)
        shared = np.concatenate([[self.stage / self.dim, min(self.used / self.budget, 1.0)],
                                 mask, self.history.features()])
        return np.stack([np.concatenate([shared[:2], x * mask, shared[2:]]) for x in self.x]).astype(np.float32)

    def reset(self):
        self.stage, self.x = 0, np.zeros((self.num_envs, self.dim))
        return self._obs()

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=float).reshape(self.num_envs)

    def step_wait(self):
        self.x[:, self.stage] = (np.tanh(self.actions) + 1) / 2
        self.stage += 1
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        if self.stage == self.dim:
            thetas = self.x.copy()
            rows = self.evaluate(thetas)
            self.last_rows = rows
            for i, (theta, row) in enumerate(zip(thetas, rows)):
                cell = self.cell(theta)
                rewards[i] = row["score"] + NOVELTY_WEIGHT * self.history.novelty(cell)
                self.history.add(theta, row["score"], row["valid_failure"], cell)
                infos[i] = {"terminal_observation": self._obs()[i], "episode": {"r": float(rewards[i]), "l": self.dim}}
            self.used += len(rows)
            dones[:] = True
            obs = self.reset()
        else:
            obs = self._obs()
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs


class RLAdversary:
    """Owns the PPO generator, its lockstep env and checkpoints. Driven by avtest.search."""

    name = "rl"
    EPISODES_PER_ENV_PER_UPDATE = 6

    def __init__(self, dim: int, seed: int, workers: int = 4, cell: Callable | None = None):
        self.dim, self.seed, self.workers, self.cell = dim, seed, workers, cell
        self.history = History(dim)
        self.model: PPO | None = None
        self.env: LockstepDesignEnv | None = None
        self.updates = 0

    def build(self, evaluate: Callable[[np.ndarray], list[dict]], budget: int, used: int) -> None:
        self.env = LockstepDesignEnv(self.dim, self.workers, evaluate, budget, self.history, self.cell, used)
        n_steps = self.dim * self.EPISODES_PER_ENV_PER_UPDATE
        self.model = PPO("MlpPolicy", self.env, n_steps=n_steps, batch_size=n_steps, n_epochs=8,
                         learning_rate=3e-4, gamma=1.0, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                         seed=self.seed, device="cpu", verbose=0,
                         policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})

    def evaluations_per_update(self) -> int:
        return self.workers * self.EPISODES_PER_ENV_PER_UPDATE

    def update(self) -> None:
        """One complete rollout (``evaluations_per_update`` scenarios) and one PPO update."""
        self.model.learn(total_timesteps=self.model.n_steps * self.workers, reset_num_timesteps=False,
                         progress_bar=False)
        self.updates += 1

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / "generator.zip")
        (folder / "generator_state.pkl").write_bytes(pickle.dumps({"history": self.history, "updates": self.updates,
                                                                     "used": self.env.used}))

    def load(self, folder: Path, evaluate: Callable, budget: int) -> None:
        state = pickle.loads((folder / "generator_state.pkl").read_bytes())
        self.history, self.updates = state["history"], state["updates"]
        self.env = LockstepDesignEnv(self.dim, self.workers, evaluate, budget, self.history, self.cell, state["used"])
        self.model = PPO.load(folder / "generator.zip", env=self.env, device="cpu")

    def propose_frozen(self, n: int, rng_seed: int) -> np.ndarray:
        """Sample n scenarios from the frozen policy with frozen history features (RQ4)."""
        import torch
        torch.manual_seed(rng_seed)
        out = []
        frozen = LockstepDesignEnv(self.dim, 1, lambda t: [], 1, self.history, self.cell, self.env.used)
        frozen.used = frozen.budget                  # budget feature fixed at "fully spent", as at training end
        for _ in range(n):
            frozen.stage, frozen.x = 0, np.zeros((1, self.dim))
            for _k in range(self.dim):
                action, _ = self.model.predict(frozen._obs(), deterministic=False)
                frozen.x[0, frozen.stage] = (np.tanh(float(np.asarray(action).ravel()[0])) + 1) / 2
                frozen.stage += 1
            out.append(frozen.x[0].copy())
        return np.asarray(out)
