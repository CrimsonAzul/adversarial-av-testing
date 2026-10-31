"""TrainingEnv in its own module: MetaDrive allows one engine per process, and test_sim keeps one open."""
import numpy as np

import torch  # noqa: F401  (before MetaDrive)
from avtest.sim import TrainingEnv


def test_training_env_samples_valid_observations():
    e = TrainingEnv("T3", seed=5)
    try:
        e.set_difficulty(0.3)
        def in_bounds(o):                       # MetaDrive can round a hair past [0, 1]
            return (o.shape == e.observation_space.shape and np.all(np.isfinite(o))
                    and o.min() >= -1e-3 and o.max() <= 1 + 1e-3)
        obs, _ = e.reset()
        assert in_bounds(obs)
        for _ in range(20):
            obs, reward, term, trunc, info = e.step(e.action_space.sample())
            assert np.isfinite(reward) and in_bounds(obs)
            if term or trunc:
                assert info["outcome"] in ("success", "collision", "offroad", "stuck", "timeout")
                break
    finally:
        e.close()
