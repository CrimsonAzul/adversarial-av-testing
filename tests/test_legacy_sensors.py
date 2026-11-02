"""Older drivers keep their sensors (own module: MetaDrive allows one engine per process)."""
import numpy as np
import pytest

import torch  # noqa: F401  (before MetaDrive)
from avtest import scenario as sc
from avtest.sim import LEGACY_LIDAR, OBS_DIM, AVTestEnv, obs_dim, sensor_config


def test_sensor_setup_follows_the_driver_input_size():
    assert sensor_config(OBS_DIM) == {}
    with pytest.raises(ValueError):
        sensor_config(123)
    env = AVTestEnv(sensor_config(obs_dim(LEGACY_LIDAR["num_lasers"])))     # d1–d3: 240 beams to 50 m
    try:
        env.load(sc.decode(np.random.default_rng(1).random(sc.DIM), 100_123, "T1"))
        obs, _ = env.reset()
        assert obs.shape == (320,)
        assert env.agent.config["lidar"]["num_lasers"] == 240 and env.agent.config["lidar"]["distance"] == 50
        assert env.agent.config["lane_line_detector"]["num_lasers"] > 0       # the other sensors are kept
    finally:
        env.close()
