"""Imitation data collection (own module: MetaDrive allows one engine per process)."""
import numpy as np

import torch  # noqa: F401  (before MetaDrive)
from scripts import pretrain_bc


class CrashingTeacher:
    name = "crashing"

    def start(self, env):
        pass

    def act(self, env, obs):
        return [0.8, 1.0]                     # full throttle, hard steer: leaves the road


def test_failed_teacher_demonstrations_are_dropped(monkeypatch):
    import avtest.drivers
    monkeypatch.setattr(avtest.drivers, "CautiousIDMDriver", CrashingTeacher)
    result = pretrain_bc._collect({"seed": 1, "episodes": 2, "model": None, "beta": 1.0})
    assert [o for _, o in result["outcomes"]] and all(o != "success" for _, o in result["outcomes"])
    assert result["obs"].shape == (0, pretrain_bc_obs_dim()) and result["act"].shape == (0, 2)


def pretrain_bc_obs_dim() -> int:
    from avtest.sim import OBS_DIM
    return OBS_DIM
