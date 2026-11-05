"""Imitation data collection (own module: MetaDrive allows one engine per process)."""
import torch  # noqa: F401  (before MetaDrive)
from metadrive.policy.idm_policy import IDMPolicy

from scripts import pretrain_bc


class BrakingTeacher(IDMPolicy):
    """A teacher that never moves: every episode it drives ends stuck."""

    steered_lane = None

    def act(self, *args, **kwargs):
        self.steered_lane = self.control_object.navigation.current_lane
        return [0.0, -1.0]


def test_failed_teacher_demonstrations_are_dropped(monkeypatch):
    import avtest.tactical
    from avtest.sim import OBS_DIM
    monkeypatch.setitem(avtest.tactical.TEACHERS, "scene_passing", BrakingTeacher)
    result = pretrain_bc._collect({"seed": 1, "episodes": 2, "model": None, "beta": 1.0, "teacher": "scene_passing", "tier": "T2",
                                  "controller": avtest.tactical.LEGACY_CONTROLLER})
    assert [o for _, o in result["outcomes"]] == ["stuck", "stuck"]
    assert result["obs"].shape == (0, OBS_DIM) and result["act"].shape == (0, 2)


def test_end_to_end_labels_are_the_teacher_final_action():
    """--action e2e: each stored sample's label is the side the teacher's controller steered for, and its final
    steering and throttle; samples mark episode starts."""
    import numpy as np
    import avtest.tactical
    from avtest.sim import OBS_DIM
    result = pretrain_bc._collect({"seed": 3, "episodes": 1, "model": None, "beta": 1.0, "teacher": "scene_passing_60",
                                  "tier": "T1", "controller": avtest.tactical.YIELD_CONTROLLER, "action": "e2e",
                                  "dart": 0.02})
    (_, outcome), = result["outcomes"]
    assert outcome == "success"
    obs, act, first = result["obs"], result["act"], result["first"]
    assert obs.shape[1] == OBS_DIM and act.shape == (len(obs), 3) and len(first) == len(obs) > 100
    assert first[0] and first.sum() == 1
    assert set(np.unique(act[:, 0])) <= {-1.0, 0.0, 1.0}
    assert np.abs(act[:, 1]).max() <= 1 and 0 < np.abs(act[:, 1]).mean() < 0.3       # steering, mostly small
    assert act[:, 2].min() >= -1 and act[:, 2].max() <= 1 and act[:, 2].max() > 0.3    # it accelerates
