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
    result = pretrain_bc._collect({"seed": 1, "episodes": 2, "model": None, "beta": 1.0, "teacher": "scene_passing"})
    assert [o for _, o in result["outcomes"]] == ["stuck", "stuck"]
    assert result["obs"].shape == (0, OBS_DIM) and result["act"].shape == (0, 2)
