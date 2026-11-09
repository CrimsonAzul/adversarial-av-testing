"""The teacher's brake hold (TacticalController brake_hold; real MetaDrive vehicle, own module: one engine at a time)."""
import math

import numpy as np
import pytest

import torch  # noqa: F401  (before MetaDrive)
from avtest import scenario as sc, seeds, tactical
from avtest.sim import AVTestEnv


@pytest.fixture(scope="module")
def env():
    e = AVTestEnv()
    rng = np.random.default_rng(3)
    while True:
        scene = sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), "T1")
        if scene.values["map"][0] == "S":
            break
    e.load(scene)
    e.reset()
    yield e
    e.close()


def label(env, settings: dict, last_throttle: float, ttc: float = math.inf, conflict: float = math.inf) -> float:
    """The controller's throttle for a cruising command when the vehicle last applied ``last_throttle``, with the
    time to collision and to a predicted conflict set."""
    ctl = tactical.TacticalController(**settings)
    ctl.time_to_collision = lambda vehicle: (ttc, math.inf)
    ctl.predicted_conflict = lambda vehicle, horizon: (conflict, math.inf)
    env.agent.last_current_action.append((0.0, last_throttle))
    return float(ctl.low_level(env.agent, [tactical.KEEP, 0.5])[1])


def test_the_hold_reads_the_throttle_the_observation_reports(env):
    obs, *_ = env.step([0.0, -1.0])
    assert env.agent.last_current_action[-1][1] == -1.0 and obs[28] == 0.0      # (throttle + 1) / 2
    obs, *_ = env.step([0.0, 0.6])
    assert env.agent.last_current_action[-1][1] == pytest.approx(0.6) and obs[28] == pytest.approx(0.8)


def test_emergency_and_predictive_brakes_hold_one_second_past_their_thresholds_after_hard_braking(env):
    hold, plain = tactical.BRAKE_HOLD_CONTROLLER, tactical.YIELD_PD_CONTROLLER
    assert hold["aeb_ttc"] == 2.5 and hold["yield_s"] == 3.0 and hold["brake_hold"] == 1.0
    # time to collision 3.0 s: past the 2.5 s threshold, within the hold
    assert label(env, hold, -1.0, ttc=3.0) == -1.0
    assert label(env, hold, 0.5, ttc=3.0) > -0.5                 # not braking at its last step: no hold
    assert label(env, plain, -1.0, ttc=3.0) > -0.5               # no hold in the d16-d18 teacher
    assert label(env, hold, -1.0, ttc=3.6) > -0.5                # danger past the hold
    assert label(env, hold, -0.35, ttc=3.0) > -0.5               # moderate braking does not hold
    # a predicted conflict 3.5 s ahead: past the 3 s threshold, within the hold
    assert label(env, hold, -1.0, conflict=3.5) == -1.0
    assert label(env, hold, 0.5, conflict=3.5) > -0.5
    assert label(env, hold, -1.0, conflict=4.2) > -0.5
    assert label(env, hold, 0.5, ttc=2.0) == -1.0 and label(env, hold, 0.5, conflict=2.5) == -1.0   # thresholds
