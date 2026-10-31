"""Simulator behaviour (real MetaDrive episodes, ~1–2 minutes): determinism, event geometry, recording."""
import numpy as np
import pytest

import torch  # noqa: F401  (before MetaDrive)
from avtest import scenario as sc
from avtest.drivers import IDMDriver
from avtest.sim import DT, AVTestEnv, run_episode

SEED = 100_123                     # validation split


@pytest.fixture(scope="module")
def env():
    e = AVTestEnv()
    yield e
    e.close()


def scene(event: str, **fixed) -> sc.Scenario:
    theta = sc.with_event(np.random.default_rng(3).random(sc.DIM), event, "T2")
    names = [p.name for p in sc.PARAMS]
    for name, value in fixed.items():
        theta[names.index(name)] = value
    return sc.decode(theta, SEED, "T2")


def test_episode_replays_identically(env):
    s = scene("cut_in")
    a, b = run_episode(env, s, IDMDriver()), run_episode(env, s, IDMDriver())
    exact = ("outcome", "steps", "event_triggered_step")
    assert {k: a[k] for k in exact} == {k: b[k] for k in exact}
    # Bullet physics is not bit-exact across resets (~1e-6 relative); the replay check compares outcomes
    for k in ("min_ttc_s", "min_edge_margin_m"):
        assert (a[k] is None and b[k] is None) or a[k] == pytest.approx(b[k], rel=1e-3, abs=1e-3)


def test_lead_vehicle_is_where_and_when_the_scenario_says(env):
    s = scene("lead_brake", event_trigger_s=0.25, event_ahead=0.5)     # 6 s, 28.5 m
    v = s.values
    env.load(s)
    obs, _ = env.reset()
    driver = IDMDriver()
    driver.start(env)
    manager = env.engine.managers["event_manager"]
    trigger_step = None
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(driver.act(env, obs))
        if manager.triggered_at is not None and trigger_step is None:
            trigger_step = manager.triggered_at
            gap = manager.s - manager.ego_path.project(env.agent.position)
            assert abs(gap - v["event_ahead_m"]) < 1.5
            assert manager.speed == pytest.approx(max(0.0, env.agent.speed + v["event_rel_speed_mps"]), abs=0.6)
        if term or trunc:
            break
    assert trigger_step == round(v["event_trigger_s"] / DT)


def test_recording_is_a_faithful_episode(env, tmp_path):
    from avtest.recording import record_episode
    s = scene("obstacle")
    plain = run_episode(env, s, IDMDriver())
    recorded = record_episode(env, s, IDMDriver(), tmp_path / "clip", "test")
    assert (recorded["outcome"], recorded["steps"]) == (plain["outcome"], plain["steps"])
    assert (tmp_path / "clip.gif").stat().st_size > 10_000 and (tmp_path / "clip_keyframes.png").exists()
