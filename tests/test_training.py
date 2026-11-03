"""Driver-training logic: run modes, curriculum, critic warm-up freezing the actor, imitation fitting."""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from scripts import pretrain_bc, train_driver


def test_curriculum_ramps_then_holds():
    cfg = {"curriculum_until": 0.3, "total_steps": 1000}
    assert train_driver.difficulty(0, cfg) == 0.3
    assert 0.3 < train_driver.difficulty(150, cfg) < 1.0
    assert train_driver.difficulty(300, cfg) == train_driver.difficulty(900, cfg) == 1.0


def test_each_mode_stops_at_its_own_step_count(tmp_path):
    from argparse import Namespace
    from pytest import raises

    def settings(**mode):
        args = Namespace(**{"name": "d9", "envs": train_driver.DEFAULTS["envs"], "check": False, "pilot": False, **mode})
        return train_driver.run_settings(args, root=tmp_path)

    out, cfg, panel, stop_at = settings()
    assert (out.name, stop_at, panel) == ("d9", 40_000_000, 200) and cfg == train_driver.DEFAULTS
    out, cfg, panel, stop_at = settings(pilot=True)
    assert (out.name, stop_at, panel, cfg["critic_warmup_steps"]) == ("d9_pilot", 700_000, 100, 100_000)
    out, cfg, panel, stop_at = settings(check=True)     # a check must end in minutes, not at the session limit
    assert (out.name, stop_at, panel) == ("d9_check", 6144, 4) and stop_at == cfg["total_steps"]
    warm = tmp_path / "runs/driver/d9_bccheck"
    warm.mkdir(parents=True)
    (warm / "model_last.zip").touch()
    out, cfg, panel, stop_at = settings(check=True)
    assert (out.name, stop_at) == ("d9_bccheck", 6144)
    with raises(SystemExit):
        settings(check=True, pilot=True)


def small_ppo():
    env = gym.make("Pendulum-v1")
    return PPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=2, seed=0, device="cpu",
               policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})


def test_critic_warmup_changes_only_the_value_network():
    model = small_ppo()
    actor_before = [p.detach().clone() for p in train_driver.actor_parameters(model)]
    critic_before = [p.detach().clone() for p in model.policy.value_net.parameters()]
    for p in train_driver.actor_parameters(model):
        p.requires_grad_(False)
    model.learn(256)
    for p in train_driver.actor_parameters(model):
        p.requires_grad_(True)
    assert all(torch.equal(a, b) for a, b in zip(actor_before, train_driver.actor_parameters(model)))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, model.policy.value_net.parameters()))
    model.learn(256)                                   # unfrozen: the actor moves again
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, train_driver.actor_parameters(model)))


def test_imitation_fit_learns_teacher_actions():
    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(-0.0, 1.0, (259,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    model = PPO("MlpPolicy", Spaces(), seed=0, device="cpu", policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64]}})
    rng = np.random.default_rng(0)
    obs = rng.random((8000, 259)).astype(np.float16)
    act = np.stack([np.tanh(3 * (obs[:, 0] - 0.5)), np.tanh(3 * (obs[:, 1] - obs[:, 2]))], axis=1).astype(np.float32)
    first = pretrain_bc.fit(model, obs, act, epochs=1, seed=0)
    later = pretrain_bc.fit(model, obs, act, epochs=40, seed=1)
    assert later["action_mae"] < 0.6 * first["action_mae"] and later["action_mae"] < 0.08
    test = rng.random((1000, 259)).astype(np.float32)          # generalises, not just memorises
    truth = np.stack([np.tanh(3 * (test[:, 0] - 0.5)), np.tanh(3 * (test[:, 1] - test[:, 2]))], axis=1)
    assert np.abs(model.predict(test, deterministic=True)[0] - truth).mean() < 0.1


def test_model_file_records_its_controller(tmp_path):
    """From d7 the training controller is written into the model zip; the model still loads, SUTDriver uses it."""
    from avtest.drivers import SUTDriver
    from avtest.sim import OBS_DIM
    from avtest.tactical import HOLD_CONTROLLER, SUT_CONTROLLER, controller_for_model, stamp_controller

    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    path = tmp_path / "model.zip"
    PPO("MlpPolicy", Spaces(), n_steps=64, batch_size=64, device="cpu").save(path)
    assert controller_for_model(path, OBS_DIM) == SUT_CONTROLLER          # unstamped 800-input model: d4-d6
    stamp_controller(path, HOLD_CONTROLLER)
    assert controller_for_model(path, OBS_DIM) == HOLD_CONTROLLER
    assert SUTDriver(path).controller_settings == HOLD_CONTROLLER and SUTDriver(path).controller.hold_lane
    import pytest
    with pytest.raises(ValueError):
        stamp_controller(path, HOLD_CONTROLLER)                          # recorded once, never overwritten


def test_tactical_model_classifies_the_lane_decision(tmp_path):
    """d8's student: a 3-way lane class plus throttle, saved and loaded like the PPO models it replaces."""
    from avtest.drivers import SUTDriver
    from avtest.policy import TacticalModel, fit_classify, is_tactical_model, lane_classes, load_policy
    from avtest.tactical import SMOOTH_CONTROLLER, stamp_controller
    assert lane_classes(np.array([1.0, 0.0, -1.0, 0.3])).tolist() == [0, 1, 2, 1]
    rng = np.random.default_rng(0)
    obs = rng.random((3000, 12)).astype(np.float32)
    lane = np.where(obs[:, 0] > 0.8, 1.0, np.where(obs[:, 0] < 0.2, -1.0, 0.0))     # a learnable rule
    act = np.stack([lane, 2 * obs[:, 1] - 1], 1).astype(np.float32)
    model = TacticalModel(12, (32, 32))
    stats = fit_classify(model, obs, act, epochs=150, seed=0)
    assert stats["lane_command_agreement"] > 0.95 and stats["lane_change_recall"] > 0.9
    action, _ = model.predict(obs[0])
    assert -1 <= action[0] <= 1 and -1 <= action[1] <= 1                  # P(left) - P(right), throttle
    decided = np.array([model.predict(o)[0][0] for o in obs[:300]])
    assert np.mean((np.abs(decided) > 0.5) == (lane[:300] != 0)) > 0.9  # above the threshold where the rule says
    path = tmp_path / "student.zip"
    model.save(path)
    stamp_controller(path, SMOOTH_CONTROLLER)
    assert is_tactical_model(path)
    again = load_policy(path)
    assert again.observation_space.shape == (12,)
    assert np.allclose(again.predict(obs[5])[0], model.predict(obs[5])[0])
    driver = SUTDriver(path)
    assert driver.obs_size == 12 and driver.controller_settings == SMOOTH_CONTROLLER


def test_copy_with_controller_changes_only_the_controller(tmp_path):
    from avtest.policy import TacticalModel, load_policy
    from avtest.tactical import SMOOTH_CONTROLLER, controller_for_model, copy_with_controller, stamp_controller
    model = TacticalModel(6, (8,))
    model.save(tmp_path / "a.zip")
    stamp_controller(tmp_path / "a.zip", SMOOTH_CONTROLLER)
    copy_with_controller(tmp_path / "a.zip", tmp_path / "b.zip", {**SMOOTH_CONTROLLER, "threshold": 0.3})
    assert controller_for_model(tmp_path / "a.zip", 6)["threshold"] == 0.5
    assert controller_for_model(tmp_path / "b.zip", 6) == {**SMOOTH_CONTROLLER, "threshold": 0.3}
    x = np.linspace(0, 1, 6, dtype=np.float32)
    assert np.allclose(load_policy(tmp_path / "a.zip").predict(x)[0], load_policy(tmp_path / "b.zip").predict(x)[0])


def test_an_imitation_run_may_only_add_rounds():
    base = {"rounds": 5, "tier": "T1", "head": "classify"}
    assert pretrain_bc.same_run(base, dict(base)) and pretrain_bc.same_run(base, {**base, "rounds": 9})
    assert not pretrain_bc.same_run(base, {**base, "rounds": 3})
    assert not pretrain_bc.same_run(base, {**base, "tier": "T2"})



def test_binned_throttle_learns_hard_braking(tmp_path):
    """d9 throttle head: the most likely of THROTTLE_BINS; a rule with rare hard braking is learned."""
    from avtest.policy import THROTTLE_BINS, TacticalModel, fit_classify, load_policy, throttle_classes
    assert throttle_classes(np.array([-1.0, -0.6, 0.02, 0.97])).tolist() == [0, 1, 4, 8]
    rng = np.random.default_rng(1)
    obs = rng.random((4000, 10)).astype(np.float32)
    throttle = np.where(obs[:, 0] > 0.92, -1.0, 0.7)                   # 8% hard braking, else steady throttle
    act = np.stack([np.zeros(len(obs)), throttle], 1).astype(np.float32)
    model = TacticalModel(10, (32, 32), throttle_bins=THROTTLE_BINS)
    stats = fit_classify(model, obs, act, epochs=400, seed=0)
    assert stats["hard_brake_recall"] > 0.9 and stats["throttle_mae"] < 0.1
    model.save(tmp_path / "m.zip")
    again = load_policy(tmp_path / "m.zip")
    assert again.throttle_bins == THROTTLE_BINS and again.predict(obs[0])[0][1] in THROTTLE_BINS
