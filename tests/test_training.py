"""Driver-training logic: run modes, curriculum, critic warm-up freezing the actor, imitation fitting."""
import gymnasium as gym
import numpy as np
import pytest
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


def test_best_round_is_chosen_on_the_training_tier():
    from scripts.pretrain_bc import all_tiers_success, selection_key
    assert [selection_key(t) for t in ("T1", "T2", "T3")] == ["val_t1_success", "val_success", "val_t3_success"]
    assert selection_key("all") == "val_all_success"
    assert all_tiers_success({"val_t1_success": 0.9, "val_success": 0.6, "val_t3_success": 0.6}) == pytest.approx(0.7)


def test_frame_history_gives_the_change_over_two_steps():
    """The end-to-end input: the observation and its change over 2 steps (0.2 s), zero at the episode's start,
    from float16-rounded values as stored for training; stored samples (every 2nd step) give the same input."""
    from avtest.policy import FrameHistory, stored_features
    history = FrameHistory()
    frames = [np.full(3, v, dtype=np.float32) for v in (0.1, 0.2, 0.3, 0.45, 0.6)]
    out = [history(f) for f in frames]
    assert np.allclose(out[0][3:], 0) and np.allclose(out[2][3:], 0.2, atol=1e-3) and np.allclose(out[4][3:], 0.3, atol=1e-3)
    assert out[1].dtype == np.float32 and np.array_equal(out[1][:3], frames[1].astype(np.float16).astype(np.float32))
    stored = np.stack(frames[::2]).astype(np.float16)                   # steps 0, 2, 4 of one episode
    first = np.array([True, False, False])
    assert np.allclose(stored_features(stored, first, np.arange(3)), np.stack([out[0], out[2], out[4]]))
    history.reset()
    assert np.allclose(history(frames[4])[3:], 0)


def end_to_end_data(n_episodes: int, length: int, seed: int):
    """Synthetic episodes: intent from x0, steering per intent (when keeping: from x4's change between stored samples,
    i.e. motion), throttle with rare hard braking from x3."""
    from avtest.policy import stored_features
    rng = np.random.default_rng(seed)
    obs = rng.random((n_episodes * length, 8)).astype(np.float32)
    speed = rng.uniform(-0.04, 0.04, n_episodes).repeat(length)
    step = np.tile(np.arange(length), n_episodes)
    obs[:, 4] = 0.5 + speed * step
    obs = obs.astype(np.float16)
    first = step == 0
    delta = stored_features(obs, first, np.arange(len(obs)))[:, 8 + 4]
    x = obs.astype(np.float32)
    intent = np.where(x[:, 0] > 0.8, 1.0, np.where(x[:, 0] < 0.2, -1.0, 0.0))
    steer = np.where(intent > 0, 0.3 + 0.1 * x[:, 2], np.where(intent < 0, -0.3 - 0.1 * x[:, 2], 2.0 * delta))
    throttle = np.where(x[:, 3] > 0.92, -1.0, 0.35)
    return obs, np.stack([intent, steer, throttle], 1).astype(np.float32), first


def test_end_to_end_fit_learns_steering_per_intent_and_from_motion():
    from avtest.policy import EndToEndModel, fit_end_to_end, stored_features
    obs, act, first = end_to_end_data(600, 10, 0)
    heldout = np.arange(len(act)) >= 5400                               # the last 60 episodes
    model = EndToEndModel(8, (64, 64))
    train = np.flatnonzero(~heldout)
    model.set_normalization(stored_features(obs, first, train), act[train, 1], act[train, 0])
    stats = fit_end_to_end(model, obs, act, first, train, np.flatnonzero(heldout), epochs=200, seed=0)
    assert stats["lane_command_agreement"] > 0.9 and stats["lane_change_recall"] > 0.9
    # keeping steering comes from the motion input: 0.0026 with it, 0.0155 with it zeroed (labels' spread 0.047)
    assert stats["steer_mae_keep_teacher_intent"] < 0.15 * stats["steer_label_std_keep"]
    assert stats["steer_mae"] < 0.15 * stats["steer_label_std"]                 # with its own intent decisions
    assert stats["hard_brake_recall"] > 0.8


def test_end_to_end_model_drives_without_a_controller(tmp_path):
    """Saved and loaded with its input scaling; SUTDriver passes its steering and throttle to the simulator
    unchanged and resets its frame history each episode; a model and a recorded controller that disagree are
    refused."""
    from avtest.drivers import SUTDriver
    from avtest.policy import EndToEndModel, is_end_to_end_model, load_policy, stored_features
    from avtest.tactical import END_TO_END, SUT_CONTROLLER, stamp_controller
    obs, act, first = end_to_end_data(40, 10, 1)
    model = EndToEndModel(8, (16,))
    model.set_normalization(stored_features(obs, first, np.arange(len(obs))), act[:, 1], act[:, 0])
    path = tmp_path / "e2e.zip"
    model.save(path)
    stamp_controller(path, END_TO_END)
    assert is_end_to_end_model(path)
    again = load_policy(path)
    assert isinstance(again, EndToEndModel) and again.observation_space.shape == (8,)
    assert np.allclose(again.net.std.numpy(), model.net.std.numpy())
    driver = SUTDriver(path)
    assert driver.end_to_end and driver.controller is None and driver.controller_settings == END_TO_END
    episode = [obs[i].astype(np.float32) for i in range(5)]
    driver.start(None)
    first_run = [driver.act(None, o) for o in episode]
    driver.start(None)
    assert all(np.array_equal(a, driver.act(None, o)) for a, o in zip(first_run, episode))   # history was reset
    model.reset()
    assert np.allclose(first_run[0], model.predict(episode[0])[0])
    assert all(-1 <= a[0] <= 1 and a[1] in model.throttle_bins for a in first_run)
    wrong = tmp_path / "wrong.zip"
    model.save(wrong)
    stamp_controller(wrong, SUT_CONTROLLER)
    with pytest.raises(ValueError):
        SUTDriver(wrong)


def test_end_to_end_rounds_load_in_order_with_round_zero_held_out(tmp_path):
    rng = np.random.default_rng(0)
    for k, episodes in ((0, 41), (1, 3)):
        first = np.tile([True, False, False], episodes)
        np.savez(tmp_path / f"round_{k:02d}.npz", obs=rng.random((len(first), 4)).astype(np.float16),
                 act=np.full((len(first), 3), k, dtype=np.float32), first=first)
    obs, act, first, heldout = pretrain_bc.load_rounds(tmp_path, 1)
    assert obs.shape == (132, 4) and obs.dtype == np.float16 and act[:123, 0].max() == 0 and act[123:, 0].min() == 1
    assert first.sum() == 44 and heldout.sum() == 3 * 3                 # round-0 episodes 0, 20 and 40, 3 samples each
    assert heldout[:3].all() and heldout[60:63].all() and not heldout[123:].any()


def test_end_to_end_steering_network_sees_only_the_leading_inputs(tmp_path):
    """steer_features: the steering outputs come from their own network on the chosen values and their changes (the
    ego and lane state), so other inputs cannot move them; saved and loaded with the model."""
    import torch
    from avtest.policy import EndToEndModel, load_policy, stored_features
    obs, act, first = end_to_end_data(300, 10, 2)
    model = EndToEndModel(8, (32,), steer_features=(0, 1, 2, 3, 4), steer_hidden=(16,))
    model.set_normalization(stored_features(obs, first, np.arange(len(obs))), act[:, 1], act[:, 0])
    x = torch.as_tensor(stored_features(obs, first, np.arange(20)))
    y = x.clone()
    y[:, 5:8] += 0.5                                                    # change inputs 5-7 and their changes
    y[:, 13:16] += 0.5
    with torch.no_grad():
        assert torch.allclose(model.net(x)[1], model.net(y)[1]) and not torch.allclose(model.net(x)[0], model.net(y)[0])
    model.save(tmp_path / "m.zip")
    again = load_policy(tmp_path / "m.zip")
    assert again.steer_features == (0, 1, 2, 3, 4) and again.steer_hidden == (16,)
    with torch.no_grad():
        assert torch.allclose(again.net(x)[1], model.net(x)[1])
