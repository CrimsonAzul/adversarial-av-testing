"""Driver-training logic: curriculum, critic warm-up freezing the actor, imitation fitting."""
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
