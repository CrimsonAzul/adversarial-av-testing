"""Drivers that control the ego vehicle from outside the environment.

- ``SUTDriver``: the trained policy (PPO checkpoint) choosing a lane command and throttle/brake
  deterministically from its observation; avtest.tactical's PID controller does the steering.
  Policy plus controller together are the system under test.
- References (privileged, independent of the SUT; used only to decide avoidability):
  ``IDMDriver`` (MetaDrive IDM with lane changes, 30 km/h cruise), ``CautiousIDMDriver``
  (20 km/h cruise, larger gap and headway), ``ExpertDriver`` (MetaDrive's bundled PPO expert).
A scenario is avoidable if any reference completes it.
"""
from __future__ import annotations

from pathlib import Path

from metadrive.policy.idm_policy import IDMPolicy


class SUTDriver:
    name = "sut"

    def __init__(self, model_path: Path):
        from stable_baselines3 import PPO
        from .tactical import TacticalController
        self.model = PPO.load(str(model_path), device="cpu")
        self.controller = TacticalController()

    def start(self, env):
        self.controller.reset()

    def act(self, env, obs):
        tactical, _ = self.model.predict(obs, deterministic=True)
        return self.controller.low_level(env.agent, tactical)


class CautiousIDMPolicy(IDMPolicy):
    NORMAL_SPEED = 20
    DISTANCE_WANTED = 15.0
    TIME_WANTED = 2.5


class IDMDriver:
    name = "idm"
    policy_class = IDMPolicy

    def start(self, env):
        self.policy = self.policy_class(env.agent, env.scenario.map_seed)

    def act(self, env, obs):
        return self.policy.act(env.agent.id)


class CautiousIDMDriver(IDMDriver):
    name = "idm_cautious"
    policy_class = CautiousIDMPolicy


class ExpertDriver:
    name = "expert"

    def start(self, env):
        from metadrive.examples.ppo_expert.numpy_expert import expert
        self.expert = expert

    def act(self, env, obs):
        return self.expert(env.agent, deterministic=True)


def references() -> list:
    return [IDMDriver(), CautiousIDMDriver(), ExpertDriver()]
