"""Drivers that control the ego vehicle from outside the environment.

- ``SUTDriver``: the trained policy (PPO checkpoint) choosing a lane command and throttle/brake
  deterministically from its observation; avtest.tactical's PID controller does the steering.
  Policy plus controller together are the system under test.
- References (privileged, independent of the SUT; used only to decide avoidability), from 2026-11-02
  for the 60 km/h SUT: ``RoadIDMDriver`` (IDM with lane changes, 60 km/h cruise, slowing for curves),
  ``CautiousRoadIDMDriver`` (the same with a larger gap and headway), ``ExpertDriver`` (MetaDrive's
  bundled PPO expert). ``IDMDriver`` (30 km/h) and ``CautiousIDMDriver`` (20 km/h) were the references
  of qualification attempt 1.
A scenario is avoidable if any reference completes it.
"""
from __future__ import annotations

from pathlib import Path

from metadrive.policy.idm_policy import IDMPolicy


class SUTDriver:
    name = "sut"

    def __init__(self, model_path: Path):
        from .policy import load_policy
        from .tactical import TacticalController, controller_for_model
        self.model = load_policy(model_path)                 # TacticalModel (d8 on) or PPO (d1-d7)
        # each driver drives with the controller it was trained with (recorded in the file from d7 on)
        self.controller_settings = controller_for_model(model_path, self.obs_size)
        self.controller = TacticalController(**self.controller_settings)

    @property
    def obs_size(self) -> int:
        """Input size of the policy; avtest.sim.sensor_config gives the matching sensors."""
        return int(self.model.observation_space.shape[0])

    def start(self, env):
        self.controller.reset()

    def act(self, env, obs):
        tactical, _ = self.model.predict(obs, deterministic=True)
        self.last_decision = tactical                  # (lane command, throttle), shown by scripts.watch_driver
        return self.controller.low_level(env.agent, tactical)


class CautiousIDMPolicy(IDMPolicy):
    NORMAL_SPEED = 20
    DISTANCE_WANTED = 15.0
    TIME_WANTED = 2.5


class RoadIDMPolicy(IDMPolicy):
    """MetaDrive's IDM made fit for 60 km/h (LOG.md, 2026-11-02): it slows down for curves, looks 80 m ahead,
    and computes its following gap in SI units.

    - Curves: the target speed is at most sqrt(A_LAT * radius) on a curve (lateral acceleration 0.3 g, like a
      road's advisory speed) and, before a curve, at most the speed from which it can still slow down to that
      at B_COMFORT. MetaDrive's IDM has no curve logic: at 60 km/h it would leave the road in every curve.
    - Look-ahead: front and adjacent-lane objects within LOOKAHEAD_M (MetaDrive: 30 m, 1.8 s at 60 km/h).
    - Gap: d0 + v*T + v*dv / (2*sqrt(a*b)) with v in m/s. MetaDrive multiplies km/h by seconds, which at
      60 km/h asks for 100–165 m behind any vehicle and brakes hard whenever one is in view.
    """

    NORMAL_SPEED = 60                # km/h
    A_LAT = 3.0                      # m/s², lateral acceleration allowed in curves
    B_COMFORT = 2.0                  # m/s², deceleration used to slow down before a curve
    LOOKAHEAD_M = 80.0
    MAX_LONG_DIST = 80.0
    A_MAX, B_MAX = 2.0, 5.0          # m/s², IDM acceleration and comfortable deceleration for the gap term

    def act(self, *args, **kwargs):
        from metadrive.policy.idm_policy import FrontBackObjects
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object, int(self.LOOKAHEAD_M))
        try:
            if success and self.enable_lane_change:
                front_obj, front_dist, target_lane = self.lane_change_policy(all_objects)
            else:
                near = FrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane,
                                                                 self.control_object.position, self.MAX_LONG_DIST)
                front_obj, front_dist, target_lane = near.front_object(), near.front_min_distance(), self.routing_target_lane
        except Exception:                     # MetaDrive's own fallback, kept
            front_obj, front_dist, target_lane = None, 5, self.routing_target_lane
        action = [self.steering_control(target_lane), self.acceleration(front_obj, front_dist)]
        self.action_info["action"] = action
        return action

    def curve_limit_kmh(self) -> float:
        """Highest speed from which the car can still take every curve within the next 120 m comfortably."""
        import math
        from metadrive.component.lane.circular_lane import CircularLane
        from .sim import RoutePath
        vehicle = self.control_object
        key = (id(vehicle.navigation.checkpoints), vehicle.lane_index[2])
        if getattr(self, "_route_key", None) != key:
            self._route = RoutePath(vehicle.navigation.map.road_network, vehicle.navigation.checkpoints, key[1])
            self._route_key = key
        route = self._route
        here = route.project(vehicle.position)
        limit = self.NORMAL_SPEED / 3.6
        for start, lane in zip(route.starts, route.lanes):
            if start > here + 120.0 or start + lane.length < here or not isinstance(lane, CircularLane):
                continue
            curve_speed = math.sqrt(self.A_LAT * lane.radius)
            limit = min(limit, math.sqrt(curve_speed ** 2 + 2 * self.B_COMFORT * max(start - here, 0.0)))
        return limit * 3.6

    def acceleration(self, front_obj, dist_to_front) -> float:
        from metadrive.utils.math import not_zero
        vehicle = self.control_object
        target = not_zero(min(self.target_speed, self.curve_limit_kmh()), 0)
        acc = self.ACC_FACTOR * (1 - (max(vehicle.speed_km_h, 0) / target) ** self.DELTA)
        if front_obj and not self.disable_idm_deceleration:
            acc -= self.ACC_FACTOR * (self.desired_gap(vehicle, front_obj) / not_zero(dist_to_front)) ** 2
        return acc

    def steering_control(self, target_lane) -> float:
        """MetaDrive's lane-tracking PID with the lateral error capped at 1 m, as in the SUT's controller: the
        uncapped lane change overshoots onto the centre line at 60 km/h (LOG.md, 2026-11-02)."""
        import numpy as np
        from metadrive.utils.math import wrap_to_pi
        vehicle = self.control_object
        long, lat = target_lane.local_coordinates(vehicle.position)
        steering = self.heading_pid.get_result(-wrap_to_pi(target_lane.heading_theta_at(long + 1) - vehicle.heading_theta))
        steering += self.lateral_pid.get_result(-float(np.clip(lat, -1.0, 1.0)))
        return float(steering)

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        import numpy as np
        v = ego_vehicle.speed_km_h / 3.6
        dv = float(np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading)) if projected \
            else v - front_obj.speed_km_h / 3.6
        return self.DISTANCE_WANTED + v * self.TIME_WANTED + v * dv / (2 * np.sqrt(self.A_MAX * self.B_MAX))


class CautiousRoadIDMPolicy(RoadIDMPolicy):
    """RoadIDMPolicy with the cautious gaps: 15 m standstill gap, 2 s time gap."""

    DISTANCE_WANTED = 15.0
    TIME_WANTED = 2.0


class TrafficIDMPolicy(RoadIDMPolicy):
    """Background traffic on the 60 km/h road (avtest.sim.ROAD_TRAFFIC): RoadIDMPolicy cruising at a speed drawn
    per vehicle from SPEED_RANGE_KMH with the policy's seeded generator, looking 50 m ahead. MetaDrive's traffic
    cruises at 30 km/h: d4 changed lanes 4.4 times per scene overtaking it, and 62 of its 65 T1 failures, like
    every lead-brake collision of the 60 km/h teacher, came with lane changes (LOG.md, 2026-11-03)."""

    SPEED_RANGE_KMH = (50.0, 60.0)
    LOOKAHEAD_M = MAX_LONG_DIST = 50.0

    def __init__(self, control_object, random_seed):
        super().__init__(control_object, random_seed)
        self.NORMAL_SPEED = float(self.np_random.uniform(*self.SPEED_RANGE_KMH))   # per vehicle; reset() reads it
        self.target_speed = self.NORMAL_SPEED


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


class RoadIDMDriver(IDMDriver):
    """Reference at 60 km/h with curve slowing (RoadIDMPolicy); with CautiousRoadIDMDriver and the expert, the
    reference set from 2026-11-02 on. The 20/30 km/h drivers above reproduce qualification attempt 1."""
    name = "idm60"
    policy_class = RoadIDMPolicy


class CautiousRoadIDMDriver(IDMDriver):
    name = "idm_cautious60"
    policy_class = CautiousRoadIDMPolicy


class ExpertDriver:
    name = "expert"

    def start(self, env):
        from metadrive.examples.ppo_expert.numpy_expert import expert
        self.expert = expert

    def act(self, env, obs):
        return self.expert(env.agent, deterministic=True)


def references() -> list:
    """The avoidability references for the 60 km/h SUT: IDM and cautious IDM at its cruise speed, and the expert."""
    return [RoadIDMDriver(), CautiousRoadIDMDriver(), ExpertDriver()]


def qualified_sut(root: Path) -> tuple[Path, dict]:
    """models/sut.zip and models/QUALIFIED.json, after checking that the model file and the SUT controller
    settings in avtest.tactical are still the ones that passed qualification."""
    from .records import read_json, sha256_file
    record, model = root / "models/QUALIFIED.json", root / "models/sut.zip"
    if not record.exists():
        raise SystemExit("No qualified SUT: run scripts.qualify first (docs/RUNBOOK.md, step 2).")
    q = read_json(record)
    if sha256_file(model) != q["model_sha256"]:
        raise SystemExit("models/sut.zip does not match models/QUALIFIED.json")
    from .tactical import controller_for_model
    controller = controller_for_model(model, q.get("obs_size"))
    if q.get("controller") != controller:
        raise SystemExit(f"models/sut.zip drives with controller {controller}, "
                         f"but the qualified SUT used {q.get('controller')}")
    return model, q
