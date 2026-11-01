"""MetaDrive environment for one decoded scenario, plus the training wrapper.

``AVTestEnv`` is MetaDriveEnv with:
- per-episode road layout, lane count and lane width (maps are rebuilt every reset);
- MetaDrive's own IDM background traffic at the scenario's density (deterministic per map seed);
- one scripted event (``EventManager``): a lead vehicle that brakes and later recovers, a
  vehicle that cuts in from an adjacent lane, or a broken-down vehicle with a warning
  triangle. Event vehicles move kinematically along the ego's route, so their speeds,
  decelerations and lane-change durations are exactly the scenario's bounded values;
- telemetry for the severity score: minimum time-to-collision with anything ahead,
  minimum distance to the road edge, and a stuck detector (no 1 m of progress in 10 s).

Outcome of an episode: success (destination reached), collision (vehicle, object or
building), offroad, stuck, or timeout (horizon reached).

Import torch before this module on Windows (MetaDrive first triggers a c10.dll load failure).
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from metadrive.component.static_object.traffic_object import TrafficWarning
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.base_manager import BaseManager

from . import scenario as sc
from . import seeds

DT = 0.1                      # seconds per env step (MetaDrive default: 5 × 0.02 s physics)
HORIZON = 1500                # 150 s; reaching it is a timeout failure
STUCK_WINDOW = 100            # 10 s
STUCK_DISTANCE_M = 1.0
LEAD_RECOVERY_S = 2.0         # a braking lead vehicle waits this long at its low point, then speeds up again
LEAD_RECOVERY_ACCEL = 1.5
TTC_LOOKAHEAD_M = 60.0

# Reward (docs/LOG.md, 2026-11-01): a whole route earns ~15–20 progress points + 20 for arriving,
# and every failure costs 30, so a crash always outweighs the progress made before it.
# (Run d1 used MetaDrive's 1 point per metre against a 10-point crash penalty: speed paid
# more than safety, and PPO learned to drive fast and crash.)
PROGRESS_REWARD_PER_M = 0.05
SPEED_REWARD = 0.02            # per step at maximum speed: keeps the driver from creeping into timeouts
SUCCESS_REWARD = 20.0
FAILURE_PENALTY = 30.0         # collision, off-road, stuck and timeout alike
NEAREST_VEHICLES = 4           # object list: relative position/velocity of the 4 nearest vehicles
# Lane-line and road-edge detection (a camera's lane markings and kerbs). Without them MetaDrive gives
# only one lane-relative number (offset from the current lane centre), no lane boundaries and no view
# of the road ahead; the pilot student weaved and crossed the centre line (docs/LOG.md, 2026-11-01).
LANE_LINE_BEAMS, LANE_LINE_RANGE_M = 24, 20
ROAD_EDGE_BEAMS, ROAD_EDGE_RANGE_M = 24, 50
# MetaDrive's size formula (obs/state_obs.py): 259 by default; the road-edge beams replace its 2 edge
# distances, the lane-line beams replace its 1 lane-centre offset, and each nearest vehicle adds 4
OBS_DIM = 259 + 4 * NEAREST_VEHICLES + (ROAD_EDGE_BEAMS - 2) + (LANE_LINE_BEAMS - 1)

BASE_CONFIG = dict(
    use_render=False, log_level=50, store_map=False, start_seed=0, num_scenarios=seeds.SPLITS["heldout"][1],
    map="SSS", traffic_density=0.0, accident_prob=0.0, random_traffic=False, random_lane_width=False,
    random_lane_num=False, horizon=HORIZON, crash_vehicle_done=True, crash_object_done=True,
    out_of_road_done=True, success_reward=SUCCESS_REWARD, out_of_road_penalty=FAILURE_PENALTY,
    crash_vehicle_penalty=FAILURE_PENALTY, crash_object_penalty=FAILURE_PENALTY,
    crash_sidewalk_penalty=FAILURE_PENALTY, driving_reward=PROGRESS_REWARD_PER_M, speed_reward=SPEED_REWARD,
    use_lateral_reward=True,
    vehicle_config={"lidar": {"num_lasers": 240, "distance": 50, "num_others": NEAREST_VEHICLES},
                    "lane_line_detector": {"num_lasers": LANE_LINE_BEAMS, "distance": LANE_LINE_RANGE_M},
                    "side_detector": {"num_lasers": ROAD_EDGE_BEAMS, "distance": ROAD_EDGE_RANGE_M}},
)


def outcome_from(info: dict, stuck: bool) -> str | None:
    if info.get("arrive_dest"):
        return "success"
    if info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_building") or info.get("crash_human"):
        return "collision"
    if info.get("out_of_road") or info.get("crash_sidewalk"):
        return "offroad"
    if stuck:
        return "stuck"
    if info.get("max_step"):
        return "timeout"
    return None


class RoutePath:
    """Concatenated lanes with a fixed lane index along the ego's route; s = arc length."""

    def __init__(self, network, checkpoints: list, lane_index: int):
        self.lanes = []
        for a, b in zip(checkpoints[:-1], checkpoints[1:]):
            lanes = network.graph[a][b]
            self.lanes.append(lanes[min(max(lane_index, 0), len(lanes) - 1)])
        self.starts = np.cumsum([0.0] + [lane.length for lane in self.lanes])

    @property
    def length(self) -> float:
        return float(self.starts[-1])

    def position(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length - 1e-6)
        i = int(np.searchsorted(self.starts, s, side="right") - 1)
        return np.asarray(self.lanes[i].position(s - self.starts[i], 0.0))

    def project(self, position) -> float:
        """Arc length of the closest point (searches every lane; routes are short)."""
        best, best_s = math.inf, 0.0
        for start, lane in zip(self.starts, self.lanes):
            long, lat = lane.local_coordinates(position)
            if -2.0 <= long <= lane.length + 2.0 and abs(lat) < best:
                best, best_s = abs(lat), start + min(max(long, 0.0), lane.length)
        return best_s


class EventManager(BaseManager):
    """Spawns and drives the scenario's one scripted event, defined relative to the ego.

    lead_brake / cut_in: until the trigger time the event vehicle escorts the ego at
    ``event_ahead_m`` ahead along the route (its lane: same or adjacent) at the ego's speed.
    At the trigger it takes speed ego + ``event_rel_speed_mps`` and then drives on its own:
    a lead vehicle brakes to a stop at the scenario's deceleration, waits 2 s and returns to
    its trigger speed; a cut-in vehicle moves into the ego's lane over the scenario's duration.
    obstacle: a stopped vehicle with a warning triangle 8 m behind it, ``event_ahead_m``
    ahead of the ego's start, in the event lane. Vehicles that reach the end of the route are removed.
    """

    PRIORITY = 11      # after map, agent and traffic managers

    def __init__(self):
        super().__init__()
        self.values: dict | None = None
        self.reset_state()

    def reset_state(self):
        self.npc, self.path, self.target_path, self.ego_path = None, None, None, None
        self.s, self.speed, self.trigger_speed, self.blend = 0.0, 0.0, 0.0, 0.0
        self.phase, self.phase_t, self.triggered_at = "idle", 0.0, None

    def before_reset(self):
        super().before_reset()
        self.reset_state()

    def after_reset(self):
        if not self.values:
            return
        ego = next(iter(self.engine.agents.values()))
        network = self.engine.current_map.road_network
        checkpoints = ego.navigation.checkpoints
        ego_lane = ego.lane_index[2]
        lanes_here = len(network.graph[ego.lane_index[0]][ego.lane_index[1]])
        side = {"same": 0, "left": -1, "right": 1}[self.values["event_lane"]]
        lane = ego_lane + side
        if not 0 <= lane < lanes_here:
            lane = ego_lane - side                     # mirror when the ego is already at that edge
        self.ego_path = RoutePath(network, checkpoints, ego_lane)
        self.path = RoutePath(network, checkpoints, lane)
        self.target_path = self.ego_path if self.values["event"] == "cut_in" else self.path
        ego_s = self.ego_path.project(ego.position)
        if self.values["event"] == "obstacle":
            self.phase = "static"
            self.s = min(ego_s + self.values["event_ahead_m"], self.path.length - 10.0)
            heading = self._heading(self.s)
            self.npc = self.spawn_object(DefaultVehicle, vehicle_config={}, position=self._point(self.s), heading=heading)
            lane = self.path.lanes[int(np.searchsorted(self.path.starts, max(self.s - 8.0, 0.0), side="right") - 1)]
            self.spawn_object(TrafficWarning, position=self._point(self.s - 8.0), heading_theta=heading, lane=lane,
                              static=True)
        else:
            self.phase = "escort"
            self.s = ego_s + self.values["event_ahead_m"]
            self.speed = 0.0
            self.npc = self.spawn_object(DefaultVehicle, vehicle_config={}, position=self._point(self.s),
                                         heading=self._heading(self.s))
        self._place()

    def _heading(self, s: float) -> float:
        a, b = self._point(s - 1.0), self._point(s + 1.0)
        return float(math.atan2(b[1] - a[1], b[0] - a[0]))

    def _point(self, s: float) -> np.ndarray:
        p = self.path.position(s)
        if self.target_path is self.path or self.blend <= 0:
            return p
        return (1 - self.blend) * p + self.blend * self.target_path.position(s)

    def _place(self):
        heading = self._heading(self.s)
        self.npc.set_position(self._point(self.s))
        self.npc.set_heading_theta(heading)
        self.npc.set_velocity([math.cos(heading), math.sin(heading)], self.speed)

    def before_step(self, *args, **kwargs):
        if self.npc is None or self.values is None:
            return {}
        if self.phase == "static":
            self._place()
            return {}
        ego = next(iter(self.engine.agents.values()))
        v = self.values
        if self.phase == "escort":
            self.s = self.ego_path.project(ego.position) + v["event_ahead_m"]
            self.speed = float(ego.speed)
            if self.engine.episode_step * DT >= v["event_trigger_s"]:
                self.triggered_at = self.engine.episode_step
                self.trigger_speed = self.speed = max(0.0, float(ego.speed) + v["event_rel_speed_mps"])
                self.phase, self.phase_t = ("brake" if v["event"] == "lead_brake" else "cut_in"), 0.0
        else:
            self.phase_t += DT
            if self.phase == "brake":
                self.speed = max(0.0, self.speed - v["brake_decel_mps2"] * DT)
                if self.speed <= 1e-6:
                    self.phase, self.phase_t = "hold", 0.0
            elif self.phase == "hold" and self.phase_t >= LEAD_RECOVERY_S:
                self.phase = "recover"
            elif self.phase == "recover":
                self.speed = min(self.trigger_speed, self.speed + LEAD_RECOVERY_ACCEL * DT)
            elif self.phase == "cut_in":
                x = min(self.phase_t / v["cut_in_duration_s"], 1.0)
                self.blend = 0.5 - 0.5 * math.cos(math.pi * x)          # smooth lateral profile
            self.s += self.speed * DT
        if self.s >= self.path.length - 1.0:
            self.clear_objects([self.npc.id])
            self.npc = None
            return {}
        self._place()
        return {}

    def after_step(self, *args, **kwargs):
        """Refresh the event vehicle's lane localisation after it is moved.

        MetaDrive's IDM finds the vehicle ahead through each object's ``lane``, which comes from
        the vehicle's navigation module. Vehicles owned by this manager are not stepped by the
        traffic manager, so without this their lane stays at the spawn point and IDM-based
        drivers (references and background traffic) cannot see them.
        """
        if self.npc is not None:
            self.npc.after_step()
        return {}


class AVTestEnv(MetaDriveEnv):
    """MetaDriveEnv configured per episode from a decoded Scenario."""

    def __init__(self, config: dict | None = None):
        super().__init__(dict(BASE_CONFIG, **(config or {})))
        self.scenario: sc.Scenario | None = None
        self._positions: list = []
        self.telemetry: dict = {}

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager("event_manager", EventManager())

    def _is_out_of_road(self, vehicle):
        """Off-road = off every lane, onto the sidewalk, or across the yellow centre line.

        MetaDrive's default also ends the episode on any solid white line. That is a traffic
        violation, not a safety failure, and it makes reference drivers fail on harmless lane
        changes near ramps (docs/LOG.md, 2026-10-31). White-line crossings are counted in telemetry
        (once per step, in ``step``; MetaDrive calls this method from both reward and termination).
        """
        return (not vehicle.on_lane) or vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line

    def load(self, scenario: sc.Scenario):
        v = scenario.values
        self.config["map_config"].update({"type": "block_sequence", "config": v["map"],
                                          "lane_num": v["lane_num"], "lane_width": v["lane_width"]})
        self.config["traffic_density"] = v["traffic_density"]
        self.scenario = scenario

    def reset(self, *args, seed: int | None = None, **kwargs):
        if self.scenario is None:
            raise RuntimeError("load(scenario) before reset")
        self.engine_values()
        obs, info = super().reset(seed=self.scenario.map_seed)
        self._positions = [np.asarray(self.agent.position)]
        self.telemetry = {"min_ttc_s": None, "min_edge_margin_m": None, "white_line_steps": 0}
        return obs, info

    def engine_values(self):
        if self.engine is not None and "event_manager" in self.engine.managers:
            self.engine.managers["event_manager"].values = self.scenario.values

    def lazy_init(self):
        super().lazy_init()
        self.engine.managers["event_manager"].values = self.scenario.values if self.scenario else None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_telemetry()
        if self.agent.on_white_continuous_line:
            self.telemetry["white_line_steps"] += 1
        self._positions.append(np.asarray(self.agent.position))
        stuck = (len(self._positions) > STUCK_WINDOW and not info.get("arrive_dest")
                 and np.linalg.norm(self._positions[-1] - self._positions[-1 - STUCK_WINDOW]) < STUCK_DISTANCE_M)
        if stuck and not terminated:
            terminated = True
        info["stuck"] = bool(stuck)
        info["outcome"] = outcome_from(info, stuck) if (terminated or truncated) else None
        if (terminated or truncated) and info["outcome"] is None:
            info["outcome"] = "timeout"
        if info["outcome"] in ("stuck", "timeout"):        # MetaDrive only penalises crashes and off-road
            reward = -FAILURE_PENALTY
        return obs, reward, terminated, truncated, info

    def _update_telemetry(self):
        ego = self.agent
        t = self.telemetry
        left, right = ego.dist_to_left_side, ego.dist_to_right_side
        if left is not None and right is not None:
            margin = float(min(left, right)) - ego.WIDTH / 2
            t["min_edge_margin_m"] = margin if t["min_edge_margin_m"] is None else min(t["min_edge_margin_m"], margin)
        heading = np.asarray([math.cos(ego.heading_theta), math.sin(ego.heading_theta)])
        normal = np.asarray([-heading[1], heading[0]])
        ego_pos, ego_v = np.asarray(ego.position), float(ego.speed)
        for obj in self.engine.get_objects(lambda o: o is not ego and hasattr(o, "position")).values():
            if not hasattr(obj, "WIDTH"):
                continue
            rel = np.asarray(obj.position)[:2] - ego_pos
            dist = float(np.linalg.norm(rel))
            if dist > TTC_LOOKAHEAD_M:
                continue
            length = getattr(obj, "LENGTH", 1.0)
            x, y = float(rel @ heading), float(rel @ normal)
            if x <= 0 or abs(y) > (ego.WIDTH + getattr(obj, "WIDTH", 1.0)) / 2 + 0.3:
                continue
            other_v = float(np.asarray(getattr(obj, "velocity", (0.0, 0.0)))[:2] @ heading)
            closing = ego_v - other_v
            if closing > 0.1:
                ttc = max(x - (ego.LENGTH + length) / 2, 0.0) / closing
                t["min_ttc_s"] = ttc if t["min_ttc_s"] is None else min(t["min_ttc_s"], ttc)


def run_episode(env: AVTestEnv, scenario: sc.Scenario, driver) -> dict:
    """One episode with an external driver; returns the episode summary (no frames)."""
    import time
    env.load(scenario)
    obs, info = env.reset()
    driver.start(env)
    started, steps, reward = time.time(), 0, 0.0
    while True:
        obs, r, terminated, truncated, info = env.step(driver.act(env, obs))
        reward += r
        steps += 1
        if terminated or truncated:
            break
    return {"driver": driver.name, "outcome": info["outcome"], "steps": steps, "reward": reward,
            "route_completion": float(info.get("route_completion", 0.0)), **env.telemetry,
            "event_triggered_step": env.engine.managers["event_manager"].triggered_at,
            "wall_s": time.time() - started}


class TrainingEnv(gym.Env):
    """Gym wrapper for PPO training: every reset draws θ over the tier and a train-split seed.

    Actions are tactical (lane command, throttle/brake; avtest.tactical); the PID controller turns
    them into MetaDrive's steering and throttle, exactly as SUTDriver does at evaluation time.

    ``set_difficulty(d)`` (0..1) scales traffic density and event intensity down and moves
    the event offset and relative speed towards their easy ends (far ahead, not slower than
    the ego), for the curriculum in scripts/train_driver.py.
    """

    def __init__(self, tier: str, seed: int):
        super().__init__()
        self.tier, self.rng, self.difficulty = tier, np.random.default_rng(seed), 1.0
        from .tactical import TacticalController
        self.env = AVTestEnv()
        self.controller = TacticalController()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space          # same Box(-1, 1, (2,)); meaning: lane command, throttle
        self.last_outcome = None

    def set_difficulty(self, d: float):
        self.difficulty = float(d)

    def reset(self, *, seed=None, options=None):
        theta = sc.sample(self.rng, 1)[0]
        names = [p.name for p in sc.PARAMS]
        theta[names.index("traffic_density")] *= self.difficulty
        theta[names.index("event_intensity")] *= self.difficulty
        for easy_high in ("event_ahead", "event_rel_speed_mps"):   # farther and faster-moving events are easier
            i = names.index(easy_high)
            theta[i] = 1 - self.difficulty * (1 - theta[i])
        self.env.load(sc.decode(theta, seeds.draw(self.rng, "train"), self.tier))
        self.controller.reset()
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.controller.low_level(self.env.agent, action))
        if terminated or truncated:
            self.last_outcome = info["outcome"]
        return obs, reward, terminated, truncated, {k: v for k, v in info.items() if k in ("outcome", "route_completion")}

    def close(self):
        self.env.close()


def training_env_fn(tier: str, rank: int, seed: int):
    def make():
        import torch  # noqa: F401
        return TrainingEnv(tier, seed + rank)
    return make
