"""Tactical action interface: the learned policy decides, a classical controller steers.

Action (both values in [-1, 1], same Box space as before):
    a[0]  lane command: > +0.5 change to the left lane, < -0.5 change to the right lane, otherwise keep
    a[1]  throttle (> 0) or brake (< 0), applied directly
Steering comes from the lane-tracking PID controller of MetaDrive's IDM and LaneChangePolicy
(heading PID 1.7/0.01/3.5, lateral PID 0.3/0.002/0.05), aimed at the commanded lane.

So the learned part chooses speed, braking and lane changes, the decisions the adversarial events
probe, while lane keeping is done by a controller. That is the split used by modular driving
stacks (learned planner, classical tracking controller). Reason: an end-to-end student could not
reproduce the teacher's steering loop precisely enough to stay in lane (docs/LOG.md, 2026-11-01).

Teacher labels: ``LabelledCautiousIDM`` records the lane the cautious IDM steers toward at each step;
``teacher_command`` turns that into left/keep/right relative to the ego's current lane.
"""
from __future__ import annotations

import numpy as np
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.policy.idm_policy import FrontBackObjects
from metadrive.utils.math import wrap_to_pi

from .drivers import CautiousIDMPolicy, CautiousRoadIDMPolicy

LEFT, KEEP, RIGHT = 1.0, 0.0, -1.0
THRESHOLD = 0.5
# The controller of the system under test: part of its identity, so qualification records it in
# QUALIFIED.json and the search runs refuse to start if it has changed since. The 60 km/h generation
# (d4 on) trains and drives with it; drivers d1–d3 (20 km/h, 320 inputs) used LEGACY_CONTROLLER.
# lat_cap: at 60 km/h the uncapped lane-change steering left the road (LOG.md, 2026-11-02).
SUT_CONTROLLER = {"threshold": THRESHOLD, "commit": False, "lat_cap": 1.0}
LEGACY_CONTROLLER = {"threshold": THRESHOLD, "commit": False, "lat_cap": None}


class TacticalController:
    """``threshold``: lane-command magnitude that starts a change. ``commit``: once a change starts, keep
    steering for that lane until the ego is in it, so a command that falls back inside the threshold
    mid-change no longer steers it back; only a command the other way cancels the change (LOG.md,
    2026-11-02: the imitated command hovers near the threshold and flickers). ``lat_cap``: the lateral
    error the steering sees is capped at this many metres, so a lane change drifts over instead of turning
    hard toward the new lane's centre (at 60 km/h the uncapped turn overshot onto the centre line)."""

    def __init__(self, threshold: float = THRESHOLD, commit: bool = False, lat_cap: float | None = None):
        self.threshold, self.commit, self.lat_cap = threshold, commit, lat_cap
        self.reset()

    def reset(self):
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)
        self.committed = None           # (direction, lane index) of a committed change under way

    def target_lane(self, vehicle, command: float):
        navigation = vehicle.navigation
        current = navigation.current_lane
        refs = navigation.current_ref_lanes
        if current not in refs:
            self.committed = None
            return current
        index = refs.index(current)
        direction = 1 if command > self.threshold else -1 if command < -self.threshold else 0     # +1 = left
        if self.committed is not None:
            way, goal = self.committed
            if index == goal or goal >= len(refs):
                self.committed = None                 # arrived (or the lane ended): commands apply again
            elif direction == -way:
                self.committed = None                 # cancelled: back to the lane it is still in
                return current
            else:
                return refs[goal]
        if direction == 0:
            return current
        goal = max(index - 1, 0) if direction > 0 else min(index + 1, len(refs) - 1)
        if self.commit and goal != index:
            self.committed = (direction, goal)
        return refs[goal]

    def low_level(self, vehicle, action) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        lane = self.target_lane(vehicle, action[0])
        long, lat = lane.local_coordinates(vehicle.position)
        if self.lat_cap is not None:
            lat = float(np.clip(lat, -self.lat_cap, self.lat_cap))
        heading_error = -wrap_to_pi(lane.heading_theta_at(long + 1) - vehicle.heading_theta)
        steering = self.heading_pid.get_result(heading_error) + self.lateral_pid.get_result(-lat)
        return np.array([np.clip(steering, -1.0, 1.0), action[1]])


class LabelledCautiousIDM(CautiousIDMPolicy):
    """The cautious-IDM teacher, remembering which lane it steered toward this step."""

    steered_lane = None

    def steering_control(self, target_lane) -> float:
        self.steered_lane = target_lane
        return super().steering_control(target_lane)


class ScenePassingTeacher(LabelledCautiousIDM):
    """The imitation teacher from run d3 on: cautious IDM that passes a stopped or near-stopped object in its
    lane as soon as it is within view (MAX_LONG_DIST) and an adjacent lane is clear, whatever its own speed.

    MetaDrive's IDM overtakes only once its own speed is 3 km/h off target. Under DAgger the student's speed
    then decided the "change lane" label, so d2 learned to slow down first and pass late, with a label that
    flipped at 17 km/h (LOG.md, 2026-11-02). This rule depends only on the scene."""

    BLOCKED_KMH = 5.0
    STATIC_ONLY = False              # pass only static objects (warning triangles), not vehicles that have stopped
    FOLLOW_TRAFFIC = False           # overtake only by the scene: stopped, static, or much slower than NORMAL_SPEED
    PASS_SLOWER_BY_KMH = 20.0        # with FOLLOW_TRAFFIC: pass a vehicle at least this much slower than our cruise
    CLOSING_KMH = 10.0               # ... and at least this much slower than we are (we are closing in on it)

    def move_to_next_road(self):
        """On a new road, follow the lane that continues the current one. MetaDrive takes the first connected
        lane, lane 0, and corrects a step later: a one-step 'change lane' label at every road boundary."""
        lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is not None and self.routing_target_lane not in lanes:
            for lane in lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    self.routing_target_lane = lane
                    return True
        return super().move_to_next_road()

    def lane_change_policy(self, all_objects):
        navigation = self.control_object.navigation
        lanes, next_lanes = navigation.current_ref_lanes, navigation.next_ref_lanes
        if next_lanes is None or len(next_lanes) >= len(lanes):           # lane drops: MetaDrive's merge logic
            near = FrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane,
                                                             self.control_object.position, self.MAX_LONG_DIST, lanes)
            blocker = near.front_object()
            static = isinstance(blocker, TrafficObject)
            slow = (self.FOLLOW_TRAFFIC and blocker is not None
                    and blocker.speed_km_h < self.NORMAL_SPEED - self.PASS_SLOWER_BY_KMH
                    and blocker.speed_km_h < self.control_object.speed_km_h - self.CLOSING_KMH)
            if blocker is not None and (blocker.speed_km_h < self.BLOCKED_KMH or slow) and (static or not self.STATIC_ONLY):
                index = lanes.index(self.routing_target_lane)
                for step, exists, front, back, ahead in (
                        (-1, near.left_lane_exist(), near.front_dist[0], near.back_dist[0], near.front_objs[0]),
                        (1, near.right_lane_exist(), near.front_dist[-1], near.back_dist[-1], near.front_objs[-1])):
                    clear = (exists and front > self.SAFE_LANE_CHANGE_DISTANCE and back > self.SAFE_LANE_CHANGE_DISTANCE
                             and (ahead is None or ahead.speed_km_h >= self.BLOCKED_KMH))
                    if clear:
                        self.target_speed = self.NORMAL_SPEED
                        return ahead, front, lanes[index + step]
            if self.FOLLOW_TRAFFIC:                    # no discretionary overtaking: keep the lane behind it
                self.target_speed = self.NORMAL_SPEED
                return near.front_object(), near.front_min_distance(), self.routing_target_lane
        return super().lane_change_policy(all_objects)


class ScenePassingTeacher60(ScenePassingTeacher, CautiousRoadIDMPolicy):
    """The scene-passing teacher at 60 km/h: curve slowing, 80 m look-ahead and SI gaps from
    avtest.drivers.RoadIDMPolicy (decision 2026-11-02: the driver must cruise at least 60 km/h)."""

    # the 20 km/h cautious values come first in the method resolution order, so the 60 km/h ones are restated
    NORMAL_SPEED = CautiousRoadIDMPolicy.NORMAL_SPEED
    DISTANCE_WANTED = CautiousRoadIDMPolicy.DISTANCE_WANTED
    TIME_WANTED = CautiousRoadIDMPolicy.TIME_WANTED
    MAX_LONG_DIST = CautiousRoadIDMPolicy.MAX_LONG_DIST
    # MetaDrive's IDM also overtakes any slower car whenever its own speed is 3 km/h below target. Under DAgger that
    # is the student's speed: d4, slower than its teacher, got "overtake" labels and changed lanes 4.4 times per
    # scene, often while the teacher said keep (LOG.md, 2026-11-03). The 60 km/h teacher passes only by the scene:
    # a static object, a stopped vehicle, or one it is closing in on that is at least PASS_SLOWER_BY_KMH below its
    # cruise; it follows other traffic. Never passing moving vehicles left it stuck behind leads that resume slowly
    # (79% on T1), and passing only static objects made no measurable difference, so stopped cars count. Without
    # the closing condition it also passed the lead car before its trigger, which drives at the ego's own speed:
    # d5 asked for lane changes on up to 92% of steps in lead-brake scenes (LOG.md, 2026-11-03).
    FOLLOW_TRAFFIC = True


TEACHERS = {"cautious_idm": LabelledCautiousIDM, "scene_passing": ScenePassingTeacher,
            "scene_passing_60": ScenePassingTeacher60}


class TeacherDriver:
    """A teacher driving through its own tactical labels and this controller: the best a perfect student can do."""

    def __init__(self, teacher: str):
        self.name, self.policy_class = f"teacher_{teacher}", TEACHERS[teacher]

    def start(self, env):
        self.policy = self.policy_class(env.agent, env.scenario.map_seed)
        self.controller = TacticalController(**(SUT_CONTROLLER if self.policy_class is ScenePassingTeacher60
                                                 else LEGACY_CONTROLLER))

    def act(self, env, obs):
        _, acceleration = self.policy.act(env.agent.id)
        return self.controller.low_level(env.agent, [teacher_command(self.policy, env.agent),
                                                     float(np.clip(acceleration, -1, 1))])


def teacher_command(policy: LabelledCautiousIDM, vehicle) -> float:
    """Left / keep / right of the teacher's steering target relative to the ego's current lane."""
    navigation = vehicle.navigation
    current, refs, target = navigation.current_lane, navigation.current_ref_lanes, policy.steered_lane
    if target is None or target is current or current not in refs or target not in refs:
        return KEEP
    return LEFT if refs.index(target) < refs.index(current) else RIGHT
