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
# From d7 on, the controller a driver was trained with is written into its model file (CONTROLLER_ENTRY), so it
# travels with every copy (models/sut.zip) and is covered by the model hash. HOLD_CONTROLLER (hold_lane) was tried
# and made the teacher worse; it remains an option (pretrain_bc --controller hold).
HOLD_CONTROLLER = {"threshold": THRESHOLD, "commit": False, "lat_cap": 1.0, "hold_lane": True}
# SUT_CONTROLLER with throttle smoothing: 42% less throttle change per step at d6's success (LOG.md, 2026-11-03)
SMOOTH_CONTROLLER = {**SUT_CONTROLLER, "smooth": 0.5}
# d10 on: trained and driven with the emergency-braking layer and the 0.3 lane threshold that suited d8/d9;
# d9 + this layer at 2.5 s reached 94.4% on the T1 dev panel with 1.4% crashes (LOG.md, 2026-11-03)
AEB_CONTROLLER = {**SMOOTH_CONTROLLER, "threshold": 0.3, "aeb_ttc": 2.5}
# d11 on: AEB_CONTROLLER plus the lane-change guard and speed-scaled steering along the route; d10 driven with it
# reached 97.6% on the T1 dev panel with no off-road (LOG.md, 2026-11-03)
GUARD_CONTROLLER = {**AEB_CONTROLLER, "lc_guard_m": 70.0, "route_steer": True, "steer_ref_kmh": 40.0}
CONTROLLER_ENTRY = "avtest_controller.json"


def stamp_controller(model_path, settings: dict) -> None:
    """Record the controller a freshly saved model was trained with inside its zip."""
    import json
    import zipfile
    with zipfile.ZipFile(model_path, "a") as archive:
        if CONTROLLER_ENTRY in archive.namelist():
            raise ValueError(f"{model_path} already records a controller")
        archive.writestr(CONTROLLER_ENTRY, json.dumps(settings, sort_keys=True))


def copy_with_controller(source, destination, settings: dict) -> None:
    """A copy of a model file recording other controller settings, e.g. a decision threshold tuned on a failed
    qualification panel after training (the model itself is unchanged; the copy has its own hash)."""
    import zipfile
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "w") as new:
        for item in old.infolist():
            if item.filename != CONTROLLER_ENTRY:
                new.writestr(item, old.read(item.filename))
    stamp_controller(destination, settings)


def controller_for_model(model_path, obs_size: int) -> dict:
    """The controller a driver drives with: the one recorded in its file, else its generation's (d1-d3 legacy,
    d4-d6 SUT_CONTROLLER)."""
    import json
    import zipfile
    from .sim import OBS_DIM
    with zipfile.ZipFile(model_path) as archive:
        if CONTROLLER_ENTRY in archive.namelist():
            return json.loads(archive.read(CONTROLLER_ENTRY))
    return SUT_CONTROLLER if obs_size == OBS_DIM else LEGACY_CONTROLLER


class TacticalController:
    """``threshold``: lane-command magnitude that starts a change. ``commit``: once a change starts, keep
    steering for that lane until the ego is in it, so a command that falls back inside the threshold
    mid-change no longer steers it back; only a command the other way cancels the change (LOG.md,
    2026-11-02: the imitated command hovers near the threshold and flickers). ``lat_cap``: the lateral
    error the steering sees is capped at this many metres, so a lane change drifts over instead of turning
    hard toward the new lane's centre (at 60 km/h the uncapped turn overshot onto the centre line).

    ``hold_lane`` (from d7): the controller holds a lane of its own instead of steering relative to the lane
    the ego's centre is in. "Keep" means stay in the held lane, on whichever side of a line the centre is; a
    change starts only when the command newly passes the threshold (a held command does not chain a second
    change after arrival); the opposite command cancels it. With commands relative to the current lane, a car
    straddling a line flipped its target with every wobble, and DAgger taught d6 to oscillate there: 5.8 lane
    changes per scene against its teacher's 1.1 (LOG.md, 2026-11-03)."""

    def __init__(self, threshold: float = THRESHOLD, commit: bool = False, lat_cap: float | None = None,
                 hold_lane: bool = False, smooth: float | None = None, debounce: int = 0,
                 aeb_ttc: float | None = None, lc_guard_m: float | None = None, route_steer: bool = False,
                 steer_ref_kmh: float | None = None):
        """``smooth``: throttle/brake becomes smooth * previous + (1 - smooth) * command. ``debounce``: a new lane
        direction (left/keep/right) applies only after it has been commanded for this many steps in a row.
        Both against d6's visible vibration (LOG.md, 2026-11-03); off by default.

        ``aeb_ttc``: an emergency-braking layer, as in production cars, independent of the learned planner. For
        every detected object (vehicles and warning triangles, from the ego's lidar) that overlaps the ego's
        path, measured along its route so curves are handled, it computes the time to collision; below
        aeb_ttc seconds, or with anything closer than AEB_HOLD_GAP_M in its path, it brakes fully. Steering and
        lane choice stay with the planner. (A first version also braked half below twice aeb_ttc: braking and
        the planner's throttle then alternated every step and the car crept into the obstacle.) Six 60 km/h students plateaued at 87-91% on T1, and most of their collisions came without any
        braking (LOG.md, 2026-11-03).

        ``lc_guard_m``: a lane-change guard, the counterpart of the braking layer for lane choice. A commanded
        change is replaced by "keep" while the target lane holds a stopped vehicle or a static object within this
        many metres ahead (``stopped_object_ahead``), unless the ego's own lane is blocked nearer. In 10 of
        d10's 24 failures on the T1 dev panel it moved into the obstacle's lane 14-51 m before the obstacle,
        against the teacher's label, and was then held behind it by the braking layer (LOG.md, 2026-11-03).

        ``route_steer``: the steering's lateral offset and its heading target (1 m ahead) are taken along the
        route in the target lane (sim.RoutePath) instead of on the current road segment. MetaDrive moves the ego
        onto the next segment only after it has passed the end of the current one, so at a curve's end the target
        followed the arc's continuation and then jumped by about twice the arc's curvature: a steering kick
        (+0.22) and a lightly damped 2 Hz wobble for 2-3 s after every curve (LOG.md, 2026-11-03).

        ``steer_ref_kmh``: above this speed the steering is scaled by steer_ref_kmh / speed. The PID gains are
        MetaDrive's IDM's, whose traffic drives at about 30 km/h; the yaw a steering input produces grows with
        speed, and at 45-60 km/h the loop oscillates at about 2.5 Hz (LOG.md, 2026-11-03)."""
        self.threshold, self.commit, self.lat_cap, self.hold_lane = threshold, commit, lat_cap, hold_lane
        self.smooth, self.debounce, self.aeb_ttc, self.lc_guard_m = smooth, debounce, aeb_ttc, lc_guard_m
        self.route_steer, self.steer_ref_kmh = route_steer, steer_ref_kmh
        self.reset()

    def reset(self):
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)
        self.committed = None           # (direction, lane index) of a committed change under way
        self.home, self.goal, self.last_direction = None, None, 0       # hold_lane state
        self.throttle, self.pending, self.pending_steps, self.effective = 0.0, 0, 0, 0   # smooth / debounce state
        self.guarding = False           # lc_guard_m: the last command was replaced by "keep"
        self.routes = {}                # route_steer: RoutePath per lane index, for this episode's route

    def target_lane(self, vehicle, command: float):
        navigation = vehicle.navigation
        current = navigation.current_lane
        refs = navigation.current_ref_lanes
        if current not in refs:
            self.committed = None
            return current
        index = refs.index(current)
        direction = 1 if command > self.threshold else -1 if command < -self.threshold else 0     # +1 = left
        if self.hold_lane:
            return refs[self._held_target(index, len(refs), direction)]
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

    def _held_target(self, index: int, lanes: int, direction: int) -> int:
        """Lane index the hold_lane controller steers for: the lane it is changing to, else its held lane."""
        self.home = index if self.home is None else min(self.home, lanes - 1)
        onset = direction != 0 and direction != self.last_direction
        self.last_direction = direction
        if self.goal is not None:
            way, goal = self.goal
            if goal >= lanes:
                self.goal = None
            elif index == goal:
                self.home, self.goal = goal, None             # arrived: this is the held lane now
            elif onset and direction == -way:
                self.goal = None                              # cancelled: back to the held lane
                return self.home
        if onset and self.goal is None:
            goal = max(self.home - 1, 0) if direction > 0 else min(self.home + 1, lanes - 1)
            if goal != self.home:
                self.goal = (direction, goal)
        return self.goal[1] if self.goal is not None else self.home

    AEB_HOLD_GAP_M = 1.5

    @staticmethod
    def time_to_collision(vehicle, lookahead_m: float = 40.0, margin_m: float = 0.3) -> tuple[float, float]:
        """(seconds until the ego reaches the nearest detected object overlapping its path at the current
        closing speed, bumper gap to the nearest such object), inf if none. Positions are taken along the ego's
        route with the index of its current lane (sim.RoutePath), so on a curve an object in the lane ahead counts
        and one in the next lane does not, also across road segments. Taken along the current lane alone, a stopped
        car in the outer lane of a curve lay on the straight lane's extension, and the car braked to a stop beside
        it (LOG.md, 2026-11-03). Off its route, the ego measures along its lane."""
        from .sim import RoutePath
        speed = vehicle.speed_km_h / 3.6
        lane = vehicle.lane
        path = RoutePath(vehicle.engine.current_map.road_network, vehicle.navigation.checkpoints, lane.index[-1])
        where = path.local if np.isfinite(path.local(vehicle.position)[1]) else lane.local_coordinates
        ego_long, ego_lat = where(vehicle.position)
        heading = np.array([np.cos(vehicle.heading_theta), np.sin(vehicle.heading_theta)])
        best, nearest = float("inf"), float("inf")
        for obj in vehicle.lidar.get_surrounding_objects(vehicle):
            if obj is vehicle or not hasattr(obj, "position"):
                continue
            long, lat = where(obj.position)
            gap = long - ego_long - (vehicle.LENGTH + getattr(obj, "LENGTH", 1.0)) / 2
            if not 0.0 < gap < lookahead_m:
                continue
            if abs(lat - ego_lat) > (vehicle.WIDTH + getattr(obj, "WIDTH", 1.0)) / 2 + margin_m:
                continue
            nearest = min(nearest, gap)
            closing = speed - float(np.asarray(getattr(obj, "velocity", (0.0, 0.0)))[:2] @ heading)
            if closing > 0.1:
                best = min(best, gap / closing)
        return best, nearest

    GUARD_STOPPED_KMH = 5.0

    @classmethod
    def stopped_object_ahead(cls, vehicle, lane, max_m: float) -> float:
        """Metres along the route from the ego to the nearest stopped vehicle (< GUARD_STOPPED_KMH) or static
        object (warning triangle) in ``lane`` and the lanes with its index further along the route; inf if none
        within max_m. An object up to one car length behind the ego still counts, so a change back is not started
        beside it. Objects come from the ego's lidar and are placed by the lane they are in, as MetaDrive's IDM
        places them, but over any number of road segments (IDM looks one segment ahead)."""
        from .sim import RoutePath
        path = RoutePath(vehicle.engine.current_map.road_network, vehicle.navigation.checkpoints, lane.index[-1])
        here = next((k for k, l in enumerate(path.lanes) if l is lane), None)
        if here is None:
            return float("inf")
        ego_s = path.starts[here] + lane.local_coordinates(vehicle.position)[0]
        nearest = float("inf")
        for obj in vehicle.lidar.get_surrounding_objects(vehicle, radius=int(max_m) + 10):
            obj_lane = getattr(obj, "lane", None)
            if obj is vehicle or obj_lane is None:
                continue
            if not isinstance(obj, TrafficObject) and getattr(obj, "speed_km_h", 0.0) >= cls.GUARD_STOPPED_KMH:
                continue
            k = next((k for k, l in enumerate(path.lanes) if l is obj_lane), None)
            if k is None:
                continue
            ahead = path.starts[k] + obj_lane.local_coordinates(obj.position)[0] - ego_s
            if -vehicle.LENGTH < ahead < max_m:
                nearest = min(nearest, ahead)
        return nearest

    def guard(self, vehicle, command: float) -> float:
        """lc_guard_m: the command, or "keep" if it would start a change into a lane blocked nearer than the
        ego's own (see __init__)."""
        navigation = vehicle.navigation
        current, refs = navigation.current_lane, navigation.current_ref_lanes
        direction = 1 if command > self.threshold else -1 if command < -self.threshold else 0     # +1 = left
        self.guarding = False
        if direction == 0 or current not in refs:
            return command
        goal = refs.index(current) - direction
        if not 0 <= goal < len(refs):
            return command
        if (self.stopped_object_ahead(vehicle, refs[goal], self.lc_guard_m)
                < self.stopped_object_ahead(vehicle, current, self.lc_guard_m)):
            self.guarding = True
            return KEEP
        return command

    def low_level(self, vehicle, action) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        if self.lc_guard_m is not None:
            action[0] = self.guard(vehicle, action[0])
        if self.debounce:
            raw = 1 if action[0] > self.threshold else -1 if action[0] < -self.threshold else 0
            self.pending_steps = self.pending_steps + 1 if raw == self.pending else 1
            self.pending = raw
            if self.pending_steps >= self.debounce:
                self.effective = raw
            action[0] = float(self.effective)
        if self.smooth is not None:
            self.throttle = action[1] = self.smooth * self.throttle + (1 - self.smooth) * action[1]
        if self.aeb_ttc is not None:
            ttc, gap = self.time_to_collision(vehicle)
            if ttc < self.aeb_ttc or gap < self.AEB_HOLD_GAP_M:
                action[1] = -1.0                       # brake fully; hold it while something is right ahead
            self.last_ttc, self.last_gap = ttc, gap
        lane = self.last_target = self.target_lane(vehicle, action[0])
        long, lat = lane.local_coordinates(vehicle.position)
        target_heading = lane.heading_theta_at(long + 1)
        if self.route_steer:
            from .sim import RoutePath
            key = (id(vehicle.engine.current_map), lane.index[-1])
            if key not in self.routes:
                self.routes[key] = RoutePath(vehicle.engine.current_map.road_network, vehicle.navigation.checkpoints,
                                             lane.index[-1])
            s, route_lat = self.routes[key].local(vehicle.position)
            if np.isfinite(route_lat):                 # on its route; else as before, on the current segment
                lat, target_heading = route_lat, self.routes[key].heading(s + 1)
        if self.lat_cap is not None:
            lat = float(np.clip(lat, -self.lat_cap, self.lat_cap))
        heading_error = -wrap_to_pi(target_heading - vehicle.heading_theta)
        steering = self.heading_pid.get_result(heading_error) + self.lateral_pid.get_result(-lat)
        if self.steer_ref_kmh is not None and vehicle.speed_km_h > self.steer_ref_kmh:
            steering *= self.steer_ref_kmh / vehicle.speed_km_h
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

    def __init__(self, teacher: str, controller: dict | None = None):
        self.name, self.policy_class = f"teacher_{teacher}", TEACHERS[teacher]
        # by default the controller its students train with: SUT_CONTROLLER for the 60 km/h teacher, else legacy
        self.controller_settings = controller or (SUT_CONTROLLER if self.policy_class is ScenePassingTeacher60
                                                  else LEGACY_CONTROLLER)

    def start(self, env):
        self.policy = self.policy_class(env.agent, env.scenario.map_seed)
        self.controller = TacticalController(**self.controller_settings)

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
