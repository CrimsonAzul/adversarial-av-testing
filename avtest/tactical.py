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
from metadrive.utils.math import wrap_to_pi

from .drivers import CautiousIDMPolicy

LEFT, KEEP, RIGHT = 1.0, 0.0, -1.0
THRESHOLD = 0.5
# The controller of the system under test: part of its identity, so qualification records it in
# QUALIFIED.json and the search runs refuse to start if it has changed since. Training keeps the
# defaults (THRESHOLD, no commitment).
SUT_CONTROLLER = {"threshold": THRESHOLD, "commit": False}


class TacticalController:
    """``threshold``: lane-command magnitude that starts a change. ``commit``: once a change starts, keep
    steering for that lane until the ego is in it, so a command that falls back inside the threshold
    mid-change no longer steers it back; only a command the other way cancels the change (LOG.md,
    2026-11-02: the imitated command hovers near the threshold and flickers)."""

    def __init__(self, threshold: float = THRESHOLD, commit: bool = False):
        self.threshold, self.commit = threshold, commit
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
        heading_error = -wrap_to_pi(lane.heading_theta_at(long + 1) - vehicle.heading_theta)
        steering = self.heading_pid.get_result(heading_error) + self.lateral_pid.get_result(-lat)
        return np.array([np.clip(steering, -1.0, 1.0), action[1]])


class LabelledCautiousIDM(CautiousIDMPolicy):
    """The cautious-IDM teacher, remembering which lane it steered toward this step."""

    steered_lane = None

    def steering_control(self, target_lane) -> float:
        self.steered_lane = target_lane
        return super().steering_control(target_lane)


def teacher_command(policy: LabelledCautiousIDM, vehicle) -> float:
    """Left / keep / right of the teacher's steering target relative to the ego's current lane."""
    navigation = vehicle.navigation
    current, refs, target = navigation.current_lane, navigation.current_ref_lanes, policy.steered_lane
    if target is None or target is current or current not in refs or target not in refs:
        return KEEP
    return LEFT if refs.index(target) < refs.index(current) else RIGHT
