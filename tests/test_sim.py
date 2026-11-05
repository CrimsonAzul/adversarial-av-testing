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


def test_reward_makes_safety_dominate(env):
    """Run d1's failure mode: speed paid more than safety. A safe success must out-score any failure."""
    from avtest.drivers import CautiousIDMDriver
    from avtest.sim import FAILURE_PENALTY, OBS_DIM

    class Fixed:
        def __init__(self, name, action):
            self.name, self.action = name, action

        def start(self, env):
            pass

        def act(self, env, obs):
            return self.action

    returns = {}
    s = scene("lead_brake")
    for driver in (CautiousIDMDriver(), Fixed("reckless", [0.0, 1.0]), Fixed("frozen", [0.0, -1.0])):
        env.load(s)
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)
        driver.start(env)
        total = 0.0
        while True:
            obs, r, term, trunc, info = env.step(driver.act(env, obs))
            total += r
            if term or trunc:
                break
        returns[driver.name] = (info["outcome"], total)
    assert returns["idm_cautious"][0] == "success" and returns["frozen"][0] == "stuck"
    assert returns["reckless"][0] in ("collision", "offroad")
    assert returns["idm_cautious"][1] > 30 > 0 > returns["reckless"][1]
    assert returns["frozen"][1] == pytest.approx(-FAILURE_PENALTY, abs=0.5)


def test_controller_keeps_lane_and_executes_lane_changes(env):
    """Tactical interface: 'keep' holds the lane centre on an empty road; 'left' moves one lane left."""
    from avtest.tactical import LEFT, KEEP, TacticalController
    names = [p.name for p in sc.PARAMS]
    theta = sc.with_event(np.random.default_rng(8).random(sc.DIM), "obstacle", "T2")
    theta[names.index("traffic_density")] = 0.0
    theta[names.index("lane_num")] = 0.9                       # 3 lanes
    theta[names.index("event_lane")] = 0.99
    s = sc.decode(theta, SEED, "T2")
    env.load(s)
    obs, _ = env.reset()
    controller = TacticalController()
    start_lane = env.agent.lane_index[2]
    lateral = []
    for step in range(300):
        command = LEFT if 150 <= step < 190 and start_lane > 0 else KEEP     # held until the change completes
        throttle = float(np.clip(0.5 * (30 - env.agent.speed_km_h) / 3.6, -1, 1))   # hold about 30 km/h
        obs, _, term, trunc, info = env.step(controller.low_level(env.agent, [command, throttle]))
        if step < 150:
            lateral.append(abs(env.agent.lane.local_coordinates(env.agent.position)[1]))
        if term or trunc:
            break
    assert info.get("outcome") in (None, "success")
    assert max(lateral[50:]) < 0.3                           # centred once up to speed
    if start_lane > 0:
        assert env.agent.lane_index[2] == start_lane - 1


def test_recycled_obstacle_objects_start_at_rest(env):
    """MetaDrive recycles objects between episodes; a warning triangle hit earlier kept ~5 km/h, which IDM-based
    drivers read, so results depended on what the worker ran before (LOG.md, 2026-11-02)."""
    from metadrive.component.static_object.traffic_object import TrafficWarning
    from avtest.tactical import KEEP, TacticalController
    env.load(scene("obstacle", event_lane=0.0, event_ahead=0.0, traffic_density=0.0))
    env.reset()
    controller = TacticalController()
    while True:                                     # keep the lane at full throttle: straight into the triangle
        _, _, term, trunc, info = env.step(controller.low_level(env.agent, [KEEP, 1.0]))
        if term or trunc:
            break
    assert info["crash_object"]
    env.load(scene("obstacle", event_lane=0.0, event_ahead=0.6, traffic_density=0.0))
    env.reset()
    warnings = list(env.engine.get_objects(lambda o: isinstance(o, TrafficWarning)).values())
    assert warnings and all(w.speed_km_h == 0.0 for w in warnings)


def test_controller_threshold_and_commitment():
    """Which lane the controller steers for, on a fake 3-lane road (index 0 = leftmost)."""
    from types import SimpleNamespace
    from avtest.tactical import TacticalController
    lanes = ["L", "M", "R"]
    car = SimpleNamespace(navigation=SimpleNamespace(current_lane="M", current_ref_lanes=lanes))

    default = TacticalController()
    assert default.target_lane(car, -0.3) == "M" and default.target_lane(car, -0.6) == "R"
    assert default.target_lane(car, 0.0) == "M"                  # without commitment a dropped command steers back
    assert TacticalController(threshold=0.25).target_lane(car, -0.3) == "R"

    committed = TacticalController(commit=True)
    assert committed.target_lane(car, 0.7) == "L"
    assert committed.target_lane(car, 0.0) == "L"                # still in M: the change goes on
    assert committed.target_lane(car, 0.4) == "L"
    assert committed.target_lane(car, -0.7) == "M" and committed.committed is None   # the other way cancels
    assert committed.target_lane(car, 0.0) == "M"
    assert committed.target_lane(car, 0.7) == "L"
    car.navigation.current_lane = "L"                            # arrived
    assert committed.target_lane(car, 0.0) == "L" and committed.committed is None
    car.navigation.current_lane = "R"
    assert committed.target_lane(car, -0.9) == "R" and committed.committed is None   # no lane further right
    committed.reset()
    assert committed.committed is None


def test_teacher_through_tactical_labels_drives_like_itself(env):
    from avtest.drivers import CautiousIDMDriver
    from avtest.tactical import LabelledCautiousIDM, TacticalController, teacher_command
    s = scene("lead_brake")
    full = run_episode(env, s, CautiousIDMDriver())
    env.load(s)
    obs, _ = env.reset()
    teacher, controller = LabelledCautiousIDM(env.agent, s.map_seed), TacticalController()
    while True:
        _, acceleration = teacher.act(env.agent.id)
        label = [teacher_command(teacher, env.agent), float(np.clip(acceleration, -1, 1))]
        obs, _, term, trunc, info = env.step(controller.low_level(env.agent, label))
        if term or trunc:
            break
    assert full["outcome"] == "success" and info["outcome"] == "success"


def test_scene_passing_teacher_moves_over_as_soon_as_it_sees_the_obstacle(env):
    """The d3 teacher labels 'change lane' when the warning triangle (8 m before the stopped car) enters its
    30 m view, about 38 m before the car, not after it has slowed down (LOG.md, 2026-11-02)."""
    from avtest.tactical import TeacherDriver, teacher_command
    env.load(scene("obstacle", event_lane=0.0, event_ahead=0.6, traffic_density=0.0))
    obs, _ = env.reset()
    driver = TeacherDriver("scene_passing")
    driver.start(env)
    events = env.engine.managers["event_manager"]
    first_change = None
    while True:
        obs, _, term, trunc, info = env.step(driver.act(env, obs))
        if first_change is None and teacher_command(driver.policy, env.agent) != 0:
            first_change = events.s - events.ego_path.project(env.agent.position)
        if term or trunc:
            break
    assert info["outcome"] == "success"
    assert 30 < first_change < 40


def test_60_kmh_reference_cruises_and_slows_for_curves(env):
    """RoadIDMPolicy: about 60 km/h where the road allows, never faster in a curve than its 0.3 g limit."""
    import math
    from metadrive.component.lane.circular_lane import CircularLane
    from avtest.drivers import RoadIDMDriver, RoadIDMPolicy
    names = [p.name for p in sc.PARAMS]
    theta = sc.with_event(np.random.default_rng(3).random(sc.DIM), "obstacle", "T1")
    theta[names.index("traffic_density")] = 0.0
    theta[names.index("event_lane")] = 0.99                  # the obstacle is in another lane
    for k in range(40):                                      # a map with at least one curve
        s = sc.decode(theta, SEED + k, "T1")
        if "C" in s.values["map"]:
            break
    env.load(s)
    obs, _ = env.reset()
    driver = RoadIDMDriver()
    driver.start(env)
    top, curve_excess = 0.0, []
    while True:
        obs, _, term, trunc, info = env.step(driver.act(env, obs))
        speed, lane = env.agent.speed_km_h, env.agent.lane
        top = max(top, speed)
        if isinstance(lane, CircularLane):
            curve_excess.append(speed - math.sqrt(RoadIDMPolicy.A_LAT * lane.radius) * 3.6)
        if term or trunc:
            break
    assert info["outcome"] == "success"
    assert 55 < top <= 63
    assert curve_excess and max(curve_excess) < 5          # within 5 km/h of the curve limit (it brakes gently)


def test_60_kmh_teacher_moves_over_from_80_m(env):
    """At 60 km/h the teacher looks 80 m ahead: it starts passing when the triangle is about 80 m away."""
    from avtest.tactical import TeacherDriver, teacher_command
    env.load(scene("obstacle", event_lane=0.0, event_ahead=1.0, traffic_density=0.0))
    obs, _ = env.reset()
    driver = TeacherDriver("scene_passing_60")
    driver.start(env)
    events = env.engine.managers["event_manager"]
    first_change = None
    while True:
        obs, _, term, trunc, info = env.step(driver.act(env, obs))
        if first_change is None and teacher_command(driver.policy, env.agent) != 0:
            first_change = events.s - events.ego_path.project(env.agent.position)
        if term or trunc:
            break
    assert info["outcome"] == "success"
    assert 70 < first_change < 95


def test_reused_environment_does_not_leak_between_scenarios(env):
    """Evaluator workers reuse one environment for many scenarios; results must not depend on run order."""
    from avtest.drivers import CautiousIDMDriver
    a = sc.decode(np.random.default_rng(5).random(sc.DIM), SEED + 300, "T2")
    first = run_episode(env, a, CautiousIDMDriver())
    rng = np.random.default_rng(9)
    for k, tier in enumerate(("T3", "T1")):
        run_episode(env, sc.decode(rng.random(sc.DIM), SEED + 400 + k, tier), IDMDriver())
    again = run_episode(env, a, CautiousIDMDriver())
    assert (first["outcome"], first["steps"], first["event_triggered_step"]) == \
        (again["outcome"], again["steps"], again["event_triggered_step"])


def test_60_kmh_lane_change_stays_on_the_road(env):
    """With the SUT controller's 1 m lateral cap, lane changes at 60 km/h right and then left (towards the centre
    line) stay on the road; MetaDrive's uncapped lane-tracking left the road (LOG.md, 2026-11-02)."""
    from avtest.tactical import KEEP, LEFT, RIGHT, SUT_CONTROLLER, TacticalController
    names = [p.name for p in sc.PARAMS]
    theta = sc.with_event(np.random.default_rng(8).random(sc.DIM), "obstacle", "T1")
    for name, value in (("traffic_density", 0.0), ("lane_num", 0.9), ("event_lane", 0.99), ("event_ahead", 1.0),
                        ("block_1", 0.0), ("block_2", 0.0), ("block_3", 0.0)):
        theta[names.index(name)] = value
    env.load(sc.decode(theta, 100_005, "T1"))                 # three straight blocks, 3 lanes
    env.reset()
    controller = TacticalController(**SUT_CONTROLLER)
    plan, start = [], env.agent.lane_index[2]
    while True:
        lane = env.agent.lane_index[2]
        if not plan and env.agent.speed_km_h > 58:
            plan = [(RIGHT, start + 1), (LEFT, start)]
        command = KEEP
        if plan and lane != plan[0][1]:
            command = plan[0][0]
        elif plan and len(plan) == 2:
            plan.pop(0) if env.agent.lane.local_coordinates(env.agent.position)[1] ** 2 < 0.04 else None
        throttle = float(np.clip(0.5 * (60 - env.agent.speed_km_h) / 3.6, -1, 1))
        _, _, term, trunc, info = env.step(controller.low_level(env.agent, [command, throttle]))
        if term or trunc:
            break
    assert info["outcome"] == "success"
    assert len(plan) == 1 and env.agent.lane_index[2] == start     # both changes happened


def test_background_traffic_cruises_at_road_speed(env):
    """Background traffic is TrafficIDMPolicy at 50-60 km/h (sim.ROAD_TRAFFIC), not MetaDrive's 30 km/h IDM."""
    import collections
    from avtest.drivers import RoadIDMDriver, TrafficIDMPolicy
    names = [p.name for p in sc.PARAMS]
    theta = sc.with_event(np.random.default_rng(4).random(sc.DIM), "obstacle", "T1")
    for name, value in (("traffic_density", 1.0), ("block_1", 0.0), ("block_2", 0.0), ("block_3", 0.0)):
        theta[names.index(name)] = value
    env.load(sc.decode(theta, 100_321, "T1"))                  # straight road, the tier's densest traffic
    obs, _ = env.reset()
    driver = RoadIDMDriver()
    driver.start(env)
    top, policies = collections.defaultdict(float), set()
    while True:
        obs, _, term, trunc, info = env.step(driver.act(env, obs))
        for v in env.engine.traffic_manager._traffic_vehicles:
            policies.add(type(env.engine.get_policy(v.name)).__name__)
            top[v.name] = max(top[v.name], v.speed_km_h)
        if term or trunc:
            break
    assert policies == {"TrafficIDMPolicy"}
    moving = [s for s in top.values() if s > 5]
    assert moving and all(s <= TrafficIDMPolicy.SPEED_RANGE_KMH[1] + 3 for s in moving)
    assert max(moving) > 40                                    # well above MetaDrive's 30 km/h


def test_hold_lane_controller():
    """hold_lane: keep means the held lane; a change starts on a fresh command and does not chain."""
    from types import SimpleNamespace
    from avtest.tactical import TacticalController
    lanes = ["L", "M", "R"]
    nav = SimpleNamespace(current_lane="M", current_ref_lanes=lanes)
    car = SimpleNamespace(navigation=nav)
    c = TacticalController(hold_lane=True)
    assert c.target_lane(car, 0.0) == "M"
    nav.current_lane = "R"                                   # straddling a line: the centre wobbles into R
    assert c.target_lane(car, 0.0) == "M"                    # ... but keep means the held lane
    nav.current_lane = "M"
    assert c.target_lane(car, 0.7) == "L"                    # fresh command: change left
    assert c.target_lane(car, 0.7) == "L" and c.target_lane(car, 0.0) == "L"   # held or released: goes on
    nav.current_lane = "L"                                   # arrived
    assert c.target_lane(car, 0.0) == "L" and c.home == 0

    c = TacticalController(hold_lane=True)
    nav.current_lane = "R"
    assert c.target_lane(car, 0.9) == "M"
    nav.current_lane = "M"                                   # arrives while the command is still held
    assert c.target_lane(car, 0.9) == "M" and c.target_lane(car, 0.9) == "M"   # no second change to L
    assert c.target_lane(car, 0.0) == "M"
    assert c.target_lane(car, 0.9) == "L"                    # released, then pressed again: a new change

    c = TacticalController(hold_lane=True)
    nav.current_lane = "M"
    assert c.target_lane(car, 0.8) == "L"
    assert c.target_lane(car, -0.8) == "M"                   # the other way cancels: back to the held lane ...
    assert c.target_lane(car, -0.8) == "M"                   # ... without starting a change to the right
    assert c.target_lane(car, 0.0) == "M" and c.target_lane(car, -0.8) == "R"
    c.reset()
    assert c.home is None and c.goal is None


def test_controller_smoothing_and_debounce():
    from types import SimpleNamespace
    import numpy as np
    from avtest.tactical import TacticalController
    lanes = SimpleNamespace(heading_theta_at=lambda s: 0.0, local_coordinates=lambda p: (0.0, 0.0))
    nav = SimpleNamespace(current_lane=lanes, current_ref_lanes=[lanes])
    car = SimpleNamespace(navigation=nav, position=(0, 0), heading_theta=0.0)
    c = TacticalController(smooth=0.5)
    assert c.low_level(car, [0.0, 1.0])[1] == 0.5 and c.low_level(car, [0.0, 1.0])[1] == 0.75
    seen = []
    c = TacticalController(debounce=3)
    c.target_lane = lambda vehicle, command: seen.append(command) or lanes
    for command in (0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9):
        c.low_level(car, [command, 0.0])
    assert seen == [0, 0, 0, 0, 0, 1, 1]                       # a flicker is ignored; 3 steps in a row apply



def test_emergency_braking_stops_short_of_the_triangle(env):
    """aeb_ttc: lane kept at full throttle straight at a warning triangle; without the layer the car hits it,
    with it the car stops in front of it."""
    from avtest.tactical import KEEP, SUT_CONTROLLER, TacticalController
    outcomes = {}
    for name, extra in (("without", {}), ("with", {"aeb_ttc": 1.5})):
        env.load(scene("obstacle", event_lane=0.0, event_ahead=0.3, traffic_density=0.0))
        env.reset()
        controller = TacticalController(**SUT_CONTROLLER, **extra)
        for _ in range(400):
            _, _, term, trunc, info = env.step(controller.low_level(env.agent, [KEEP, 1.0]))
            if term or trunc:
                break
        outcomes[name] = info["outcome"] if term or trunc else "still driving"
    assert outcomes["without"] == "collision"
    assert outcomes["with"] in ("stuck", "still driving")          # it stopped and stays stopped: no collision


def test_event_objects_report_their_scripted_speed(env):
    """A background car that clips the warning triangle left it in place but reporting up to 45 km/h, which the
    braking layer and IDM-based drivers read (LOG.md, 2026-11-03): after every step the triangle reports 0 and
    the stopped car its scripted speed, whatever the physics gave them."""
    env.load(scene("obstacle", event_lane=0.0, event_ahead=0.5, traffic_density=0.0))
    env.reset()
    em = env.engine.managers["event_manager"]
    where = np.asarray(em.warning.position)
    for obj in (em.warning, em.npc):
        obj.set_velocity([1.0, 0.0], 10.0)                     # as a clipping car leaves them
    env.step([0.0, 0.0])
    assert em.warning.speed_km_h == 0.0 and em.npc.speed_km_h < 0.1
    assert np.allclose(em.warning.position, where)


def test_lane_change_guard_keeps_out_of_the_obstacle_lane(env):
    """lc_guard_m: commanded at full lock towards the lane of an obstacle 52 m ahead, the car moves into it and
    ends behind it without the guard; with it, it stays in its lane until it is past, then changes."""
    from avtest.tactical import LEFT, AEB_CONTROLLER, TacticalController
    seen = {}
    for name, extra in (("without", {}), ("with", {"lc_guard_m": 70.0})):
        env.load(scene("obstacle", event_lane=0.5, event_ahead=0.1, traffic_density=0.0, lane_num=0.0))
        env.reset()
        em = env.engine.managers["event_manager"]
        start = env.agent.lane_index[2]
        assert em.path.lanes[0].index[-1] == start - 1           # the obstacle is in the lane to the left
        controller = TacticalController(**{**AEB_CONTROLLER, **extra})
        entered_before, passed = False, False
        for _ in range(400):
            ahead = em.s - em.ego_path.project(env.agent.position)
            _, _, term, trunc, info = env.step(controller.low_level(env.agent, [LEFT, 0.5]))
            entered_before |= ahead > 0 and env.agent.lane_index[2] == start - 1
            passed |= ahead < -10
            if term or trunc:
                break
        seen[name] = (entered_before, passed, env.agent.lane_index[2])
    assert seen["without"][0] and not seen["without"][1]           # moved in ahead of it and never got past
    assert seen["with"] == (False, True, start - 1)                # kept out until past, then changed left


def test_braking_layer_ignores_the_outer_lane_of_a_curve(env):
    """Attempt-1 T1 scene 272 without traffic: a stopped car stands in the right lane of a left curve. Driving in the
    left lane towards it, the braking layer measured along the current (straight) lane found the car on that lane's
    extension and braked the ego to a stop beside it, 22 steps with an object 'in its path' on this drive. Measured
    along the route it sees nothing (LOG.md, 2026-11-03)."""
    from avtest import seeds
    from avtest.tactical import KEEP, LEFT, SUT_CONTROLLER, TacticalController
    theta = [0.2378, 0.6087, 0.7142, 0.0889, 0.1767, 0.0, 0.8178, 0.0478, 0.4937, 0.4336, 0.2474, 0.1836]
    env.load(sc.decode(np.asarray(theta), seeds.check(115238, "qual"), "T1"))
    env.reset()
    em = env.engine.managers["event_manager"]
    assert env.agent.lane_index[2] == 1 and em.path.lanes[0].index[-1] == 1       # obstacle ahead in the ego's lane
    controller = TacticalController(**SUT_CONTROLLER)
    in_path, passed = [], False
    for k in range(600):                            # at 25 km/h, which the 17 m curve allows (faster, it slides out)
        throttle = 0.5 if env.agent.speed_km_h < 25 else 0.0
        _, _, term, trunc, info = env.step(controller.low_level(env.agent, [LEFT if k < 40 else KEEP, throttle]))
        if k >= 40 and env.agent.lane_index[2] == 0:
            in_path.append(TacticalController.time_to_collision(env.agent)[1])
        if em.s - em.ego_path.project(env.agent.position) < -10:
            passed = True
            break
        if term or trunc:
            break
    assert passed and len(in_path) > 50
    assert all(gap == float("inf") for gap in in_path)


def test_steering_settles_after_a_curve_at_55_kmh(env):
    """MetaDrive's lane-keeping PID gains (from its ~30 km/h IDM traffic) oscillate at 45-60 km/h, set off by the
    heading target's jump at a curve's end (LOG.md, 2026-11-03). Lane kept at 55 km/h through two curves: the
    steering flips sign about 20 times in the 5 s after each curve; along the route and scaled above 40 km/h,
    only a few times, and the car stays closer to the lane centre."""
    from avtest.tactical import KEEP, SUT_CONTROLLER, TacticalController
    s = scene("obstacle", event_lane=0.5, event_ahead=0.9, traffic_density=0.0, lane_num=0.0,
              block_1=0.3, block_2=0.3, block_3=0.0)
    assert s.values["map"] == "CCS"
    seen = {}
    for name, extra in (("default", {}), ("fixed", {"route_steer": True, "steer_ref_kmh": 40.0})):
        env.load(s)
        env.reset()
        controller = TacticalController(**SUT_CONTROLLER, **extra)
        steer, curve, lateral = [], [], []
        for _ in range(600):
            action = controller.low_level(env.agent, [KEEP, 0.6 if env.agent.speed_km_h < 55 else 0.0])
            _, _, term, trunc, info = env.step(action)
            steer.append(float(action[0]))
            curve.append(type(env.agent.lane).__name__ == "CircularLane")
            lateral.append(abs(env.agent.lane.local_coordinates(env.agent.position)[1]))
            if term or trunc:
                break
        assert info["outcome"] == "success"
        exits = [i for i in range(1, len(curve)) if curve[i - 1] and not curve[i]]
        reversals = []
        for i in exits:
            signs = [np.sign(x) for x in steer[i:i + 50] if abs(x) > 0.03]
            reversals.append(sum(a != b for a, b in zip(signs, signs[1:])))
        seen[name] = (reversals, max(lateral))
    assert len(seen["default"][0]) == 2 and min(seen["default"][0]) >= 12
    assert max(seen["fixed"][0]) <= 6
    assert seen["fixed"][1] < seen["default"][1]


def test_traffic_gap_needed_behind_grows_with_closing_speed():
    from avtest.drivers import required_back_gap
    assert required_back_gap(0.0) == 15.0 == required_back_gap(-5.0)          # MetaDrive's distance when nobody closes in
    assert required_back_gap(14.0) == 15.0 + 3.0 * 14.0                          # ego at 60 km/h, car pulling out at 10 km/h


def test_rectangles_overlap_by_separating_axes():
    from avtest.tactical import boxes_overlap
    car = (4.5, 2.0)
    behind_in_lane, next_lane, beside_close = [[3.0, 0.0]], [[0.0, 3.5]], [[0.0, 1.8]]
    for where, expected in ((behind_in_lane, True), (next_lane, False), (beside_close, True)):
        assert boxes_overlap(np.zeros((1, 2)), 0.0, *car, np.array(where), 0.0, *car).tolist() == [expected]
    assert not boxes_overlap(np.zeros((1, 2)), 0.0, *car, np.array(next_lane), 0.0, *car, 0.0, 0.3)[0]
    crossing = np.array([[4.0, 4.0], [2.5, 2.5]])                  # a car at right angles, far and near
    assert boxes_overlap(np.zeros((2, 2)), 0.0, *car, crossing, np.pi / 2, *car).tolist() == [False, True]


def test_lane_track_follows_the_lane_and_moves_across_it():
    """yield_s: another car is projected along its lane with its speed across it, capped at one lane width, and
    straight on past the lane's end."""
    from types import SimpleNamespace
    from metadrive.component.lane.straight_lane import StraightLane
    from avtest.tactical import lane_track
    lane = StraightLane(np.array([0.0, 0.0]), np.array([50.0, 0.0]), width=3.5)
    car = SimpleNamespace(position=(10.0, 0.0), velocity=(10.0, 1.0))      # 10 m/s along, 1 m/s across
    points, headings = lane_track(car, lane, np.array([1.0, 2.0, 5.0]))
    assert np.allclose(points, [[20.0, 1.0], [30.0, 2.0], [60.0, 3.5]])   # past the end at 5 s; lateral capped
    assert np.allclose(headings, 0.0)
    points, _ = lane_track(car, lane, np.array([1.0]), velocity=(5.0, 0.0))       # a tracked velocity overrides
    assert np.allclose(points[0], [15.0, 0.0])


def test_low_speed_lane_change_pulls_out_from_behind_a_stopped_car(env):
    """lat_cap_kmh: standing 12 m behind a lead that stops at once, then commanded left: with the fixed 1 m lateral
    cap the car creeps up to the lead mid-change and the braking layer holds it there ("stuck"); with the cap
    growing at low speed it pulls out and completes the route."""
    from avtest.tactical import LEFT, GUARD_CONTROLLER, TacticalController
    outcomes = {}
    for name, extra in (("without", {}), ("with", {"lat_cap_kmh": 30.0})):
        env.load(scene("lead_brake", event_ahead=0.0, event_trigger_s=0.0, traffic_density=0.0, lane_num=0.0,
                       event_rel_speed_mps=0.0))
        env.reset()
        controller = TacticalController(**{**GUARD_CONTROLLER, **extra})
        for k in range(600):
            command = [0.0, -1.0] if k < 40 else [LEFT, 0.5]            # wait until the lead has stopped
            _, _, term, trunc, info = env.step(controller.low_level(env.agent, command))
            if term or trunc:
                break
        outcomes[name] = info["outcome"] if term or trunc else "still driving"
    assert outcomes == {"without": "stuck", "with": "success"}


def test_predictive_braking_sees_a_cut_in_before_the_braking_layer(env):
    """yield_s: a car cutting in from the next lane is braked for while it is still moving across, before the
    braking layer, which waits until it is in the ego's path; the cut-in's sideways motion is estimated from its
    positions, since the scripted event vehicle reports a velocity along its path only."""
    from avtest.tactical import KEEP, GUARD_CONTROLLER, TacticalController
    env.load(scene("cut_in", event_lane=0.4, event_ahead=0.2, event_trigger_s=0.2, traffic_density=0.0,
                   event_intensity=1.0, event_rel_speed_mps=0.33, lane_num=0.0))
    env.reset()
    controller = TacticalController(**{**GUARD_CONTROLLER, "yield_s": 3.0})
    first_yield = first_aeb = None
    for k in range(300):
        _, _, term, trunc, info = env.step(controller.low_level(env.agent, [KEEP, 0.7]))
        if first_yield is None and controller.last_yield < 3.0:
            first_yield = k
        if first_aeb is None and (controller.last_ttc < 2.5 or controller.last_gap < 1.5):
            first_aeb = k
        if term or trunc:
            break
    assert first_yield is not None and (first_aeb is None or first_yield < first_aeb)
    assert not (term or trunc) or info["outcome"] != "collision"
