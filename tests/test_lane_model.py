"""The lane model of end-to-end drivers (own module: MetaDrive allows one engine per process)."""
import numpy as np

import torch  # noqa: F401  (before MetaDrive)
from avtest import scenario as sc, seeds, tactical
from avtest.sim import (E2E_OBS_DIM, LANE_MODEL_AHEAD_M, LANE_MODEL_LANES, LANE_MODEL_WIDTH_M, LANE_MODEL_X_M,
                        LANE_MODEL_Y_M, OBS_DIM, AVTestEnv, RoutePath, sensor_config)


def straight_scene(tier: str = "T1"):
    rng = np.random.default_rng(11)
    while True:
        scene = sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), tier)
        if scene.values["map"] == "SSS":
            return scene


def test_lane_model_reports_the_lane_steadily():
    """On a straight road the lane model's centre-line points lie ahead at the look-ahead distances, the first one
    beside the ego at its true offset from the lane centre, with the lane's width and the lanes beside it. While the
    teacher keeps its lane the offset changes by centimetres per step, where the lane-line beams jump by metres."""
    assert sensor_config(E2E_OBS_DIM) == {"lane_model": True} and sensor_config(OBS_DIM) == {}
    scene = straight_scene()
    env = AVTestEnv(sensor_config(E2E_OBS_DIM))
    try:
        env.load(scene)
        obs, _ = env.reset()
        assert obs.shape == (E2E_OBS_DIM,) and obs.dtype == np.float32
        teacher = tactical.TEACHERS["scene_passing_60"](env.agent, env.scenario.map_seed)
        ctl = tactical.TacticalController(**tactical.YIELD_CONTROLLER)
        offsets, true, steps = [], [], 0
        while steps < 200:
            model = obs[OBS_DIM:]
            x = (model[0:10:2] * 2 - 1) * LANE_MODEL_X_M
            y = (model[1:10:2] * 2 - 1) * LANE_MODEL_Y_M
            lane = env.agent.navigation.current_lane
            path = RoutePath(env.engine.current_map.road_network, env.agent.navigation.checkpoints, lane.index[-1])
            ahead = np.array(LANE_MODEL_AHEAD_M) < path.length - path.local(env.agent.position, clamp=False)[0] - 1
            if steps >= 20:                          # at the start the ego stands up to 2 m before the route's start
                assert np.allclose(x[ahead], np.array(LANE_MODEL_AHEAD_M)[ahead], atol=1.0)   # the points lie ahead,
                assert np.all(x[~ahead] <= np.array(LANE_MODEL_AHEAD_M)[~ahead] + 1)   # or at the route's end
                fit = np.polyval(np.polyfit(x, y, 1), x)                         # on a line, angled by the
                assert np.abs(y - fit).max() < 0.1                               # ego's heading error
            assert abs(model[10] * LANE_MODEL_WIDTH_M - lane.width) < 1e-3
            refs = env.agent.navigation.current_ref_lanes
            assert round(model[11] * LANE_MODEL_LANES) + round(model[12] * LANE_MODEL_LANES) == len(refs) - 1
            if steps >= 20:
                offsets.append(y[0])
                true.append(lane.local_coordinates(env.agent.position)[1])
            _, acc = teacher.act(env.agent.id)
            action = ctl.low_level(env.agent, [tactical.teacher_command(teacher, env.agent), float(np.clip(acc, -1, 1))])
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break
        offsets, true = np.array(offsets), np.array(true)
        # the centre's side offset in the ego's frame equals the ego's lateral coordinate in MetaDrive's lane frame
        # (MetaDrive's y axis is mirrored against the heading's normal used here)
        assert np.allclose(offsets, true, atol=0.02) and np.abs(true).max() > 0.02
        assert np.percentile(np.abs(np.diff(offsets)), 99) < 0.2                   # steady: no beam-like jumps
    finally:
        env.close()
