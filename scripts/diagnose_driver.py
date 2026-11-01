"""Why does a driver fail? Per-episode failure causes on validation scenes, plus an empty-road lane-keeping test.

    .venv\\Scripts\\python.exe -m scripts.diagnose_driver --model runs\\driver\\d2_pilot\\model_last.zip --scenes 40

For each failure it reports the exact MetaDrive flag (off every lane, sidewalk, yellow line,
vehicle or object collision), the step, speed, route completion, the lane index history
(was a lane change under way?) and the lateral offset from the lane centre in the last second.
The empty-road test runs the same scenes with traffic removed and the event pushed out of reach,
which separates lane keeping from interaction skills.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import progress, scenario as sc
from avtest.records import ROOT, write_json


def run(env, scene, driver):
    env.load(scene)
    obs, _ = env.reset()
    driver.start(env)
    lanes, lateral, steer = [], [], []
    while True:
        action = driver.act(env, obs)
        obs, _, term, trunc, info = env.step(action)
        ego = env.agent
        lanes.append(ego.lane_index[2] if ego.lane_index else None)
        try:
            lateral.append(float(ego.lane.local_coordinates(ego.position)[1]))
        except Exception:
            lateral.append(float("nan"))
        steer.append(float(np.asarray(action)[0]))
        if term or trunc:
            break
    flags = {k: bool(getattr(ego, k, False)) for k in ("on_yellow_continuous_line", "on_white_continuous_line",
                                                       "crash_sidewalk", "crash_vehicle", "crash_object")}
    flags["off_every_lane"] = not ego.on_lane
    return {"outcome": info["outcome"], "step": env.engine.episode_step, "speed_kmh": float(ego.speed_km_h),
            "route_completion": float(info.get("route_completion", 0.0)), "flags": flags,
            "lane_changes": int(sum(a != b for a, b in zip(lanes, lanes[1:]) if a is not None and b is not None)),
            "lateral_last_1s": [round(x, 2) for x in lateral[-10:]], "steer_abs_mean": float(np.mean(np.abs(steer))),
            "steer_flips_per_s": float(np.sum(np.diff(np.sign(steer)) != 0) / (len(steer) * 0.1)),
            "event": scene.values["event"], "map": scene.values["map"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=40)
    p.add_argument("--driver", default="sut", choices=("sut", "idm_cautious"))
    p.add_argument("--legacy-obs", action="store_true",
                   help="observation of runs before 2026-11-01 evening (no lane-line/road-edge detectors), for old models")
    args = p.parse_args()
    from avtest.drivers import CautiousIDMDriver, SUTDriver
    from avtest.sim import AVTestEnv
    driver = SUTDriver(args.model) if args.driver == "sut" else CautiousIDMDriver()
    thetas, map_seeds = progress.validation_panel(args.scenes)
    names = [q.name for q in sc.PARAMS]
    from avtest import sim
    vehicle = {"lidar": dict(sim.BASE_CONFIG["vehicle_config"]["lidar"])}
    env = AVTestEnv({"vehicle_config": vehicle}) if args.legacy_obs else AVTestEnv()
    report = {"normal": [], "empty_road": []}
    try:
        for theta, seed in zip(thetas, map_seeds):
            report["normal"].append(run(env, sc.decode(theta, seed, progress.TIER), driver))
            easy = np.array(theta)
            easy[names.index("traffic_density")] = 0.0
            easy[names.index("event_trigger_s")] = 1.0             # event as late as possible ...
            easy[names.index("event_ahead")] = 1.0                 # ... and as far ahead as possible
            easy = sc.with_event(easy, "obstacle", progress.TIER)
            easy[names.index("event_lane")] = 0.99                  # obstacle in a side lane
            report["empty_road"].append(run(env, sc.decode(easy, seed, progress.TIER), driver))
    finally:
        env.close()
    for name, rows in report.items():
        outcomes = Counter(r["outcome"] for r in rows)
        causes = Counter(k for r in rows if r["outcome"] in ("offroad", "collision")
                         for k, v in r["flags"].items() if v)
        print(f"\n{name}: {dict(outcomes)} | failure flags: {dict(causes)}")
        print(f"  mean |steer| {np.mean([r['steer_abs_mean'] for r in rows]):.3f}, steering sign flips/s "
              f"{np.mean([r['steer_flips_per_s'] for r in rows]):.2f}, lane changes/episode "
              f"{np.mean([r['lane_changes'] for r in rows]):.2f}")
        for r in [r for r in rows if r["outcome"] == "offroad"][:4]:
            print(f"  offroad at step {r['step']} ({r['speed_kmh']:.0f} km/h, route {r['route_completion']:.0%}, "
                  f"{r['map']}, lane changes {r['lane_changes']}): lateral offset last 1 s {r['lateral_last_1s']}")
    out = ROOT / "runs/diagnose" / f"{args.model.parent.parent.name}_{args.model.stem}_{args.driver}.json"
    write_json(out, report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
