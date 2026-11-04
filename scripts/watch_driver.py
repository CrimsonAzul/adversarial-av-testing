"""Watch a driver in MetaDrive's 3D view: random scenes of a tier, or chosen development scenes, with a running tally.

    .venv\\Scripts\\python.exe -m scripts.watch_driver --model runs\\driver\\d11\\model_best_bc.zip --tier T1
    .venv\\Scripts\\python.exe -m scripts.watch_driver --model runs\\driver\\d3\\checkpoints\\bc_round_02.zip --tier T1 --traffic30
    .venv\\Scripts\\python.exe -m scripts.watch_driver --driver idm60 --tier T1      # a reference driver, to compare
    .venv\\Scripts\\python.exe -m scripts.watch_driver --model runs\\driver\\d9\\model_best_bc.zip --row 28 --set threshold=0.3
    .venv\\Scripts\\python.exe -m scripts.watch_driver --model runs\\driver\\d9\\model_best_bc.zip --row 28 --set threshold=0.3,aeb_ttc=2.5

Opens a window with a chase camera behind the ego car. By default scenes are drawn the way random testing
draws them (uniform θ for the tier, random map seeds), so this is the distribution qualification tests, not
hand-picked situations. --row replays scenes of the development panel instead (the 500 T1 scenes of the failed
qualification attempt 1; rows as in docs/LOG.md and runs/diagnose), so a driver can be watched in the scene where
it failed. --set changes the driver's controller settings (avtest.tactical.TacticalController), e.g. to watch
a driver with or without the emergency-braking layer. --driver teacher60 is the 60 km/h imitation teacher
driving through the same controller. The overlay shows the scene (map blocks, lanes, traffic, event), the AI's
current decision (lane command and throttle/brake) and the tally so far. --speed 2 plays twice as fast, 0 as fast
as the machine can. --headless opens no window and only prints each scene's outcome. Close the window or press
Ctrl-C to stop. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario as sc, seeds
from avtest.records import ROOT

DEV_PANEL = ROOT / "runs/qualification/attempt1_T1"      # failed attempt 1: development data (docs/DESIGN.md)


def decision_text(driver) -> str:
    decision = getattr(driver, "last_decision", None)
    if decision is None:
        return "rule-based driver"
    lane, throttle = float(decision[0]), float(decision[1])
    lane_text = "change left" if lane > 0.5 else "change right" if lane < -0.5 else "keep lane"
    pedal = f"throttle {throttle:.2f}" if throttle >= 0 else f"brake {-throttle:.2f}"
    return f"{lane_text} ({lane:+.2f}), {pedal}"


def parse_settings(text: str) -> dict:
    """'threshold=0.3,aeb_ttc=none,route_steer=1' -> {'threshold': 0.3, 'aeb_ttc': None, 'route_steer': 1}."""
    settings = {}
    for item in filter(None, text.split(",")):
        key, value = item.split("=")
        low = value.strip().lower()
        settings[key.strip()] = (None if low in ("none", "off") else True if low == "true" else False if low == "false"
                                 else float(low) if "." in low else int(low))
    return settings


def dev_scenes(rows: list[int]):
    if json.loads((DEV_PANEL / "result.json").read_text())["passed"]:
        raise SystemExit(f"{DEV_PANEL} passed: its panel is not development data")
    panel = [json.loads(line) for line in (DEV_PANEL / "episodes.jsonl").read_text().splitlines()]
    for i in rows:
        yield f"dev row {i}", sc.decode(np.asarray(panel[i]["theta"]), seeds.check(panel[i]["map_seed"], "qual"), "T1")


def random_scenes(seed: int, count: int, tier: str):
    rng = np.random.default_rng([seed, 7])
    for k in range(1, count + 1):
        yield f"{k}/{count} {tier}", sc.decode(sc.sample(rng, 1)[0], int(seeds.draw(rng, "val")), tier)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=ROOT / "models/sut.zip", help="the AI driver (model file)")
    p.add_argument("--driver", default="sut",
                   choices=("sut", "teacher60", "idm", "idm_cautious", "expert", "idm60", "idm_cautious60"))
    p.add_argument("--tier", default="T1", choices=sorted(sc.TIERS))
    p.add_argument("--scenes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="which random scenes; the same seed shows the same scenes")
    p.add_argument("--row", help="development-panel rows to replay instead of random scenes, e.g. 102 or 28,134")
    p.add_argument("--set", dest="settings", default="",
                   help="controller settings to change, e.g. threshold=0.3,aeb_ttc=2.5 or aeb_ttc=none")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--headless", action="store_true", help="no window: print each scene's outcome only")
    p.add_argument("--traffic30", action="store_true",
                   help="MetaDrive's own 30 km/h background traffic, the environment the 20 km/h drivers (d2, d3) "
                        "were measured in, instead of today's 50-60 km/h traffic (avtest.sim.ROAD_TRAFFIC)")
    args = p.parse_args()
    if args.driver == "sut" and not args.model.exists():
        raise SystemExit(f"{args.model} not found; pass --model, e.g. runs\\driver\\d11\\model_best_bc.zip")
    from avtest import drivers, sim
    from avtest.sim import DT, AVTestEnv, sensor_config
    from avtest.tactical import TacticalController, TeacherDriver
    if args.traffic30:
        sim.ROAD_TRAFFIC = False                       # read when the env sets up its engine
    driver = {"sut": lambda: drivers.SUTDriver(args.model), "teacher60": lambda: TeacherDriver("scene_passing_60"),
              "idm": drivers.IDMDriver, "idm_cautious": drivers.CautiousIDMDriver, "expert": drivers.ExpertDriver,
              "idm60": drivers.RoadIDMDriver, "idm_cautious60": drivers.CautiousRoadIDMDriver}[args.driver]()
    changes = parse_settings(args.settings)
    if changes and not hasattr(driver, "controller_settings"):
        raise SystemExit("--set changes the controller of the AI driver or teacher60; reference drivers have none")
    if hasattr(driver, "controller_settings"):
        driver.controller_settings = {**driver.controller_settings, **changes}
        if args.driver == "sut":
            driver.controller = TacticalController(**driver.controller_settings)
    controller_text = (", ".join(f"{k}={v}" for k, v in sorted(driver.controller_settings.items()))
                       if hasattr(driver, "controller_settings") else "its own")
    sensors = sensor_config(driver.obs_size) if args.driver == "sut" else {}       # older drivers: their lidar
    env = AVTestEnv({**sensors, "use_render": not args.headless, "window_size": (1280, 800)})
    scenes = (dev_scenes([int(r) for r in args.row.split(",")]) if args.row
              else random_scenes(args.seed, args.scenes, args.tier))
    name = driver.name if args.driver != "sut" else f"AI ({args.model.parent.name}/{args.model.name})"
    print(f"{name}; controller: {controller_text}", flush=True)
    tally = Counter()
    try:
        for label, scene in scenes:
            v = scene.values
            env.load(scene)
            obs, _ = env.reset()
            driver.start(env)
            about = (f"{v['map']}, {v['lane_num']} lanes, traffic {v['traffic_density']:.2f} | {v['event']} "
                     f"({v['event_lane']} lane, {v['event_ahead_m']:.0f} m ahead)")
            record = ", ".join(f"{n} {o}" for o, n in tally.most_common()) or "-"
            lane, lane_changes, speeds = env.agent.lane_index, 0, []
            while True:
                started = time.time()
                obs, _, terminated, truncated, info = env.step(driver.act(env, obs))
                speeds.append(env.agent.speed_km_h)
                now = env.agent.lane_index
                lane_changes += now[:2] == lane[:2] and now[2] != lane[2]          # same road, different lane
                lane = now
                if not args.headless:
                    env.render(text={"Driver": name, "Scene": f"{label}: {about}",
                                     "AI decision": decision_text(driver),
                                     "Speed": f"{env.agent.speed_km_h:.0f} km/h",
                                     "Lane changes": str(lane_changes), "So far": record})
                    if args.speed > 0:
                        time.sleep(max(0.0, DT / args.speed - (time.time() - started)))
                if terminated or truncated:
                    break
            tally[info["outcome"]] += 1
            print(f"{label}: {info['outcome']:9s} {lane_changes} lane changes, mean {np.mean(speeds):.0f} km/h, "
                  f"{len(speeds) * DT:.0f} s | {about}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    total = sum(tally.values())
    if total:
        print(f"{tally['success']}/{total} completed; {dict(tally)}")


if __name__ == "__main__":
    main()
