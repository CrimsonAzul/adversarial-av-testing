"""Watch a driver in MetaDrive's 3D view: random scenes of a tier, one after another, with a running tally.

    .venv\\Scripts\\python.exe -m scripts.watch_driver --model runs\\driver\\d3\\model_best_bc.zip --tier T1
    .venv\\Scripts\\python.exe -m scripts.watch_driver --driver idm --tier T3        # a reference driver, to compare

Opens a window with a chase camera behind the ego car. Scenes are drawn the way random testing draws
them (uniform θ for the tier, random map seeds), so this is the distribution qualification tests, not
hand-picked situations. The overlay shows the scene (map blocks, lanes, traffic, event), the AI's
current decision (lane command and throttle/brake) and the tally so far. --speed 2 plays twice as fast,
0 as fast as the machine can. Close the window or press Ctrl-C to stop. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario as sc, seeds
from avtest.records import ROOT


def decision_text(driver) -> str:
    decision = getattr(driver, "last_decision", None)
    if decision is None:
        return "rule-based reference driver"
    lane, throttle = float(decision[0]), float(decision[1])
    lane_text = "change left" if lane > 0.5 else "change right" if lane < -0.5 else "keep lane"
    pedal = f"throttle {throttle:.2f}" if throttle >= 0 else f"brake {-throttle:.2f}"
    return f"{lane_text} ({lane:+.2f}), {pedal}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=ROOT / "models/sut.zip", help="the AI driver (PPO checkpoint)")
    p.add_argument("--driver", default="sut", choices=("sut", "idm", "idm_cautious", "expert"))
    p.add_argument("--tier", default="T1", choices=sorted(sc.TIERS))
    p.add_argument("--scenes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="which random scenes; the same seed shows the same scenes")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()
    if args.driver == "sut" and not args.model.exists():
        raise SystemExit(f"{args.model} not found; pass --model, e.g. runs\\driver\\d3\\model_best_bc.zip")
    from avtest import drivers
    from avtest.sim import DT, AVTestEnv
    driver = {"sut": lambda: drivers.SUTDriver(args.model), "idm": drivers.IDMDriver,
              "idm_cautious": drivers.CautiousIDMDriver, "expert": drivers.ExpertDriver}[args.driver]()
    env = AVTestEnv({"use_render": True, "window_size": (1280, 800)})
    rng = np.random.default_rng([args.seed, 7])
    tally = Counter()
    try:
        for k in range(1, args.scenes + 1):
            scene = sc.decode(sc.sample(rng, 1)[0], int(seeds.draw(rng, "val")), args.tier)
            v = scene.values
            env.load(scene)
            obs, _ = env.reset()
            driver.start(env)
            about = (f"{v['map']}, {v['lane_num']} lanes, traffic {v['traffic_density']:.2f} | {v['event']} "
                     f"({v['event_lane']} lane, {v['event_ahead_m']:.0f} m ahead)")
            record = ", ".join(f"{n} {o}" for o, n in tally.most_common()) or "-"
            while True:
                started = time.time()
                obs, _, terminated, truncated, info = env.step(driver.act(env, obs))
                env.render(text={"Driver": driver.name if args.driver != "sut" else f"AI ({args.model.name})",
                                 "Scene": f"{k}/{args.scenes}  {args.tier}: {about}",
                                 "AI decision": decision_text(driver),
                                 "Speed": f"{env.agent.speed_km_h:.0f} km/h",
                                 "So far": record})
                if args.speed > 0:
                    time.sleep(max(0.0, DT / args.speed - (time.time() - started)))
                if terminated or truncated:
                    break
            tally[info["outcome"]] += 1
            print(f"scene {k}: {info['outcome']:9s} {about}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    total = sum(tally.values())
    if total:
        print(f"{tally['success']}/{total} completed; {dict(tally)}")


if __name__ == "__main__":
    main()
