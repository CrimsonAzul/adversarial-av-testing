"""Play one scenario with a chosen driver and save a GIF + keyframe strip (debugging and thesis figures).

    .venv\\Scripts\\python.exe -m scripts.inspect_scenario --event cut_in --driver idm --seed 120005
    .venv\\Scripts\\python.exe -m scripts.inspect_scenario --theta "0.1,0.4,..." --seed 120005 --driver sut --model models\\sut.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario as sc
from avtest.records import ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tier", default="T2")
    p.add_argument("--event", choices=("lead_brake", "cut_in", "obstacle"), default="cut_in")
    p.add_argument("--theta", help="comma-separated θ (overrides --event)")
    p.add_argument("--seed", type=int, default=120_005)
    p.add_argument("--driver", default="idm", choices=("sut", "idm", "idm_cautious", "expert", "idm60", "idm_cautious60"))
    p.add_argument("--model", type=Path, default=ROOT / "models/sut.zip")
    p.add_argument("--out", type=Path, default=ROOT / "runs/inspect")
    args = p.parse_args()
    from avtest import drivers
    from avtest.recording import record_episode
    from avtest.sim import AVTestEnv, sensor_config

    theta = (np.asarray([float(x) for x in args.theta.split(",")]) if args.theta
             else sc.with_event(np.random.default_rng(args.seed).random(sc.DIM), args.event, args.tier))
    scene = sc.decode(theta, args.seed, args.tier)
    driver = {"sut": lambda: drivers.SUTDriver(args.model), "idm": drivers.IDMDriver,
              "idm_cautious": drivers.CautiousIDMDriver, "expert": drivers.ExpertDriver,
              "idm60": drivers.RoadIDMDriver, "idm_cautious60": drivers.CautiousRoadIDMDriver}[args.driver]()
    env = AVTestEnv(sensor_config(driver.obs_size) if args.driver == "sut" else None)
    try:
        result = record_episode(env, scene, driver, args.out / f"{args.driver}_{scene.values['event']}_{args.seed}",
                                f"inspect seed {args.seed}")
    finally:
        env.close()
    print({k: result[k] for k in ("outcome", "steps", "min_ttc_s", "min_edge_margin_m", "event_triggered_step", "gif")})


if __name__ == "__main__":
    main()
