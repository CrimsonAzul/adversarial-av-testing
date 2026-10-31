"""Play one scenario with a chosen driver and save a top-down GIF/PNG strip (debugging and thesis figures).

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


def theta_for(event: str, tier: str, rng: np.random.Generator) -> np.ndarray:
    names = [p.name for p in sc.PARAMS]
    events = sc.TIERS[tier]["events"]
    while True:
        theta = rng.random(sc.DIM)
        theta[names.index("event")] = (events.index(event) + 0.5) / len(events)
        if sc.decode(theta, 0, tier).values["event"] == event:
            return theta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tier", default="T2")
    p.add_argument("--event", choices=("lead_brake", "cut_in", "obstacle"))
    p.add_argument("--theta", help="comma-separated θ (overrides --event)")
    p.add_argument("--seed", type=int, default=120_005)
    p.add_argument("--driver", default="idm", choices=("sut", "idm", "idm_cautious", "expert"))
    p.add_argument("--model", type=Path, default=ROOT / "models/sut.zip")
    p.add_argument("--out", type=Path, default=ROOT / "runs/inspect")
    p.add_argument("--every", type=int, default=5, help="keep every n-th frame")
    args = p.parse_args()
    from PIL import Image
    from avtest import drivers
    from avtest.sim import AVTestEnv

    theta = (np.asarray([float(x) for x in args.theta.split(",")]) if args.theta
             else theta_for(args.event, args.tier, np.random.default_rng(args.seed)))
    scene = sc.decode(theta, args.seed, args.tier)
    driver = {"sut": lambda: drivers.SUTDriver(args.model), "idm": drivers.IDMDriver,
              "idm_cautious": drivers.CautiousIDMDriver, "expert": drivers.ExpertDriver}[args.driver]()
    env = AVTestEnv()
    env.load(scene)
    obs, _ = env.reset()
    driver.start(env)
    frames, step = [], 0
    while True:
        obs, _, term, trunc, info = env.step(driver.act(env, obs))
        if step % args.every == 0 or term or trunc:
            frame = env.render(mode="topdown", window=False, screen_size=(600, 600), camera_position=None,
                               target_agent_heading_up=True, film_size=(2000, 2000))
            frames.append(Image.fromarray(np.asarray(frame).swapaxes(0, 1)[:, :, :3].astype(np.uint8)))
        step += 1
        if term or trunc:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.driver}_{scene.values['event']}_{args.seed}"
    frames[0].save(args.out / f"{stem}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
    print({"outcome": info["outcome"], "steps": step, **{k: env.telemetry[k] for k in env.telemetry},
           "event_triggered_step": env.engine.managers["event_manager"].triggered_at,
           "values": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in scene.values.items()},
           "gif": str(args.out / f"{stem}.gif")})
    env.close()


if __name__ == "__main__":
    main()
