"""Train the SUT driver with PPO on randomized MetaDrive worlds, in resumable overnight sessions.

    .venv\\Scripts\\python.exe -m scripts.train_driver --name d1 --hours 8
    .venv\\Scripts\\python.exe -m scripts.train_driver --name d1 --hours 8          # next night: continues d1
    .venv\\Scripts\\python.exe -m scripts.train_driver --name d1 --check            # 2-minute pipeline check

Training samples θ uniformly over the widest tier (T3) on train-split map seeds,
with a curriculum on event intensity and traffic over the first 30% of the step
target. Every chunk (default 500k steps) the policy is saved and scored on a fixed
validation panel (val split, never the qualification split); the best-scoring
checkpoint is kept as model_best.zip. A session ends after --hours at a chunk
boundary, or when runs/driver/<name>/PAUSE exists; STOP aborts at once.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch  # noqa: F401  (before MetaDrive: avoids the Windows c10.dll load-order failure)

from avtest import scenario
from avtest.records import JsonlLog, ROOT, control_state, read_json, write_json

DEFAULTS = {"tier": "T3", "envs": 6, "total_steps": 40_000_000, "chunk_steps": 500_000, "curriculum_until": 0.3,
            "val_panel": 200, "val_panel_seed": 7, "n_steps": 1024, "batch_size": 512, "n_epochs": 10,
            "learning_rate": 1e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
            "net": [256, 256], "seed": 0}


def difficulty(done_steps: int, cfg: dict) -> float:
    """0.3 → 1.0 linearly over the curriculum fraction, then 1.0."""
    ramp = cfg["curriculum_until"] * cfg["total_steps"]
    return 1.0 if done_steps >= ramp else 0.3 + 0.7 * done_steps / ramp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--envs", type=int, default=DEFAULTS["envs"])
    p.add_argument("--check", action="store_true", help="tiny run that exercises every step, then exits")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from avtest import sim, seeds
    from avtest.evaluate import Evaluator

    out = ROOT / "runs/driver" / args.name
    cfg = dict(DEFAULTS, envs=args.envs)
    if args.check:
        cfg.update(total_steps=4096, chunk_steps=2048, n_steps=256, batch_size=256, val_panel=4, envs=2)
        out = out.with_name(args.name + "_check")
    if (out / "config.json").exists():
        saved = read_json(out / "config.json")
        if saved != cfg:
            raise SystemExit(f"{out} has a different config; use a new --name")
    else:
        write_json(out / "config.json", cfg)
    log = JsonlLog(out / "train_log.jsonl")
    history = log.read()
    done_steps = history[-1]["steps"] if history else 0
    best = max((h["val_success"] for h in history), default=-1.0)
    val_seeds = seeds.panel_seeds("val", cfg["val_panel"], cfg["val_panel_seed"])
    import numpy as np
    val_thetas = scenario.sample(np.random.default_rng(cfg["val_panel_seed"]), cfg["val_panel"])

    env = VecMonitor(SubprocVecEnv([sim.training_env_fn(cfg["tier"], rank=i, seed=cfg["seed"] + 1000 * i)
                                    for i in range(cfg["envs"])], start_method="spawn"))
    env.env_method("set_difficulty", difficulty(done_steps, cfg))
    if (out / "model_last.zip").exists():
        model = PPO.load(out / "model_last.zip", env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
                    learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
                    clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"], seed=cfg["seed"], device="cpu",
                    policy_kwargs={"net_arch": {"pi": cfg["net"], "vf": cfg["net"]}}, verbose=0)
    started = time.time()
    print(f"driver {args.name}: {done_steps:,}/{cfg['total_steps']:,} steps done; session budget {args.hours} h", flush=True)
    try:
        while done_steps < cfg["total_steps"]:
            state = control_state(out)
            if state or time.time() - started > args.hours * 3600:
                print(f"session ends ({state or 'hours used'}); continue with the same command", flush=True)
                break
            d = difficulty(done_steps, cfg)
            env.env_method("set_difficulty", d)
            t0 = time.time()
            model.learn(total_timesteps=cfg["chunk_steps"], reset_num_timesteps=False, progress_bar=False)
            done_steps += cfg["chunk_steps"]
            model.save(out / "model_last.zip")
            train_s = time.time() - t0
            with Evaluator(out / "model_last.zip", cfg["tier"], workers=2, references=False) as evaluator:
                rows = evaluator.evaluate(val_thetas, val_seeds, split="val")
            success = float(np.mean([r["sut"]["outcome"] == "success" for r in rows]))
            if success > best:
                best = success
                model.save(out / "model_best.zip")
            log.append({"steps": done_steps, "difficulty": d, "val_success": success, "best_val_success": best,
                        "val_outcomes": {k: sum(r["sut"]["outcome"] == k for r in rows)
                                         for k in ("success", "collision", "offroad", "stuck", "timeout")},
                        "train_s": train_s, "steps_per_s": cfg["chunk_steps"] / train_s, "utc": time.time()})
            print(f"{done_steps:>11,} steps  difficulty {d:.2f}  val success {success:.1%}  best {best:.1%}  "
                  f"({cfg['chunk_steps'] / train_s:.0f} steps/s)", flush=True)
    finally:
        env.close()
    if done_steps >= cfg["total_steps"]:
        write_json(out / "done.json", {"steps": done_steps, "best_val_success": best})


if __name__ == "__main__":
    main()
