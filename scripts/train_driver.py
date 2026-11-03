"""Train the SUT driver with PPO on randomized MetaDrive worlds, in resumable overnight sessions.

    .venv\\Scripts\\python.exe -m scripts.train_driver --name d2 --hours 8
    .venv\\Scripts\\python.exe -m scripts.train_driver --name d2 --hours 8          # next night: continues d2
    .venv\\Scripts\\python.exe -m scripts.train_driver --name d2 --check            # few-minute pipeline check
    .venv\\Scripts\\python.exe -m scripts.train_driver --name d2 --pilot            # ~40 min learning pilot after pretrain_bc --pilot

Starts from the imitation warm start in runs/driver/<name>/ if scripts.pretrain_bc made one,
otherwise from scratch. Training samples θ uniformly over T2 (docs/LOG.md, 2026-11-01: the
reference drivers themselves fail on 22.5% of T3 scenes) on train-split map seeds, with a
curriculum on event difficulty over the first 30% of the step target when training from scratch;
after an imitation warm start it trains on the full distribution from the first step.

After a warm start, the first 300k steps train only the value network (the policy is frozen),
because PPO updates driven by an untrained critic can undo the imitated behaviour; afterwards
small steps (learning rate 3e-5, clip 0.1, ``target_kl`` 0.01) keep each update close to the policy.

Every chunk (1M steps): the policy is saved (model_last.zip and checkpoints/), validated on the
fixed 200-scene panel, and recorded on the three showcase scenarios (avtest.progress); the best
validation checkpoint is kept as model_best.zip. train_log.jsonl gets one row per chunk with
training-episode outcomes, PPO diagnostics and validation results; progress.png is redrawn.

A session ends after --hours at a chunk boundary, or at the next boundary when
runs/driver/<name>/PAUSE exists. STOP saves model_last.zip within ~1,000 steps and exits
without validating; the next session continues from it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import time

import torch  # noqa: F401  (before MetaDrive: avoids the Windows c10.dll load-order failure)
import numpy as np

from avtest.records import JsonlLog, ROOT, control_state, read_json, write_json

DEFAULTS = {"tier": "T2", "envs": 6, "total_steps": 40_000_000, "chunk_steps": 1_000_000, "curriculum_until": 0.3,
            "critic_warmup_steps": 300_000, "val_workers": 4, "n_steps": 1024, "batch_size": 512, "n_epochs": 5,
            "learning_rate": 3e-5, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.0,
            "target_kl": 0.01, "net": [256, 256], "seed": 0}
# Run d1 (lr 1e-4, 10 epochs, clip 0.2, target_kl 0.02) moved the imitated policy too far per update; these
# settings make PPO refine the warm start (docs/LOG.md, 2026-11-01).
PPO_SETTINGS = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range",
                "ent_coef", "target_kl")


def difficulty(done_steps: int, cfg: dict) -> float:
    """0.3 → 1.0 linearly over the curriculum fraction of the step target, then 1.0."""
    ramp = cfg["curriculum_until"] * cfg["total_steps"]
    return 1.0 if done_steps >= ramp else 0.3 + 0.7 * done_steps / ramp


def actor_parameters(model) -> list:
    policy = model.policy
    return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(), policy.log_std]


def make_monitor(out):
    from stable_baselines3.common.callbacks import BaseCallback

    class Monitor(BaseCallback):
        """Counts training-episode outcomes and honours STOP within ~1,000 steps."""

        def __init__(self):
            super().__init__()
            self.outcomes, self.returns, self.lengths, self.stopped = Counter(), [], [], False

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done:
                    self.outcomes[info.get("outcome") or "unknown"] += 1
                    if "episode" in info:
                        self.returns.append(info["episode"]["r"])
                        self.lengths.append(info["episode"]["l"])
            if self.n_calls % 200 == 0 and control_state(out) == "stop":
                self.stopped = True
                return False
            return True

        def summary(self) -> dict:
            n = sum(self.outcomes.values())
            return {"train_episodes": n, "train_outcomes": dict(self.outcomes),
                    "train_success": self.outcomes["success"] / n if n else None,
                    "train_mean_return": float(np.mean(self.returns)) if self.returns else None,
                    "train_mean_length": float(np.mean(self.lengths)) if self.lengths else None}

    return Monitor()


def ppo_diagnostics(model) -> dict:
    keep = ("approx_kl", "clip_fraction", "entropy_loss", "explained_variance", "value_loss", "policy_gradient_loss", "std")
    values = model.logger.name_to_value
    return {f"ppo_{k}": float(values[f"train/{k}"]) for k in keep if f"train/{k}" in values}


def run_settings(args, root=ROOT) -> tuple:
    """(output folder, config, validation panel size, step count to stop at) for the chosen mode."""
    from avtest import progress

    if args.check and args.pilot:
        raise SystemExit("--check and --pilot are separate modes; pass one")
    out = root / "runs/driver" / args.name
    cfg = dict(DEFAULTS, envs=args.envs)
    panel = progress.VAL_PANEL
    stop_at = None
    if args.pilot:
        cfg.update(critic_warmup_steps=100_000, chunk_steps=300_000)
        panel, stop_at = 100, 700_000
        out = out.with_name(args.name + "_pilot")
    if args.check:
        cfg.update(total_steps=6144, chunk_steps=2048, critic_warmup_steps=1024, n_steps=256, batch_size=256, envs=2,
                   val_workers=2)
        panel = 4
        if not (root / "runs/driver" / (args.name + "_bccheck") / "model_last.zip").exists():
            out = out.with_name(args.name + "_check")
        else:
            out = out.with_name(args.name + "_bccheck")      # exercise the warm-start path too
    # after every override: the check once read total_steps before shrinking it and ran on to the session limit
    return out, cfg, panel, stop_at or cfg["total_steps"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--envs", type=int, default=DEFAULTS["envs"])
    p.add_argument("--check", action="store_true", help="tiny run that exercises every step, then exits")
    p.add_argument("--pilot", action="store_true",
                   help="production settings on <name>_pilot; 100k critic warm-up, then 2 x 300k chunks, 100-scene validation")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from avtest import progress, sim

    out, cfg, panel, stop_at = run_settings(args)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        if read_json(out / "config.json") != cfg:
            raise SystemExit(f"{out} has a different config; use a new --name")
    else:
        write_json(out / "config.json", cfg)
    log = JsonlLog(out / "train_log.jsonl")
    history = log.read()
    best = max((h["val_success"] for h in history if h.get("val_success") is not None), default=-1.0)
    warm_start = (out / "bc_done.json").exists() or (args.check and out.name.endswith("_bccheck"))
    resume_steps = history[-1]["steps"] if history else 0

    # different scenario streams every session, still reproducible from the step count
    env = VecMonitor(SubprocVecEnv([sim.training_env_fn(cfg["tier"], rank=i, seed=cfg["seed"] + 1000 * i + resume_steps // 1000)
                                    for i in range(cfg["envs"])], start_method="spawn"))
    if (out / "model_last.zip").exists():
        from avtest.tactical import CONTROLLER_ENTRY, SUT_CONTROLLER, controller_for_model
        import zipfile
        with zipfile.ZipFile(out / "model_last.zip") as archive:
            recorded = CONTROLLER_ENTRY in archive.namelist()
        if recorded and controller_for_model(out / "model_last.zip", None) != SUT_CONTROLLER:
            raise SystemExit("this warm start was trained with another controller (hold_lane); PPO's training env "
                             "uses SUT_CONTROLLER and its checkpoints would not record one: not wired up yet")
        model = PPO.load(out / "model_last.zip", env=env, device="cpu", **{k: cfg[k] for k in PPO_SETTINGS})
    else:
        model = PPO("MlpPolicy", env, seed=cfg["seed"], device="cpu", verbose=0,
                    policy_kwargs={"net_arch": {"pi": cfg["net"], "vf": cfg["net"]}}, **{k: cfg[k] for k in PPO_SETTINGS})
    started = time.time()
    print(f"driver {out.name}: {model.num_timesteps:,}/{cfg['total_steps']:,} steps done"
          f"{' (from imitation warm start)' if warm_start else ''}; session budget {args.hours} h", flush=True)

    def session_over() -> str | None:
        return control_state(out) or ("hours used" if time.time() - started > args.hours * 3600 else None)

    try:
        while model.num_timesteps < stop_at:
            if session_over():
                print(f"session ends ({session_over()}); continue with the same command", flush=True)
                break
            warmup = warm_start and model.num_timesteps < cfg["critic_warmup_steps"]
            steps = (cfg["critic_warmup_steps"] - model.num_timesteps) if warmup else cfg["chunk_steps"]
            # after an imitation warm start the driver is already competent: train on the full distribution it
            # is validated and qualified on (pilot 3: an easy-curriculum start taught it to speed; LOG.md 2026-11-01)
            d = 1.0 if warm_start else difficulty(model.num_timesteps, cfg)
            env.env_method("set_difficulty", d)
            for param in actor_parameters(model):
                param.requires_grad_(not warmup)
            monitor = make_monitor(out)
            t0 = time.time()
            model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False, callback=monitor)
            train_s = time.time() - t0
            for param in actor_parameters(model):
                param.requires_grad_(True)
            model.save(out / "model_last.zip")
            if monitor.stopped:
                print(f"STOP: saved model_last.zip at {model.num_timesteps:,} steps; continue with the same command",
                      flush=True)
                break
            done = model.num_timesteps
            label = f"steps_{done:010d}"
            (out / "checkpoints").mkdir(exist_ok=True)
            model.save(out / "checkpoints" / f"{label}.zip")
            row = {"steps": done, "phase": "critic_warmup" if warmup else "ppo", "difficulty": d, "train_s": train_s,
                   "steps_per_s": steps / train_s, **monitor.summary(), **ppo_diagnostics(model), "utc": time.time()}
            if not warmup:                              # the policy did not change during warm-up
                row.update(progress.checkpoint_report(out / "model_last.zip", out, label,
                                                      workers=cfg["val_workers"], panel=panel, tier=cfg["tier"],
                                                      t1_panel=min(panel, 100)))
                if row["val_success"] > best:
                    best = row["val_success"]
                    model.save(out / "model_best.zip")
                row["best_val_success"] = best
            log.append(row)
            progress.chart(log.read(), "steps", "training steps", out / "progress.png", f"{out.name}: PPO")
            val = (f" | {progress.headline(row)} best {progress.pct(best)}" if not warmup
                   else " | critic warm-up (policy frozen)")
            print(f"{done:>11,} steps | difficulty {d:.2f} | train episodes {row['train_episodes']} "
                  f"success {progress.pct(row['train_success'])}{val} | {row['steps_per_s']:.0f} steps/s", flush=True)
    finally:
        env.close()
    if model.num_timesteps >= cfg["total_steps"]:
        write_json(out / "done.json", {"steps": model.num_timesteps, "best_val_success": best})


if __name__ == "__main__":
    main()
