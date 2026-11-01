"""Warm start for the driver: imitation of the cautious-IDM teacher with DAgger, before PPO.

    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d2                 # ~2–2.5 h, resumable
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d2 --check         # few-minute pipeline check

The student sees lidar, its own state and the 4 nearest vehicles (avtest.sim), as PPO does. The teacher
(MetaDrive IDM with cautious settings, avtest.drivers.CautiousIDMDriver) reads privileged
map state and labels every visited state with the action it would take.
Round 0 lets the teacher drive. In rounds 1..R the student drives, with the teacher taking
over with probability β = 0.5^round each episode, while the teacher still labels every state
(DAgger; Ross, Gordon & Bagnell 2011). The policy is the PPO actor itself, trained by regressing
its mean action onto the teacher's actions on all data so far, so scripts.train_driver continues
with PPO from runs/driver/<name>/model_last.zip.

Training scenarios are uniform over tier T2 (docs/LOG.md, 2026-11-01). Episodes the teacher
drove and failed are discarded, so the student never imitates a crash; student-driven
episodes are always kept, because their labels are the teacher's corrections.

Data: every 2nd step is kept (consecutive 0.1 s frames are nearly identical) and stored as
float16 shards under data/, one per round, so a crash resumes at the last finished round.
After every round: validation on the fixed 200-scene panel and recordings of the three
showcase scenarios (avtest.progress), logged to bc_log.jsonl and charted in bc_progress.png.
At the end the round with the best validation becomes model_last.zip, PPO's starting point.

Disclosure: the teacher is also one of the three reference drivers. Avoidability needs any
one reference to complete a scene, so this cannot make a failure look valid; it can only
make the SUT share some of the cautious IDM's blind spots (reported in the thesis).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time

import torch
import numpy as np

from avtest.records import JsonlLog, ROOT, read_json, write_json

TIER = "T2"                  # see docs/LOG.md, 2026-11-01: reference ceiling check
KEEP_EVERY = 2
INITIAL_LOG_STD = -1.5


def _collect(task: dict) -> dict:
    import torch as th
    th.set_num_threads(1)
    from stable_baselines3 import PPO
    from avtest import scenario as sc, seeds
    from avtest.drivers import CautiousIDMDriver
    from avtest.sim import OBS_DIM, AVTestEnv
    rng = np.random.default_rng(task["seed"])
    env = AVTestEnv()
    teacher = CautiousIDMDriver()
    student = PPO.load(task["model"], device="cpu") if task["model"] else None
    obs_list, act_list, outcomes = [], [], []
    for _ in range(task["episodes"]):
        episode_obs, episode_act = [], []
        env.load(sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), TIER))
        obs, _ = env.reset()
        teacher.start(env)
        teacher_drives = student is None or rng.random() < task["beta"]
        step = 0
        while True:
            label = np.clip(np.asarray(teacher.act(env, obs), dtype=np.float32), -1, 1)
            if step % KEEP_EVERY == 0:
                episode_obs.append(obs.astype(np.float16))
                episode_act.append(label)
            action = label if teacher_drives else student.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            step += 1
            if term or trunc:
                outcomes.append((bool(teacher_drives), info["outcome"]))
                break
        # a demonstration the teacher failed teaches the failure: keep only successful teacher episodes;
        # student-driven episodes are kept (DAgger needs the teacher's corrections from the student's mistakes)
        if not teacher_drives or info["outcome"] == "success":
            obs_list.extend(episode_obs)
            act_list.extend(episode_act)
    env.close()
    return {"obs": np.asarray(obs_list, dtype=np.float16).reshape(-1, OBS_DIM),
            "act": np.asarray(act_list, dtype=np.float32).reshape(-1, 2), "outcomes": outcomes}


def fit(model, obs: np.ndarray, act: np.ndarray, epochs: int, seed: int) -> dict:
    """Regress the PPO actor's mean action onto the teacher's actions (mean squared error).

    The teacher is deterministic, so this is plain behaviour cloning of the action mean; the
    exploration noise (log_std) is set separately before PPO, and the value head is untouched.
    """
    policy = model.policy
    params = [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(),
              *policy.features_extractor.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    Y = torch.as_tensor(act)
    mse = mae = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(Y), generator=gen)
        total_mse = total_mae = 0.0
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            x = torch.as_tensor(obs[idx.numpy()].astype(np.float32))
            mean = policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(x)))
            loss = torch.nn.functional.mse_loss(mean, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item() * len(idx)
            total_mae += (mean.detach().clamp(-1, 1) - Y[idx]).abs().mean().item() * len(idx)
        mse, mae = total_mse / len(order), total_mae / len(order)
    return {"mse": mse, "action_mae": mae}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--episodes", type=int, default=600, help="episodes per round")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    from stable_baselines3 import PPO
    from scripts.train_driver import DEFAULTS
    from avtest import progress
    from avtest.sim import OBS_DIM
    import gymnasium as gym

    out = ROOT / "runs/driver" / (args.name + ("_bccheck" if args.check else ""))
    panel = progress.VAL_PANEL
    if args.check:
        args.rounds, args.episodes, args.workers, args.epochs, panel = 1, 4, 2, 1, 4
    if (out / "train_log.jsonl").exists():
        raise SystemExit(f"{out} already has PPO training; pretrain into a new --name")
    settings = {"rounds": args.rounds, "episodes": args.episodes, "epochs": args.epochs, "keep_every": KEEP_EVERY,
                "tier": TIER, "teacher": "idm_cautious", "drop_teacher_failures": True, "val_panel": panel}
    if (out / "bc_config.json").exists() and read_json(out / "bc_config.json") != settings:
        raise SystemExit(f"{out} was started with other settings; use a new --name")
    write_json(out / "bc_config.json", settings)
    log = JsonlLog(out / "bc_log.jsonl")
    done_rounds = [r["round"] for r in log.read()]

    class Spaces(gym.Env):                              # spaces only, to build the PPO policy offline
        observation_space = gym.spaces.Box(-0.0, 1.0, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    if done_rounds:
        model = PPO.load(out / "model_last.zip", device="cpu")
        print(f"resuming after round {done_rounds[-1]}", flush=True)
    else:
        model = PPO("MlpPolicy", Spaces(), n_steps=DEFAULTS["n_steps"], batch_size=DEFAULTS["batch_size"],
                    n_epochs=DEFAULTS["n_epochs"], learning_rate=DEFAULTS["learning_rate"], gamma=DEFAULTS["gamma"],
                    gae_lambda=DEFAULTS["gae_lambda"], clip_range=DEFAULTS["clip_range"], ent_coef=DEFAULTS["ent_coef"],
                    target_kl=DEFAULTS["target_kl"], seed=DEFAULTS["seed"], device="cpu",
                    policy_kwargs={"net_arch": {"pi": DEFAULTS["net"], "vf": DEFAULTS["net"]}})
    ctx = mp.get_context("spawn")
    for rnd in range(args.rounds + 1):
        if rnd in done_rounds:
            continue
        t0 = time.time()
        current = None if rnd == 0 else str(out / "model_last.zip")
        per = max(1, args.episodes // args.workers)
        tasks = [{"seed": 10_000 * rnd + w, "episodes": per, "model": current, "beta": 0.5 ** rnd}
                 for w in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_collect, tasks)
        shard = out / "data" / f"round_{rnd:02d}.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez(shard, obs=np.concatenate([r["obs"] for r in results]), act=np.concatenate([r["act"] for r in results]))
        collect_s = time.time() - t0
        shards = [np.load(out / "data" / f"round_{k:02d}.npz") for k in range(rnd + 1)]
        obs = np.concatenate([s["obs"] for s in shards])
        act = np.concatenate([s["act"] for s in shards])
        n_samples = int(len(act))
        fit_stats = fit(model, obs, act, args.epochs, rnd)
        with torch.no_grad():                         # PPO later starts with small exploration noise (σ ≈ 0.22)
            model.policy.log_std.fill_(INITIAL_LOG_STD)
        model.save(out / "model_last.zip")
        (out / "checkpoints").mkdir(exist_ok=True)
        model.save(out / "checkpoints" / f"bc_round_{rnd:02d}.zip")
        del obs, act, shards
        outcomes = [o for r in results for o in r["outcomes"]]
        rollout = [o for t, o in outcomes if not t]
        report = progress.checkpoint_report(out / "model_last.zip", out, f"bc_round_{rnd:02d}", workers=min(4, args.workers),
                                            panel=panel)
        row = {"round": rnd, "samples": n_samples, **fit_stats, "collect_s": collect_s, "wall_s": time.time() - t0,
               "teacher_episodes": sum(t for t, _ in outcomes),
               "teacher_failures_dropped": sum(t and o != "success" for t, o in outcomes),
               "student_rollouts": len(rollout),
               "student_rollout_success": (sum(o == "success" for o in rollout) / len(rollout)) if rollout else None,
               **report}
        log.append(row)
        progress.chart(log.read(), "round", "DAgger round", out / "bc_progress.png", f"{out.name}: imitation warm start")
        print(f"round {rnd}: {row['samples']:,} samples | action error {fit_stats['action_mae']:.3f} | "
              f"{progress.headline(report)} | {row['wall_s'] / 60:.1f} min", flush=True)
    # DAgger rolls out the latest student each round; PPO starts from the round that validated best
    rows = log.read()
    best = max(rows, key=lambda r: (r["val_success"], r["round"]))
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_last.zip")
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_best_bc.zip")
    write_json(out / "bc_done.json", {"rounds": args.rounds, "best_round": best["round"],
                                      "best_val_success": best["val_success"], "final": rows[-1]})
    print(f"warm start: best round {best['round']} (validation {progress.pct(best['val_success'])}) saved as "
          f"{out / 'model_last.zip'}; recordings in {out / 'recordings'}; "
          f"continue with: python -m scripts.train_driver --name {args.name}", flush=True)


if __name__ == "__main__":
    main()
