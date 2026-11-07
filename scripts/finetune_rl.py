"""PPO fine-tuning of an end-to-end driver from its imitation weights (avtest.finetune; LOG.md, 2026-11-07).

    .venv\\Scripts\\python.exe -m scripts.finetune_rl --name r1 --init runs\\driver\\d18\\model_best_bc.zip

Each iteration, --workers simulator processes drive --episodes training scenes (tier of each drawn from --tiers) with
the current network, its throttle level sampled from its throttle policy and its lane intent and steering as when it
drives. Then the critic is updated, and after --critic-warmup iterations the policy too (avtest.finetune.ppo_update:
clipped PPO, a KL penalty toward the imitation network, distillation of intent and steering). The reward is the
training environment's: progress, speed, success +20, failure (collision, off-road, stuck, timeout) -30.

Every --val-every iterations the network is saved as checkpoints/rl_iter_<k>.zip, a driver like any other (no
controller; the most likely throttle level), and validated as imitation rounds are: the 200-scene validation panel as
T2, its first 100 as T1 and all 200 as T3. model_best_rl.zip is the checkpoint with the best mean of the three.
Logged to rl_log.jsonl. The same command continues a stopped run; a file named STOP in the run folder stops it after
the current iteration.
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import shutil
import time
from collections import Counter
from pathlib import Path

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest.records import JsonlLog, ROOT, read_json, sha256_file, write_json

_worker: dict = {}
DEFAULT_TIERS = "T1:0.2,T2:0.3,T3:0.5"
CHECK_PANEL = 4


def _init(init_path: str) -> None:
    import torch as th
    th.set_num_threads(1)
    from avtest.policy import load_policy
    from avtest.sim import AVTestEnv, sensor_config
    model = load_policy(Path(init_path))
    _worker.update(model=model, env=AVTestEnv(sensor_config(model.obs_dim)))


def _rollout(task: dict) -> list[dict]:
    """One worker's episodes with the weights in the task: per step the float16 observation, the throttle level
    drawn and its log-probability, and the environment's reward."""
    import torch as th
    from avtest import scenario as sc, seeds
    from avtest.finetune import sample_action
    from avtest.policy import FrameHistory
    model, env = _worker["model"], _worker["env"]
    model.net.load_state_dict(th.load(io.BytesIO(task["weights"]), map_location="cpu", weights_only=True))
    model.net.eval()
    rng = np.random.default_rng(task["seed"])
    episodes = []
    for _ in range(task["episodes"]):
        tier = task["tiers"][int(rng.choice(len(task["tiers"]), p=task["probs"]))]
        env.load(sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), tier))
        obs, _ = env.reset()
        history = FrameHistory()
        observations, levels, logps, rewards = [], [], [], []
        while True:
            features = history(obs)
            observations.append(np.asarray(obs, dtype=np.float16))
            action, level, logp = sample_action(model, features, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            levels.append(level)
            logps.append(logp)
            rewards.append(reward)
            if terminated or truncated:
                break
        episodes.append({"obs": np.asarray(observations), "levels": np.asarray(levels, dtype=np.int64),
                         "logp": np.asarray(logps, dtype=np.float32), "rewards": np.asarray(rewards, dtype=np.float32),
                         "outcome": info["outcome"], "tier": tier})
    return episodes


def parse_tiers(text: str) -> tuple[list[str], list[float]]:
    pairs = [item.split(":") for item in text.split(",")]
    tiers, weights = [t for t, _ in pairs], np.array([float(w) for _, w in pairs])
    if set(tiers) - {"T1", "T2", "T3"} or len(set(tiers)) != len(tiers) or (weights <= 0).any():
        raise SystemExit(f"--tiers {text}: tier:weight pairs of T1, T2, T3")
    return tiers, list(weights / weights.sum())


def state_bytes(model) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.net.state_dict(), buffer)
    return buffer.getvalue()


def save_driver(model, path: Path) -> None:
    """The network as a driver file: EndToEndModel's format, recording that no controller acts on it."""
    from avtest import tactical
    model.save(path)
    tactical.stamp_controller(path, tactical.END_TO_END)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--init", type=Path, required=True, help="the imitation network to start from (an end-to-end driver)")
    p.add_argument("--iterations", type=int, default=None, help="default 300 (2 with --check)")
    p.add_argument("--episodes", type=int, default=64, help="training episodes per iteration")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tiers", default=DEFAULT_TIERS, help="tier:weight of the training scenes")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--distill-coef", type=float, default=10.0)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--critic-warmup", type=int, default=5, help="iterations that train only the critic")
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--check", action="store_true", help="2 workers, 4 episodes, 2 iterations, 4-scene validation")
    args = p.parse_args()

    from avtest import progress
    from avtest.finetune import (REWARD_SCALE, ValueNet, batched, critic_update, episode_features, explained_variance,
                                 gae, ppo_update)
    from avtest.policy import EndToEndModel, is_end_to_end_model
    from avtest.tactical import controller_for_model

    init = args.init if args.init.is_absolute() else ROOT / args.init
    if not is_end_to_end_model(init) or not controller_for_model(init, 0).get("end_to_end"):
        raise SystemExit(f"{init} is not an end-to-end driver")
    out = ROOT / "runs/driver" / (args.name + ("_rlcheck" if args.check else ""))
    panel, t1_panel = progress.VAL_PANEL, 100
    if args.check:
        args.workers, args.episodes, args.critic_warmup, args.val_every = 2, 4, 1, 2
        panel = t1_panel = CHECK_PANEL
    iterations = args.iterations or (2 if args.check else 300)
    tiers, probs = parse_tiers(args.tiers)
    settings = {"init": str(init.relative_to(ROOT)) if init.is_relative_to(ROOT) else str(init),
                "init_sha256": sha256_file(init), "episodes": args.episodes, "tiers": dict(zip(tiers, probs)),
                "lr": args.lr, "critic_lr": args.critic_lr, "kl_coef": args.kl_coef, "distill_coef": args.distill_coef,
                "clip": args.clip, "epochs": args.epochs, "target_kl": args.target_kl, "gamma": args.gamma,
                "lam": args.lam, "critic_warmup": args.critic_warmup, "val_every": args.val_every,
                "reward_scale": REWARD_SCALE, "val_panel": panel, "policy": "throttle levels sampled; intent, steering "
                "as driven and distilled to the init"}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rl_config.json").exists() and read_json(out / "rl_config.json") != settings:
        raise SystemExit(f"{out} was started with other settings; use a new --name")
    write_json(out / "rl_config.json", settings)
    log = JsonlLog(out / "rl_log.jsonl")

    ref = EndToEndModel.load(init)                       # the imitation network: KL and distillation target
    model = EndToEndModel.load(init)
    critic = ValueNet.for_model(model)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    start, best = 0, {"value": -1.0, "iteration": None}
    state_path = out / "rl_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.net.load_state_dict(state["policy"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        start, best = state["iteration"], state["best"]
        print(f"resuming after iteration {start}", flush=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    ctx = mp.get_context("spawn")
    pool = None
    try:
        for it in range(start, iterations):
            if (out / "STOP").exists():
                print(f"STOP: stopped before iteration {it + 1}; continue with the same command", flush=True)
                break
            t0 = time.time()
            if pool is None:
                pool = ctx.Pool(args.workers, initializer=_init, initargs=(str(init),))
            weights = state_bytes(model)
            counts = [args.episodes // args.workers + (w < args.episodes % args.workers) for w in range(args.workers)]
            tasks = [{"weights": weights, "seed": 1_000_000 + 1000 * it + w, "episodes": n, "tiers": tiers,
                      "probs": probs} for w, n in enumerate(counts) if n]
            episodes = [e for part in pool.map(_rollout, tasks) for e in part]
            collect_s = time.time() - t0

            features = torch.as_tensor(np.concatenate([episode_features(e["obs"]) for e in episodes]))
            levels = torch.as_tensor(np.concatenate([e["levels"] for e in episodes]))
            old_logp = torch.as_tensor(np.concatenate([e["logp"] for e in episodes]))
            values = batched(critic, features).numpy()
            advantages, returns, k = [], [], 0
            for e in episodes:
                n = len(e["rewards"])
                a, r = gae(e["rewards"] * REWARD_SCALE, values[k:k + n], args.gamma, args.lam)
                advantages.append(a)
                returns.append(r)
                k += n
            advantages, returns = np.concatenate(advantages), np.concatenate(returns)
            with torch.no_grad():
                recomputed = torch.log_softmax(batched(model.net, features)[2], -1).gather(1, levels[:, None])[:, 0]
            row = {"iteration": it + 1, "episodes": len(episodes), "steps": int(len(levels)), "collect_s": collect_s,
                   "logp_mismatch": float((recomputed - old_logp).abs().max()),
                   "explained_variance": explained_variance(values, returns),
                   "mean_return": float(np.mean([e["rewards"].sum() for e in episodes]))}
            for tier in tiers:
                mine = [e for e in episodes if e["tier"] == tier]
                row[f"train_{tier.lower()}"] = dict(Counter(e["outcome"] for e in mine))
                row[f"train_{tier.lower()}_success"] = (sum(e["outcome"] == "success" for e in mine) / len(mine)
                                                        if mine else None)
            row["train_success"] = float(np.mean([e["outcome"] == "success" for e in episodes]))
            row.update(critic_update(critic, features, torch.as_tensor(returns), critic_optimizer, args.epochs,
                                     seed=it))
            if it >= args.critic_warmup:
                row.update(ppo_update(model, ref, features, levels, old_logp, torch.as_tensor(advantages), optimizer,
                                      epochs=args.epochs, clip=args.clip, kl_coef=args.kl_coef,
                                      distill_coef=args.distill_coef, target_kl=args.target_kl, seed=it))
            save_driver(model, out / "model_last_rl.zip.tmp")
            (out / "model_last_rl.zip.tmp").replace(out / "model_last_rl.zip")

            if (it + 1) % args.val_every == 0 or it + 1 == iterations:
                pool.close()                              # free the workers' memory while validation runs its own
                pool.join()
                pool = None
                label = f"rl_iter_{it + 1:04d}"
                checkpoint = out / "checkpoints" / f"{label}.zip"
                if checkpoint.exists():
                    checkpoint.unlink()                   # a resumed run repeats an iteration that was not logged
                save_driver(model, checkpoint)
                report = progress.checkpoint_report(checkpoint, out, label, workers=args.workers, panel=panel,
                                                    record=False, tier=progress.TIER, t1_panel=t1_panel,
                                                    t3_panel=panel)
                report["val_all_success"] = float(np.mean([report["val_t1_success"], report["val_success"],
                                                           report["val_t3_success"]]))
                row.update(report)
                if report["val_all_success"] > best["value"]:
                    best = {"value": report["val_all_success"], "iteration": it + 1}
                    shutil.copy2(checkpoint, out / "model_best_rl.zip")
            row["wall_s"] = time.time() - t0
            torch.save({"iteration": it + 1, "policy": model.net.state_dict(), "critic": critic.state_dict(),
                        "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                        "best": best}, out / "rl_state.pt.tmp")
            (out / "rl_state.pt.tmp").replace(state_path)
            log.append(row)
            train = " ".join(f"{t} {progress.pct(row[f'train_{t.lower()}_success'])}" for t in tiers
                             if row[f"train_{t.lower()}_success"] is not None)
            ppo = (f" | kl_ref {row['kl_ref']:.4f} approx_kl {row['approx_kl']:.4f} clip {row['clip_fraction']:.2f}"
                   f" epochs {row['epochs_run']}" if "kl_ref" in row else " | critic only")
            val = (f" | val T1 {progress.pct(row['val_t1_success'])} T2 {progress.pct(row['val_success'])} "
                   f"T3 {progress.pct(row['val_t3_success'])}" if "val_success" in row else "")
            print(f"iter {it + 1}: {row['episodes']} episodes, {row['steps']:,} steps, sampled success {train} | "
                  f"return {row['mean_return']:.1f} | value ev {row['explained_variance']:.2f}{ppo}{val} | "
                  f"{row['wall_s'] / 60:.1f} min", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
    done = len(log.read())
    if done >= iterations:
        write_json(out / "rl_done.json", {"iterations": done, "best": best})
    print(f"{done} iterations done; best: iteration {best['iteration']} (mean validation "
          f"{progress.pct(best['value']) if best['iteration'] else '-'}) -> {out / 'model_best_rl.zip'}", flush=True)


if __name__ == "__main__":
    main()
