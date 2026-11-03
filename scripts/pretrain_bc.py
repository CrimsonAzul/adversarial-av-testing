"""Warm start for the driver: imitation of a cautious-IDM teacher with DAgger, before PPO.

    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d2                 # ~2–2.5 h, resumable
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d2 --check         # few-minute pipeline check
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d2 --pilot         # ~45 min learning pilot (real settings, fewer rounds)

The student sees lidar, lane lines, road edges, its own state and the 4 nearest vehicles
(avtest.sim), as PPO does. It outputs tactical decisions (lane command, throttle/brake;
avtest.tactical), and a PID controller steers. The teacher (--teacher, avtest.tactical.TEACHERS)
reads privileged map state; at every visited state it is labelled with the lane it steers for
(left/keep/right of the ego's current lane) and its throttle/brake. Default ``scene_passing_60``:
cautious IDM at 60 km/h, slowing for curves, that passes a stopped object in its lane as soon as it
sees it (from d4). Run d3 used the same rules at 20 km/h (``scene_passing``); run d2 used plain
cautious IDM (``cautious_idm``), whose overtaking depends on its own speed, which under DAgger is the
student's (docs/LOG.md, 2026-11-02). The teacher drives
through the same controller with those labels, which reproduces its own driving: 40/42 successes
either way (docs/LOG.md, 2026-11-01).
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

Disclosure: the teacher is (a variant of) cautious IDM, one of the three reference drivers.
Avoidability needs any one reference to complete a scene, so this cannot make a failure look
valid; it can only make the SUT share some of the cautious IDM's blind spots (reported in the thesis).
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
LANE_CHANGE_WEIGHT = 10.0


def _collect(task: dict) -> dict:
    import torch as th
    th.set_num_threads(1)
    from stable_baselines3 import PPO
    from avtest import scenario as sc, seeds
    from avtest import tactical
    from avtest.sim import OBS_DIM, AVTestEnv
    rng = np.random.default_rng(task["seed"])
    env = AVTestEnv()
    from avtest.policy import load_policy
    student = load_policy(task["model"]) if task["model"] else None
    obs_list, act_list, outcomes = [], [], []
    try:
        for _ in range(task["episodes"]):
            episode_obs, episode_act = [], []
            env.load(sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), task["tier"]))
            obs, _ = env.reset()
            teacher = tactical.TEACHERS[task["teacher"]](env.agent, env.scenario.map_seed)
            controller = tactical.TacticalController(**task["controller"])
            teacher_drives = student is None or rng.random() < task["beta"]
            step = 0
            while True:
                _, acceleration = teacher.act(env.agent.id)        # also records the lane the teacher steers for
                label = np.array([tactical.teacher_command(teacher, env.agent), np.clip(acceleration, -1, 1)],
                                 dtype=np.float32)
                if step % KEEP_EVERY == 0:
                    episode_obs.append(obs.astype(np.float16))
                    episode_act.append(label)
                decision = label if teacher_drives else student.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(controller.low_level(env.agent, decision))
                step += 1
                if term or trunc:
                    outcomes.append((bool(teacher_drives), info["outcome"]))
                    break
            # a demonstration the teacher failed teaches the failure: keep only successful teacher episodes;
            # student-driven episodes are kept (DAgger needs the teacher's corrections from the student's mistakes)
            if not teacher_drives or info["outcome"] == "success":
                obs_list.extend(episode_obs)
                act_list.extend(episode_act)
    finally:                          # an error must not leave the engine open (one per process)
        env.close()
    return {"obs": np.asarray(obs_list, dtype=np.float16).reshape(-1, OBS_DIM),
            "act": np.asarray(act_list, dtype=np.float32).reshape(-1, 2), "outcomes": outcomes}


def fit(model, obs: np.ndarray, act: np.ndarray, epochs: int, seed: int, lane_change_weight: float = LANE_CHANGE_WEIGHT) -> dict:
    """Regress the PPO actor's mean action onto the teacher's (lane command, throttle) by weighted MSE.

    Lane changes are about 2% of steps; their samples get LANE_CHANGE_WEIGHT so the student does not
    learn to always keep its lane (and then wait behind a broken-down car). The exploration noise
    (log_std) is set separately before PPO, and the value head is untouched.
    """
    policy = model.policy
    params = [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(),
              *policy.features_extractor.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    Y = torch.as_tensor(act)
    W = torch.where(Y[:, 0].abs() > 0.5, lane_change_weight, 1.0)[:, None]
    mse = mae = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(Y), generator=gen)
        total_mse = total_mae = 0.0
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            x = torch.as_tensor(obs[idx.numpy()].astype(np.float32))
            mean = policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(x)))
            loss = ((mean - Y[idx]) ** 2 * W[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item() * len(idx)
            total_mae += (mean.detach().clamp(-1, 1) - Y[idx]).abs().mean().item() * len(idx)
        mse, mae = total_mse / len(order), total_mae / len(order)
    command = np.sign(np.round(act[:, 0]))
    with torch.no_grad():
        sample = torch.as_tensor(obs[: min(len(obs), 20_000)].astype(np.float32))
        predicted = policy.action_net(policy.mlp_extractor.forward_actor(policy.extract_features(sample))).numpy()
    chosen = np.where(predicted[:, 0] > 0.5, 1, np.where(predicted[:, 0] < -0.5, -1, 0))
    changes = command[: len(chosen)] != 0
    return {"weighted_mse": mse, "action_mae": mae,
            "lane_command_agreement": float(np.mean(chosen == command[: len(chosen)])),
            "lane_change_recall": float(np.mean(chosen[changes] == command[: len(chosen)][changes])) if changes.any() else None}


def same_run(previous: dict, settings: dict) -> bool:
    """Whether a run may continue with these settings: all equal, except that more rounds may be added."""
    rest = lambda d: {k: v for k, v in d.items() if k != "rounds"}
    return rest(previous) == rest(settings) and settings["rounds"] >= previous["rounds"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=600, help="episodes per round")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--check", action="store_true")
    p.add_argument("--pilot", action="store_true", help="3 rounds x 240 episodes, 100-scene validation")
    p.add_argument("--teacher", default="scene_passing_60", choices=("scene_passing_60", "scene_passing", "cautious_idm"))
    p.add_argument("--head", default="classify", choices=("classify", "mean"),
                   help="classify: avtest.policy.TacticalModel, lane decision as a 3-way class (d8 on); "
                        "mean: the PPO actor's mean regressed onto the labels (d2-d7)")
    p.add_argument("--throttle-head", default="bins", choices=("bins", "regress"),
                   help="classify head only: throttle/brake as the most likely of avtest.policy.THROTTLE_BINS "
                        "(d9 on) or regressed as one value (d8)")
    p.add_argument("--controller", default="level", choices=("level", "smooth", "hold"),
                   help="60 km/h teacher only: lane-relative commands (d4 on) or the hold_lane controller, which "
                        "made the teacher itself worse (96.4% vs 98.0% on T1; LOG.md, 2026-11-03)")
    p.add_argument("--lane-change-weight", type=float, default=LANE_CHANGE_WEIGHT,
                   help="loss weight of lane-change samples (d2-d6: 10; d7: 3)")
    p.add_argument("--tier", default=TIER, choices=("T1", "T2"),
                   help="training scenes; with T1 the best round is chosen on 200 T1 validation scenes (from d5)")
    args = p.parse_args()
    from stable_baselines3 import PPO
    from scripts.train_driver import DEFAULTS
    from avtest import progress
    from avtest.sim import OBS_DIM
    import gymnasium as gym

    from avtest import tactical
    if args.teacher != "scene_passing_60":
        controller = tactical.LEGACY_CONTROLLER
    else:
        controller = {"hold": tactical.HOLD_CONTROLLER, "smooth": tactical.SMOOTH_CONTROLLER,
                      "level": tactical.SUT_CONTROLLER}[args.controller]
    out = ROOT / "runs/driver" / (args.name + ("_bccheck" if args.check else "_pilot" if args.pilot else ""))
    panel = progress.VAL_PANEL
    if args.check:
        args.rounds, args.episodes, args.workers, args.epochs, panel = 1, 4, 2, 1, 4
    elif args.pilot:
        args.rounds, args.episodes, panel = 2, 240, 100
    if (out / "train_log.jsonl").exists():
        raise SystemExit(f"{out} already has PPO training; pretrain into a new --name")
    settings = {"rounds": args.rounds, "episodes": args.episodes, "epochs": args.epochs, "keep_every": KEEP_EVERY,
                "tier": args.tier, "teacher": {"cautious_idm": "idm_cautious"}.get(args.teacher, args.teacher),   # d2's spelling
                "drop_teacher_failures": True, "val_panel": panel, "controller": controller,
                "action": "tactical", "lane_change_weight": args.lane_change_weight, "head": args.head,
                "throttle_head": args.throttle_head if args.head == "classify" else None}
    if (out / "bc_config.json").exists() and not same_run(read_json(out / "bc_config.json"), settings):
        raise SystemExit(f"{out} was started with other settings; use a new --name")
    write_json(out / "bc_config.json", settings)
    log = JsonlLog(out / "bc_log.jsonl")
    done_rounds = [r["round"] for r in log.read()]

    class Spaces(gym.Env):                              # spaces only, to build the PPO policy offline
        observation_space = gym.spaces.Box(-0.0, 1.0, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    from avtest.policy import THROTTLE_BINS, TacticalModel, fit_classify, load_policy
    if done_rounds:
        model = load_policy(out / "model_last.zip")
        print(f"resuming after round {done_rounds[-1]}", flush=True)
    elif args.head == "classify":
        model = TacticalModel(OBS_DIM, tuple(DEFAULTS["net"]), seed=DEFAULTS["seed"],
                              throttle_bins=THROTTLE_BINS if args.throttle_head == "bins" else None)
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
        tasks = [{"seed": 10_000 * rnd + w, "episodes": per, "model": current, "beta": 0.5 ** rnd, "teacher": args.teacher,
                  "tier": args.tier, "controller": controller}
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
        if args.head == "classify":
            fit_stats = fit_classify(model, obs, act, args.epochs, rnd)
        else:
            fit_stats = fit(model, obs, act, args.epochs, rnd, args.lane_change_weight)
            with torch.no_grad():                     # PPO later starts with small exploration noise (σ ≈ 0.22)
                model.policy.log_std.fill_(INITIAL_LOG_STD)
        model.save(out / "model_last.zip")
        (out / "checkpoints").mkdir(exist_ok=True)
        model.save(out / "checkpoints" / f"bc_round_{rnd:02d}.zip")
        for saved in (out / "model_last.zip", out / "checkpoints" / f"bc_round_{rnd:02d}.zip"):
            tactical.stamp_controller(saved, controller)       # the driver drives with what it trained with
        del obs, act, shards
        outcomes = [o for r in results for o in r["outcomes"]]
        rollout = [o for t, o in outcomes if not t]
        report = progress.checkpoint_report(out / "model_last.zip", out, f"bc_round_{rnd:02d}", workers=min(4, args.workers),
                                            panel=panel, tier=progress.TIER, t1_panel=panel if args.tier == "T1" else min(panel, 100))
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
    select = "val_t1_success" if args.tier == "T1" else "val_success"      # the training tier's validation
    best = max(rows, key=lambda r: (r[select], r["round"]))
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_last.zip")
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_best_bc.zip")
    write_json(out / "bc_done.json", {"rounds": args.rounds, "best_round": best["round"],
                                      "best_val_success": best["val_success"], "selected_on": select, "final": rows[-1]})
    print(f"warm start: best round {best['round']} ({select} {progress.pct(best[select])}) saved as "
          f"{out / 'model_last.zip'}; recordings in {out / 'recordings'}; "
          f"continue with: python -m scripts.train_driver --name {args.name}", flush=True)


if __name__ == "__main__":
    main()
