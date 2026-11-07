"""Train a driver by imitation of a rule-based teacher with DAgger (Ross, Gordon & Bagnell 2011).

    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d11 --tier T1 --controller guard   # d11, ~1-1.5 h, resumable
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d11 --tier T1 --controller guard --check   # few-minute pipeline check
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name dX --pilot          # learning pilot: 3 rounds x 240 episodes

Student (--head classify, the default from d8 on): avtest.policy.TacticalModel. It sees the 800 values of
avtest.sim (720 lidar beams to 100 m, 24 lane-line and 24 road-edge beams, the 4 nearest vehicles, its own
state) through two hidden layers of 256 (tanh) and has two heads:
- lane: 3 classes (left / keep / right), cross-entropy. When driving, its command is P(left) - P(right) and
  the controller changes lane above its threshold. Up to d7 (--head mean) the lane command was regressed
  onto -1/0/+1, which averaged conflicting labels: d6 weaved, d7 hesitated (docs/LOG.md, 2026-11-03).
- throttle/brake: the most likely of the 9 levels avtest.policy.THROTTLE_BINS, cross-entropy (from d9 on;
  --throttle-head regress for d8's single regressed value, which almost never braked hard).
Steering is not learned: avtest.tactical.TacticalController steers for the lane the model chooses.

End-to-end student (--action e2e, from d14; author's decision, docs/LOG.md 2026-11-05):
avtest.policy.EndToEndModel outputs MetaDrive's steering and throttle, and nothing acts on them. Its inputs are the
800 values and their change over 0.2 s, standardised; its heads are the lane intent (3 classes), one steering value
per intent and the throttle level. Labels are the teacher's own final action, from the teacher's controller
(--controller; its rules stay in the teacher), and the side of the lane that controller steered for. Half the
teacher's episodes get correlated steering noise (--dart, DART), so the data shows it steering back; every 20th
round-0 episode is held out for steering metrics. Stored samples also mark episode starts (``first``).

    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d14 --tier all --controller yield --action e2e

Teacher (--teacher, avtest.tactical.TEACHERS; default ``scene_passing_60``): cautious IDM at 60 km/h that
slows for curves and passes only by the scene: a warning triangle, a stopped vehicle (< 5 km/h), or one at
least 20 km/h below its cruise that it is closing in on by at least 10 km/h; it follows other traffic. Its labels are the lane it steers for
(left/keep/right of the ego's current lane, ``teacher_command``) and its throttle/brake. Run d3 used the same
idea at 20 km/h (``scene_passing``); d2 used plain cautious IDM (``cautious_idm``), whose overtaking depends
on its own speed, which under DAgger is the student's (docs/LOG.md, 2026-11-02).

Each round (rounds 0..--rounds, 6 by default; 600 episodes over 6 worker processes):
1. Scenes are drawn uniformly from --tier (default T2; d5-d11 used T1; T3 from d12, for a driver that is good on
   every tier; from d13 ``all``: each episode's tier drawn from T1, T2 and T3) with training-split map seeds.
2. Round 0: the teacher drives. Round k: the teacher drives an episode with probability 0.5^k, otherwise the
   latest student (model_last.zip) does. Either way the teacher labels every state, so the data holds the
   teacher's corrections from the states the student's own mistakes lead to.
3. Whoever decides, the decision goes through the controller (--controller), as it will when the model
   drives: from d10 the emergency-braking layer (aeb), from d11 also the lane-change guard and speed-scaled
   steering along the route (guard), from d13 also braking for traffic predicted to come in from the side, the
   check for vehicles alongside before a lane change and the low-speed lateral cap (yield). The student so
   trains on the states those layers create.
4. Episodes the teacher drove and failed are dropped, so a crash is never imitated; student episodes are kept.
   Every 2nd step is stored (0.1 s frames are nearly identical) as a float16 shard data/round_XX.npz.
5. The model is fitted for --epochs (20) on the data of all rounds so far, saved as model_last.zip and
   checkpoints/bc_round_XX.zip, and the controller is written into each zip (tactical.stamp_controller), so the
   model always drives with the controller it was trained with.
6. Validation on the fixed 200-scene panels (avtest.progress; with --tier T1 also 200 T1 scenes, with T3 or all
   also 200 T1 and 200 T3 scenes) and
   recordings of three showcase scenarios, logged to bc_log.jsonl and charted in bc_progress.png.
At the end the round with the best validation on the training tier becomes model_best_bc.zip (and
model_last.zip). Running the same command again skips finished rounds, so an interrupted run resumes at the
last finished round; other settings than the first run's are refused (more rounds may be added).

--head mean (d2-d7) builds the Stable-Baselines3 PPO actor instead and regresses its mean action onto the
labels (``fit``); only such a model can be fine-tuned further with PPO (scripts.train_driver).

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

from avtest.policy import E2E_HIDDEN, HISTORY_STEPS, stored_features
from avtest.records import JsonlLog, ROOT, read_json, write_json

TIER = "T2"                  # see docs/LOG.md, 2026-11-01: reference ceiling check
KEEP_EVERY = 2
INITIAL_LOG_STD = -1.5
LANE_CHANGE_WEIGHT = 10.0
TIERS_MIXED = ("T1", "T2", "T3")    # --tier all: each episode's tier drawn uniformly (from d13)
# --action e2e: DART noise (Laskey et al., 2017) on the steering of half the teacher's episodes, correlated over
# about 0.5 s, so the data shows the teacher steering back from off its line; the labels stay noise-free.
DART_SHARE, DART_RHO, DART_SIGMA = 0.5, 0.8, 0.02
HELDOUT_EVERY = 20                  # --action e2e: every 20th episode of round 0 is held out for steering metrics


def _collect(task: dict) -> dict:
    """One worker's episodes. ``action`` "tactical": labels (lane command, throttle) before the controller, and
    whoever drives goes through the controller. "e2e": labels (intent, steering, throttle) are the teacher's own
    final action from its controller, and the student's steering and throttle go to the simulator unchanged;
    teacher episodes get DART steering noise (``dart``) in half of them."""
    import torch as th
    th.set_num_threads(1)
    from avtest import scenario as sc, seeds
    from avtest import tactical
    from avtest.policy import FrameHistory, load_policy
    from avtest.sim import E2E_OBJ_OBS_DIM, E2E_OBS_DIM, OBS_DIM, AVTestEnv, sensor_config
    rng = np.random.default_rng(task["seed"])
    e2e = task.get("action", "tactical") == "e2e"
    obs_dim = (E2E_OBJ_OBS_DIM if task.get("object_list") else E2E_OBS_DIM) if task.get("lane_model") else OBS_DIM
    env = AVTestEnv(sensor_config(obs_dim))
    student = load_policy(task["model"]) if task["model"] else None
    obs_list, act_list, first_list, outcomes = [], [], [], []
    try:
        for _ in range(task["episodes"]):
            episode_obs, episode_act = [], []
            tier = task["tier"] if task["tier"] != "all" else TIERS_MIXED[int(rng.integers(len(TIERS_MIXED)))]
            env.load(sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), tier))
            obs, _ = env.reset()
            teacher = tactical.TEACHERS[task["teacher"]](env.agent, env.scenario.map_seed)
            controller = tactical.TacticalController(**task["controller"])
            teacher_drives = student is None or rng.random() < task["beta"]
            history = FrameHistory()
            noisy = e2e and teacher_drives and rng.random() < DART_SHARE
            noise = 0.0
            step = 0
            while True:
                _, acceleration = teacher.act(env.agent.id)        # also records the lane the teacher steers for
                command = tactical.teacher_command(teacher, env.agent)
                if e2e:                                            # the teacher's final action is the label
                    action = controller.low_level(env.agent, [command, float(np.clip(acceleration, -1, 1))])
                    label = np.array([tactical.steered_side(controller, env.agent), action[0], action[1]],
                                     dtype=np.float32)
                    features = history(obs)
                else:
                    label = np.array([command, np.clip(acceleration, -1, 1)], dtype=np.float32)
                if step % KEEP_EVERY == 0:
                    episode_obs.append(obs.astype(np.float16))
                    episode_act.append(label)
                if not e2e:
                    decision = label if teacher_drives else student.predict(obs, deterministic=True)[0]
                    executed = controller.low_level(env.agent, decision)
                elif teacher_drives:
                    executed = np.array(action, dtype=float)
                    if noisy:                                      # DART: the teacher's corrections from off its line
                        noise = DART_RHO * noise + task["dart"] * np.sqrt(1 - DART_RHO ** 2) * rng.standard_normal()
                        executed[0] = float(np.clip(executed[0] + noise, -1.0, 1.0))
                else:
                    executed = student.act_features(features)
                obs, _, term, trunc, info = env.step(executed)
                step += 1
                if term or trunc:
                    outcomes.append((bool(teacher_drives), info["outcome"]))
                    break
            # a demonstration the teacher failed teaches the failure: keep only successful teacher episodes;
            # student-driven episodes are kept (DAgger needs the teacher's corrections from the student's mistakes)
            if not teacher_drives or info["outcome"] == "success":
                obs_list.extend(episode_obs)
                act_list.extend(episode_act)
                first_list.extend([True] + [False] * (len(episode_obs) - 1))
    finally:                          # an error must not leave the engine open (one per process)
        env.close()
    return {"obs": np.asarray(obs_list, dtype=np.float16).reshape(-1, obs_dim),
            "act": np.asarray(act_list, dtype=np.float32).reshape(-1, 3 if e2e else 2),
            "first": np.asarray(first_list, dtype=bool), "outcomes": outcomes}


def load_rounds(data_dir, last_round: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """--action e2e: the stored samples of rounds 0..last_round (observations float16, labels, episode starts), each
    in one preallocated array so memory peaks at the data plus one round, and the held-out mask: every
    HELDOUT_EVERY-th episode of round 0 (teacher-driven), never trained on."""
    files = [np.load(data_dir / f"round_{k:02d}.npz") for k in range(last_round + 1)]
    labels = [f["act"] for f in files]
    n = sum(len(a) for a in labels)
    obs = np.empty((n, files[0]["obs"].shape[1]), dtype=np.float16)
    act = np.empty((n, labels[0].shape[1]), dtype=np.float32)
    first = np.empty(n, dtype=bool)
    heldout = np.zeros(n, dtype=bool)
    at = 0
    for k, (f, a) in enumerate(zip(files, labels)):
        size = len(a)
        obs[at:at + size], act[at:at + size], first[at:at + size] = f["obs"], a, f["first"]
        if k == 0:
            heldout[at:at + size] = (np.cumsum(first[at:at + size]) - 1) % HELDOUT_EVERY == 0
        at += size
    return obs, act, first, heldout


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


def selection_key(tier: str) -> str:
    """The validation result the best round is chosen on: the training tier's (T2 is the main validation panel);
    with --tier all the mean of the three tiers' (val_all_success)."""
    return {"T1": "val_t1_success", "T2": "val_success", "T3": "val_t3_success", "all": "val_all_success"}[tier]


def all_tiers_success(report: dict) -> float:
    """Mean validation success over T1, T2 and T3 (--tier all)."""
    return float(np.mean([report["val_t1_success"], report["val_success"], report["val_t3_success"]]))


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
    p.add_argument("--controller", default="level", choices=("level", "smooth", "aeb", "guard", "yield", "yield_pd", "hold"),
                   help="60 km/h teacher only: lane-relative commands (d4 on), with throttle smoothing, with the "
                        "emergency-braking layer (d10), plus the lane-change guard and speed-scaled route steering "
                        "(guard, d11), plus the predictive layers for traffic from the side and the low-speed lateral "
                        "cap (yield, d13; LOG.md, 2026-11-05), the same without the PID integral (yield_pd: the "
                        "end-to-end teacher from d16, LOG.md 2026-11-06); or the hold_lane controller, which made the teacher itself worse (96.4%% vs "
                        "98.0%% on T1; LOG.md, 2026-11-03)")
    p.add_argument("--lane-change-weight", type=float, default=LANE_CHANGE_WEIGHT,
                   help="loss weight of lane-change samples (d2-d6: 10; d7: 3)")
    p.add_argument("--tier", default=TIER, choices=("T1", "T2", "T3", "all"),
                   help="training scenes; with T1 the best round is chosen on 200 T1 validation scenes (from d5), "
                        "with T3 on 200 T3 validation scenes (from d12: a driver for every tier), with all (each "
                        "episode's tier drawn from T1/T2/T3, from d13) on the mean of the three tiers' validation")
    p.add_argument("--action", default="tactical", choices=("tactical", "e2e"),
                   help="tactical: the model chooses lane and throttle and --controller drives (d4-d13); e2e: "
                        "avtest.policy.EndToEndModel outputs steering and throttle itself, imitating the teacher's "
                        "final action through --controller, and drives with no controller (from d14)")
    p.add_argument("--dart", type=float, default=DART_SIGMA,
                   help="e2e only: std of the steering noise on half the teacher's episodes (0: none)")
    p.add_argument("--hidden", default=",".join(map(str, E2E_HIDDEN)), help="e2e only: hidden layer sizes")
    p.add_argument("--lane-model", action=argparse.BooleanOptionalAction, default=True,
                   help="e2e only: add the lane model (avtest.sim.lane_model) to the observation; the lane-line beams "
                        "flicker between dashes and gaps (LOG.md, 2026-11-05); the d14 pilot ran without it")
    p.add_argument("--object-list", action=argparse.BooleanOptionalAction, default=False,
                   help="e2e with --lane-model only: add the object list (avtest.sim.object_list: the 10 nearest "
                        "objects with heading, tracked velocity and their predicted path; from d18)")
    p.add_argument("--brake-weight", type=float, default=1.0,
                   help="e2e only: weight of the hard-braking throttle levels in the cross-entropy (d16: 1; its "
                        "network caught 55%% of the teacher's hard brakes, LOG.md 2026-11-06)")
    p.add_argument("--steer-net", action=argparse.BooleanOptionalAction, default=True,
                   help="e2e only: steering from its own network on the ego state and lane model "
                        "(avtest.policy.steering_features); the d14 pilot steered from the shared trunk")
    args = p.parse_args()
    e2e = args.action == "e2e"
    if e2e and args.head != "classify":
        raise SystemExit("--action e2e has its own heads; leave --head at classify")
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
                      "aeb": tactical.AEB_CONTROLLER, "guard": tactical.GUARD_CONTROLLER,
                      "yield": tactical.YIELD_CONTROLLER, "yield_pd": tactical.YIELD_PD_CONTROLLER,
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
    hidden = tuple(int(h) for h in args.hidden.split(","))
    if e2e:                                     # the lane-change weight and heads of the tactical student do not apply
        settings = {k: v for k, v in settings.items() if k not in ("lane_change_weight", "head", "throttle_head")}
        settings.update({"action": "e2e", "student_controller": tactical.END_TO_END, "dart": args.dart,
                         "dart_share": DART_SHARE, "dart_rho": DART_RHO, "hidden": list(hidden),
                         "history_steps": HISTORY_STEPS, "heldout_every": HELDOUT_EVERY})
        if args.lane_model or args.steer_net:          # absent from the pilot's settings, which had neither
            settings.update({"lane_model": args.lane_model, "steer_net": args.steer_net})
        if args.brake_weight != 1.0:                   # absent from d16's settings, which had none
            settings["brake_weight"] = args.brake_weight
        if args.object_list:                           # absent from d16's and d17's settings
            if not args.lane_model:
                raise SystemExit("--object-list needs --lane-model")
            settings["object_list"] = True
    if (out / "bc_config.json").exists() and not same_run(read_json(out / "bc_config.json"), settings):
        raise SystemExit(f"{out} was started with other settings; use a new --name")
    write_json(out / "bc_config.json", settings)
    log = JsonlLog(out / "bc_log.jsonl")
    done_rounds = [r["round"] for r in log.read()]

    class Spaces(gym.Env):                              # spaces only, to build the PPO policy offline
        observation_space = gym.spaces.Box(-0.0, 1.0, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    from avtest.policy import THROTTLE_BINS, EndToEndModel, TacticalModel, fit_classify, fit_end_to_end, load_policy
    if done_rounds:
        model = load_policy(out / "model_last.zip")
        print(f"resuming after round {done_rounds[-1]}", flush=True)
    elif e2e:
        from avtest.policy import steering_features
        from avtest.sim import E2E_OBJ_OBS_DIM, E2E_OBS_DIM
        obs_dim = (E2E_OBJ_OBS_DIM if args.object_list else E2E_OBS_DIM) if args.lane_model else OBS_DIM
        model = EndToEndModel(obs_dim, hidden, seed=DEFAULTS["seed"],
                              steer_features=steering_features(obs_dim) if args.steer_net else None)
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
                  "tier": args.tier, "controller": controller, "action": args.action, "dart": args.dart,
                  "lane_model": e2e and args.lane_model, "object_list": e2e and args.object_list}
                 for w in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_collect, tasks)
        shard = out / "data" / f"round_{rnd:02d}.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez(shard, obs=np.concatenate([r["obs"] for r in results]), act=np.concatenate([r["act"] for r in results]),
                 first=np.concatenate([r["first"] for r in results]))
        collect_s = time.time() - t0
        if e2e:
            obs, act, first, heldout = load_rounds(out / "data", rnd)
            if not done_rounds and rnd == 0:            # a new model: input scaling from its first data, then fixed
                train0 = np.flatnonzero(~heldout)
                sample = np.random.default_rng(0).choice(train0, min(len(train0), 200_000), replace=False)
                model.set_normalization(stored_features(obs, first, np.sort(sample)), act[train0, 1], act[train0, 0])
            n_samples = int(len(act))
            fit_stats = fit_end_to_end(model, obs, act, first, np.flatnonzero(~heldout), np.flatnonzero(heldout),
                                       args.epochs, rnd, brake_weight=args.brake_weight)
            del first, heldout
        else:
            shards = [np.load(out / "data" / f"round_{k:02d}.npz") for k in range(rnd + 1)]
            obs = np.concatenate([s["obs"] for s in shards])
            act = np.concatenate([s["act"] for s in shards])
            n_samples = int(len(act))
            del shards
            if args.head == "classify":
                fit_stats = fit_classify(model, obs, act, args.epochs, rnd)
            else:
                fit_stats = fit(model, obs, act, args.epochs, rnd, args.lane_change_weight)
                with torch.no_grad():                 # PPO later starts with small exploration noise (σ ≈ 0.22)
                    model.policy.log_std.fill_(INITIAL_LOG_STD)
        model.save(out / "model_last.zip")
        (out / "checkpoints").mkdir(exist_ok=True)
        model.save(out / "checkpoints" / f"bc_round_{rnd:02d}.zip")
        for saved in (out / "model_last.zip", out / "checkpoints" / f"bc_round_{rnd:02d}.zip"):
            # the driver drives with what it trained with: its controller, or none (END_TO_END)
            tactical.stamp_controller(saved, tactical.END_TO_END if e2e else controller)
        del obs, act
        outcomes = [o for r in results for o in r["outcomes"]]
        rollout = [o for t, o in outcomes if not t]
        report = progress.checkpoint_report(out / "model_last.zip", out, f"bc_round_{rnd:02d}", workers=min(4, args.workers),
                                            panel=panel, tier=progress.TIER,
                                            t1_panel=panel if args.tier in ("T1", "T3", "all") else min(panel, 100),
                                            t3_panel=panel if args.tier in ("T3", "all") else 0)
        if args.tier == "all":
            report["val_all_success"] = all_tiers_success(report)
        row = {"round": rnd, "samples": n_samples, **fit_stats, "collect_s": collect_s, "wall_s": time.time() - t0,
               "teacher_episodes": sum(t for t, _ in outcomes),
               "teacher_failures_dropped": sum(t and o != "success" for t, o in outcomes),
               "student_rollouts": len(rollout),
               "student_rollout_success": (sum(o == "success" for o in rollout) / len(rollout)) if rollout else None,
               **report}
        log.append(row)
        progress.chart(log.read(), "round", "DAgger round", out / "bc_progress.png", f"{out.name}: imitation warm start")
        error = (f"steering error {fit_stats['steer_mae']:.4f} (labels' spread {fit_stats['steer_label_std']:.4f})"
                 if e2e else f"action error {fit_stats['action_mae']:.3f}")
        print(f"round {rnd}: {row['samples']:,} samples | {error} | "
              f"{progress.headline(report)} | {row['wall_s'] / 60:.1f} min", flush=True)
    # DAgger rolls out the latest student each round; PPO starts from the round that validated best
    rows = log.read()
    select = selection_key(args.tier)
    best = max(rows, key=lambda r: (r[select], r["round"]))
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_last.zip")
    shutil.copy2(out / "checkpoints" / f"bc_round_{best['round']:02d}.zip", out / "model_best_bc.zip")
    write_json(out / "bc_done.json", {"rounds": args.rounds, "best_round": best["round"],
                                      "best_val_success": best["val_success"], "selected_on": select,
                                      "selected_value": best[select], "final": rows[-1]})
    print(f"best round {best['round']} ({select} {progress.pct(best[select])}) saved as "
          f"{out / 'model_best_bc.zip'}; recordings in {out / 'recordings'}"
          + (f"; PPO can continue with: python -m scripts.train_driver --name {args.name}" if args.head == "mean" else ""),
          flush=True)


if __name__ == "__main__":
    main()
