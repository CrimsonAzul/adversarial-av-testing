"""Warm start for the driver: imitation of the cautious-IDM teacher with DAgger, before PPO.

    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d1                 # ~1–1.5 h
    .venv\\Scripts\\python.exe -m scripts.pretrain_bc --name d1 --check         # 2-minute pipeline check

The student sees only the lidar-and-state observation, the same one PPO uses. The teacher
(MetaDrive IDM with cautious settings, avtest.drivers.CautiousIDMDriver) reads privileged
map state and labels every visited state with the action it would take.
Round 0 lets the teacher drive. In rounds 1..R the student drives, with the teacher taking
over with probability β = 0.5^round each episode, while the teacher still labels every state
(DAgger; Ross, Gordon & Bagnell 2011). The policy is the PPO actor itself, trained by
maximum likelihood of the teacher's actions on all data so far, so scripts.train_driver
continues with PPO from runs/driver/<name>/model_last.zip.

Disclosure: the teacher is also one of the three reference drivers. Avoidability needs any
one reference to complete a scene, so this cannot make a failure look valid; it can only
make the SUT share some of the cautious IDM's blind spots (reported in the thesis).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import time

import torch
import numpy as np

from avtest.records import JsonlLog, ROOT, write_json

TIER = "T3"


def _collect(task: dict) -> dict:
    import torch as th
    th.set_num_threads(1)
    from stable_baselines3 import PPO
    from avtest import scenario as sc, seeds
    from avtest.drivers import CautiousIDMDriver
    from avtest.sim import AVTestEnv
    rng = np.random.default_rng(task["seed"])
    env = AVTestEnv()
    teacher = CautiousIDMDriver()
    student = PPO.load(task["model"], device="cpu") if task["model"] else None
    obs_list, act_list, outcomes, steps = [], [], [], 0
    for _ in range(task["episodes"]):
        env.load(sc.decode(sc.sample(rng, 1)[0], seeds.draw(rng, "train"), TIER))
        obs, _ = env.reset()
        teacher.start(env)
        teacher_drives = student is None or rng.random() < task["beta"]
        while True:
            label = np.asarray(teacher.act(env, obs), dtype=np.float32)
            obs_list.append(obs.astype(np.float32))
            act_list.append(np.clip(label, -1, 1))
            action = label if teacher_drives else student.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc:
                outcomes.append((bool(teacher_drives), info["outcome"]))
                break
    env.close()
    return {"obs": np.asarray(obs_list), "act": np.asarray(act_list), "outcomes": outcomes, "steps": steps}


def fit(model, obs: np.ndarray, act: np.ndarray, epochs: int, seed: int) -> float:
    """Maximum likelihood of teacher actions under the PPO actor (value head untouched)."""
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters()) \
        + [policy.log_std] + list(policy.features_extractor.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    gen = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(obs), torch.as_tensor(act)
    loss_value = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 512):
            idx = order[i:i + 512]
            _, log_prob, _ = policy.evaluate_actions(X[idx], Y[idx])
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
    return loss_value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=600, help="episodes per round")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    from stable_baselines3 import PPO
    from scripts.train_driver import DEFAULTS
    import gymnasium as gym

    out = ROOT / "runs/driver" / (args.name + ("_bccheck" if args.check else ""))
    if args.check:
        args.rounds, args.episodes, args.workers, args.epochs = 1, 4, 2, 1
    if (out / "train_log.jsonl").exists():
        raise SystemExit(f"{out} already has PPO training; pretrain into a new --name")
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "bc_log.jsonl")

    class Spaces(gym.Env):                              # spaces only, to build the PPO policy offline
        observation_space = gym.spaces.Box(-0.0, 1.0, (259,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    model = PPO("MlpPolicy", Spaces(), n_steps=DEFAULTS["n_steps"], batch_size=DEFAULTS["batch_size"],
                n_epochs=DEFAULTS["n_epochs"], learning_rate=DEFAULTS["learning_rate"], gamma=DEFAULTS["gamma"],
                gae_lambda=DEFAULTS["gae_lambda"], clip_range=DEFAULTS["clip_range"], ent_coef=DEFAULTS["ent_coef"],
                seed=DEFAULTS["seed"], device="cpu", policy_kwargs={"net_arch": {"pi": DEFAULTS["net"], "vf": DEFAULTS["net"]}})
    data_obs, data_act = [], []
    ctx = mp.get_context("spawn")
    for rnd in range(args.rounds + 1):
        t0 = time.time()
        current = None if rnd == 0 else str(out / "model_last.zip")
        per = max(1, args.episodes // args.workers)
        tasks = [{"seed": 10_000 * rnd + w, "episodes": per, "model": current, "beta": 0.5 ** rnd}
                 for w in range(args.workers)]
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_collect, tasks)
        for r in results:
            data_obs.append(r["obs"])
            data_act.append(r["act"])
        obs, act = np.concatenate(data_obs), np.concatenate(data_act)
        loss = fit(model, obs, act, args.epochs, rnd)
        with torch.no_grad():                         # start PPO with modest exploration noise
            model.policy.log_std.fill_(-1.0)
        model.save(out / "model_last.zip")
        outcomes = [o for r in results for o in r["outcomes"]]
        student = [o for t, o in outcomes if not t]
        log.append({"round": rnd, "samples": int(len(obs)), "nll": loss, "wall_s": time.time() - t0,
                    "teacher_episodes": sum(t for t, _ in outcomes),
                    "student_episodes": len(student),
                    "student_success": (sum(o == "success" for o in student) / len(student)) if student else None})
        print(f"round {rnd}: {len(obs):,} samples, NLL {loss:.3f}, student success "
              f"{log.read()[-1]['student_success']}, {time.time() - t0:.0f}s", flush=True)
    write_json(out / "bc_done.json", {"rounds": args.rounds, "samples": int(len(obs))})
    print(f"warm start saved to {out / 'model_last.zip'}; continue with: python -m scripts.train_driver --name {args.name}")


if __name__ == "__main__":
    main()
