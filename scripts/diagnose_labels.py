"""What would each imitation teacher tell the student? DAgger labels along the student's own drive, own-lane obstacle scenes.

    .venv\\Scripts\\python.exe -m scripts.diagnose_labels --model runs\\driver\\d2\\model_best.zip --attempt 1 --tier T1

The SUT drives every obstacle scene of a failed qualification attempt (development data) whose obstacle
stands in its lane. At every step each teacher in avtest.tactical.TEACHERS labels the state, as
scripts.pretrain_bc does for student-driven episodes. Per scene and teacher:

- ``first_change_m``: distance to the stopped vehicle (along the route) where the first sustained
  "change lane" label (at least 3 steps) starts, with the obstacle at most 50 m ahead and the ego in its lane;
- ``flips``: how often the label then went back to "keep" before the ego left its lane (an inconsistent
  label the student cannot satisfy);
- ``blips``: "change lane" labels lasting under 3 steps anywhere in the episode (label noise).

The SUT's own driving is not affected by the labels. Writes runs/diagnose/attempt<N>_<tier>_labels.jsonl.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest.records import ROOT

_worker: dict = {}


def _init(model: str):
    import torch  # noqa: F401
    torch.set_num_threads(1)
    from avtest.drivers import SUTDriver
    from avtest.sim import AVTestEnv
    _worker["env"], _worker["sut"] = AVTestEnv(), SUTDriver(Path(model))


def run(task: dict) -> dict:
    from avtest import scenario as sc, seeds, tactical
    env, sut = _worker["env"], _worker["sut"]
    scene = sc.decode(np.asarray(task["theta"]), seeds.check(task["map_seed"], "qual"), task["tier"])
    env.load(scene)
    obs, _ = env.reset()
    sut.start(env)
    em = env.engine.managers["event_manager"]
    teachers = {name: cls(env.agent, scene.map_seed) for name, cls in tactical.TEACHERS.items()}
    first = {name: None for name in teachers}
    flips, blips = {name: 0 for name in teachers}, {name: 0 for name in teachers}
    run = {name: 0 for name in teachers}                  # length of the current "change lane" run
    run_start = {name: None for name in teachers}         # (ahead, in its lane and in the window) where it began
    start_lane = env.agent.lane_index[2]
    while True:
        ahead = em.s - em.ego_path.project(env.agent.position) if em.npc is not None else None
        in_window = ahead is not None and 0 < ahead <= 50 and env.agent.lane_index[2] == start_lane
        for name, teacher in teachers.items():
            teacher.act(env.agent.id)
            label = tactical.teacher_command(teacher, env.agent)
            if label != 0:
                if run[name] == 0:
                    run_start[name] = (ahead, in_window)
                run[name] += 1
                if run[name] == 3 and first[name] is None and run_start[name][1]:
                    first[name] = float(run_start[name][0])
            else:
                if 0 < run[name] < 3:
                    blips[name] += 1
                elif run[name] >= 3 and first[name] is not None and in_window:
                    flips[name] += 1
                run[name] = 0
        obs, _, terminated, truncated, info = env.step(sut.act(env, obs))
        if terminated or truncated:
            break
    return {"row": task["row"], "map": scene.values["map"], "sut_outcome": info["outcome"],
            **{f"{name}_first_change_m": first[name] for name in teachers},
            **{f"{name}_flips": flips[name] for name in teachers}, **{f"{name}_blips": blips[name] for name in teachers}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--tier", default="T1")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    from avtest.tactical import TEACHERS
    folder = ROOT / "runs/qualification" / f"attempt{args.attempt}_{args.tier}"
    if json.loads((folder / "result.json").read_text())["passed"]:
        raise SystemExit(f"{folder} passed: its panel is not development data")
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    tasks = [{"row": i, "theta": r["theta"], "map_seed": r["map_seed"], "tier": args.tier} for i, r in enumerate(rows)
             if r["values"]["event"] == "obstacle" and r["values"]["event_lane"] == "same"]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        results = pool.map(run, tasks, chunksize=1)
    out = ROOT / "runs/diagnose" / f"attempt{args.attempt}_{args.tier}_labels.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"{len(results)} scenes, the SUT driving -> {out}")
    for name in TEACHERS:
        firsts = [r[f"{name}_first_change_m"] for r in results if r[f"{name}_first_change_m"] is not None]
        never = len(results) - len(firsts)
        spread = f"median {np.median(firsts):.1f} m, 10th-90th pct {np.percentile(firsts, 10):.1f}-{np.percentile(firsts, 90):.1f} m" if firsts else "n/a"
        print(f"  {name:14s} first sustained 'change lane' label at {spread}; none within 50 m: {never}; "
              f"scenes with a flip back to 'keep': {sum(r[f'{name}_flips'] > 0 for r in results)}; "
              f"one-step blips per scene: {np.mean([r[f'{name}_blips'] for r in results]):.1f}")


if __name__ == "__main__":
    main()
