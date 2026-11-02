"""How far before an obstacle in its own lane does each driver get out of the way? Failed-qualification panels only.

    .venv\\Scripts\\python.exe -m scripts.diagnose_obstacles --model runs\\driver\\d2\\model_best.zip --attempt 1 --tier T1

Runs every obstacle scene of a failed qualification attempt (development data after the failure,
docs/DESIGN.md) whose obstacle stands in the ego's lane, with the SUT, IDM and the cautious-IDM
imitation teacher. Measures, without reference to any driver's internals:

- ``left_at_m``: distance to the stopped vehicle, along the route, at the last step the ego's
  centre was within 1 m of the obstacle lane's centre line, i.e. where it finally moved out
  (the warning triangle stands 8 m before the vehicle);
- whether the obstacle stands on a curve;
- for the SUT, ``aborts``: how often its lane command fell back inside ±0.5 while it was still
  in the obstacle's lane and already more than 0.3 m off its centre, which steers it back.

Writes runs/diagnose/attempt<N>_<tier>_obstacles.jsonl and prints the summary.
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
    from metadrive.component.lane.circular_lane import CircularLane
    from avtest import scenario as sc, seeds
    from avtest.drivers import CautiousIDMDriver, IDMDriver
    env = _worker["env"]
    scene = sc.decode(np.asarray(task["theta"]), seeds.check(task["map_seed"], "qual"), task["tier"])
    out = {"row": task["row"], "map": scene.values["map"], "lanes": scene.values["lane_num"]}
    for name, driver in (("sut", _worker["sut"]), ("idm", IDMDriver()), ("idm_cautious", CautiousIDMDriver())):
        env.load(scene)
        obs, info = env.reset()
        driver.start(env)
        em = env.engine.managers["event_manager"]
        out["curve"] = isinstance(em.path.lanes[int(np.searchsorted(em.path.starts, em.s, side="right") - 1)],
                                  CircularLane)
        left_at, aborts, commanded = None, 0, False
        while True:
            ego = env.agent
            s = em.path.project(ego.position)
            ahead = em.s - s
            offset = float(np.linalg.norm(np.asarray(ego.position) - em.path.position(s)))
            if name == "sut":
                tactical, _ = driver.model.predict(obs, deterministic=True)
                in_obstacle_lane = em.npc is not None and ego.lane_index[2] == em.npc.lane_index[2]
                if commanded and abs(float(tactical[0])) <= 0.5 and ahead > 0 and in_obstacle_lane and offset > 0.3:
                    aborts += 1
                commanded = abs(float(tactical[0])) > 0.5
                action = driver.controller.low_level(ego, tactical)
            else:
                action = driver.act(env, obs)
            if em.npc is not None and 0 < ahead < 200 and offset < 1.0:
                left_at = float(ahead)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        out[name] = {"outcome": info["outcome"], "left_at_m": left_at, "aborts": aborts if name == "sut" else None,
                     "crash": [k for k in ("crash_object", "crash_vehicle") if info.get(k)]}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--tier", default="T1")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    folder = ROOT / "runs/qualification" / f"attempt{args.attempt}_{args.tier}"
    result = json.loads((folder / "result.json").read_text())
    if result["passed"]:
        raise SystemExit(f"{folder} passed: its panel is not development data")
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    tasks = [{"row": i, "theta": r["theta"], "map_seed": r["map_seed"], "tier": args.tier} for i, r in enumerate(rows)
             if r["values"]["event"] == "obstacle" and r["values"]["event_lane"] == "same"]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        results = pool.map(run, tasks, chunksize=1)
    out = ROOT / "runs/diagnose" / f"attempt{args.attempt}_{args.tier}_obstacles.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"{len(results)} scenes with the obstacle in the ego's lane -> {out}")
    for label, sub in (("straight", [r for r in results if not r["curve"]]), ("curve", [r for r in results if r["curve"]]),
                       ("all", results)):
        print(f"{label} ({len(sub)} scenes)")
        for name in ("sut", "idm", "idm_cautious"):
            left = [r[name]["left_at_m"] for r in sub if r[name]["left_at_m"] is not None and r[name]["outcome"] != "stuck"]
            ok = sum(r[name]["outcome"] == "success" for r in sub)
            spread = f"median {np.median(left):.1f} m, 10th percentile {np.percentile(left, 10):.1f} m" if left else "n/a"
            print(f"  {name:12s} success {ok}/{len(sub)}; leaves the obstacle lane at {spread}")
        print(f"  SUT scenes with an abort: {sum(r['sut']['aborts'] > 0 for r in sub)}")
    failed = [r for r in results if r["sut"]["outcome"] != "success"]
    print("SUT failures:", [(r["row"], r["map"], r["sut"]["outcome"], r["sut"]["crash"], round(r["sut"]["left_at_m"] or -1, 1))
                            for r in failed])


if __name__ == "__main__":
    main()
