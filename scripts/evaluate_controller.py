"""Compare tactical-controller settings for a fixed policy on a failed qualification panel (development data).

    .venv\\Scripts\\python.exe -m scripts.evaluate_controller --model runs\\driver\\d2\\model_best.zip --attempt 1 --tier T1 --settings 0.5:off,0.5:on,0.35:on,0.25:on

Each setting is threshold:commit (avtest.tactical.TacticalController) for the SUT, or teacher:<name>
for an imitation teacher driving through its own tactical labels (avtest.tactical.TEACHERS). Every
panel scene is driven with each setting; success is counted, as in the qualification gate, over the scenes at least
one reference driver completed in that attempt (their reference runs are reused, not repeated). Also
reported: collision/off-road rate, success per event, and lane changes per episode, which would rise
if a lower threshold made the driver weave. The panel was failed, so tuning on it is allowed
(docs/DESIGN.md); a chosen setting is then qualified on a new panel.

Writes runs/diagnose/attempt<N>_<tier>_controller.jsonl as results arrive; the same command continues an
interrupted comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import stats
from avtest.records import JsonlLog, ROOT, sha256_file

_worker: dict = {}


def _init(model: str):
    import torch  # noqa: F401
    torch.set_num_threads(1)
    from avtest.drivers import SUTDriver
    from avtest.sim import AVTestEnv
    _worker["env"], _worker["sut"] = AVTestEnv(), SUTDriver(Path(model))


def run(task: dict) -> dict:
    from avtest import scenario as sc, seeds
    from avtest.tactical import TacticalController, TeacherDriver
    env = _worker["env"]
    if task["teacher"]:
        driver = TeacherDriver(task["teacher"])
    else:
        driver = _worker["sut"]
        driver.controller = TacticalController(threshold=task["threshold"], commit=task["commit"])
    env.load(sc.decode(np.asarray(task["theta"]), seeds.check(task["map_seed"], "qual"), task["tier"]))
    obs, info = env.reset()
    driver.start(env)
    lane, changes = env.agent.lane_index, 0
    while True:
        obs, _, terminated, truncated, info = env.step(driver.act(env, obs))
        now = env.agent.lane_index
        changes += now[:2] == lane[:2] and now[2] != lane[2]           # same road, different lane
        lane = now
        if terminated or truncated:
            break
    return {"setting": task["setting"], "row": task["row"], "outcome": info["outcome"], "lane_changes": changes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--tier", default="T1")
    p.add_argument("--settings", default="0.5:off,0.5:on,0.35:on,0.25:on")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    folder = ROOT / "runs/qualification" / f"attempt{args.attempt}_{args.tier}"
    if json.loads((folder / "result.json").read_text())["passed"]:
        raise SystemExit(f"{folder} passed: its panel is not development data")
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    solvable = {i for i, r in enumerate(rows) if any(ref["outcome"] == "success" for ref in r["references"])}
    settings = []
    for item in args.settings.split(","):
        first, second = item.split(":")
        settings.append((item, None, False, second) if first == "teacher" else (item, float(first), second == "on", None))
    # results are appended as they arrive, so an interrupted comparison continues where it stopped
    out = ROOT / "runs/diagnose" / f"attempt{args.attempt}_{args.tier}_controller.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    model_hash = sha256_file(args.model)
    log = JsonlLog(out)
    previous = log.read()
    if any(r["model_sha256"] != model_hash for r in previous):
        raise SystemExit(f"{out} holds results of another model; move it away first")
    done = {(r["setting"], r["row"]) for r in previous}
    tasks = [{"setting": name, "threshold": threshold, "commit": commit, "teacher": teacher, "row": i,
              "theta": r["theta"], "map_seed": r["map_seed"], "tier": args.tier}
             for name, threshold, commit, teacher in settings for i, r in enumerate(rows) if (name, i) not in done]
    print(f"{len(rows)} scenes, {len(solvable)} solvable by a reference (attempt {args.attempt}); "
          f"{len(tasks)} episodes to run, {len(done)} already in {out}", flush=True)
    if tasks:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
            for k, result in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
                log.append({**result, "model_sha256": model_hash})
                if k % 100 == 0:
                    print(f"  {k}/{len(tasks)} episodes", flush=True)
    results = log.read()
    for name, *_ in settings:
        mine = [r for r in results if r["setting"] == name and r["row"] in solvable]
        ok = sum(r["outcome"] == "success" for r in mine)
        crashes = sum(r["outcome"] in ("collision", "offroad") for r in mine)
        low, _ = stats.wilson(ok, len(mine))
        events = Counter(rows[r["row"]]["values"]["event"] for r in mine)
        by_event = {e: f"{sum(r['outcome'] == 'success' for r in mine if rows[r['row']]['values']['event'] == e)}/{n}"
                    for e, n in sorted(events.items())}
        print(f"{name:9s} success {ok}/{len(mine)} = {ok / len(mine):.1%}  Wilson low {low:.3f}  "
              f"collision/off-road {crashes / len(mine):.1%}  {dict(Counter(r['outcome'] for r in mine))}  "
              f"by event {by_event}  lane changes/episode {np.mean([r['lane_changes'] for r in mine]):.2f}")


if __name__ == "__main__":
    main()
