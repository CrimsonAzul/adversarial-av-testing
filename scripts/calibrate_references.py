"""How often do the reference drivers solve uniformly random scenes? (no SUT involved)

    .venv\\Scripts\\python.exe -m scripts.calibrate_references --tier T2 --scenes 60

Uses validation-split seeds. Reports per-reference and union success, by event type,
and the timing per scene. Writes runs/calibration/<tier>/episodes.jsonl and summary.json.
"""
from __future__ import annotations

import argparse
import collections
import time

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario, seeds
from avtest.records import JsonlLog, ROOT, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tier", default="T2")
    p.add_argument("--scenes", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--panel-seed", type=int, default=31)
    args = p.parse_args()
    from avtest.evaluate import Evaluator
    out = ROOT / "runs/calibration" / args.tier
    log = JsonlLog(out / "episodes.jsonl")
    done = log.read()
    map_seeds = seeds.panel_seeds("val", args.scenes, args.panel_seed)
    thetas = scenario.sample(np.random.default_rng(args.panel_seed), args.scenes)
    t0 = time.time()
    if len(done) < args.scenes:
        with Evaluator(None, args.tier, workers=args.workers, references=True, all_references=True) as ev:
            for start in range(len(done), args.scenes, 12):
                log.extend(ev.evaluate(thetas[start:start + 12], map_seeds[start:start + 12], split="val"))
                print(f"{min(start + 12, args.scenes)}/{args.scenes}", flush=True)
    rows = log.read()
    names = [r["driver"] for r in rows[0]["references"]]
    by_event = collections.defaultdict(list)
    for r in rows:
        by_event[r["values"]["event"]].append(r["avoidable"])
    summary = {"tier": args.tier, "scenes": len(rows),
               "union_success": float(np.mean([r["avoidable"] for r in rows])),
               "per_reference": {n: float(np.mean([r["references"][i]["outcome"] == "success" for r in rows]))
                                 for i, n in enumerate(names)},
               "reference_outcomes": {n: dict(collections.Counter(r["references"][i]["outcome"] for r in rows))
                                      for i, n in enumerate(names)},
               "union_by_event": {k: [int(sum(v)), len(v)] for k, v in by_event.items()},
               "mean_scene_wall_s": float(np.mean([r["wall_s"] for r in rows])),
               "batch_wall_s": time.time() - t0}
    write_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
