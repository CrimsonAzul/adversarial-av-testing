"""All RQ1–RQ3 search runs against the qualified SUT, in preregistered order, resumable.

    .venv\\Scripts\\python.exe -m scripts.run_search --hours 8
    .venv\\Scripts\\python.exe -m scripts.run_search --check          # 2 tiny runs on the check model

Order: seed 101 for rl, cmaes, bo, random; then 202; then 303. Interleaving by seed means
an interrupted campaign always has complete matched seeds. Finished runs are skipped,
unfinished ones continue. Runs refuse to start without models/QUALIFIED.json (the gate).
"""
from __future__ import annotations

import argparse
import time

import torch  # noqa: F401  (before MetaDrive)

from avtest import search
from avtest.adversaries import METHODS
from avtest.drivers import qualified_sut
from avtest.records import ROOT

SEEDS = (101, 202, 303)
BUDGET = 1000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    from avtest.evaluate import Evaluator

    if args.check:
        candidates = sorted((ROOT / "runs/driver").glob("*check/model_best.zip"), key=lambda c: c.stat().st_mtime)
        if not candidates:
            raise SystemExit("run scripts.train_driver --name <name> --check first")
        sut = candidates[-1]                           # the most recent check model
        tier, budget, seeds_, methods = "T2", 28, (101,), METHODS
        root = ROOT / "runs/search_check"
    else:
        sut, q = qualified_sut(ROOT)                 # model and controller exactly as qualified
        tier, budget, seeds_, methods = q["tier"], BUDGET, SEEDS, METHODS
        root = ROOT / "runs/search"
    deadline = time.time() + args.hours * 3600
    plan = [(m, s) for s in seeds_ for m in methods]
    with Evaluator(sut, tier, workers=args.workers) as evaluator:
        for method, seed in plan:
            out = root / f"{method}_s{seed}"
            if (out / "done.json").exists():
                continue
            print(f"\n== {method} seed {seed} (tier {tier}, budget {budget})", flush=True)
            try:
                search.run(method, seed, tier, budget, out, evaluator, sut, batch=args.workers, deadline=deadline)
            except search.Paused as reason:
                print(f"paused: {reason}. Run the same command to continue.", flush=True)
                return
    print("all search runs finished. Next: python -m scripts.rq4, then python -m scripts.report", flush=True)


if __name__ == "__main__":
    main()
