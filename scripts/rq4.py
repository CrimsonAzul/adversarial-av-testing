"""RQ4: the RL generator, frozen at the end of each search run, against random on held-out map seeds.

    .venv\\Scripts\\python.exe -m scripts.rq4

For each seed: 300 scenarios proposed by the frozen policy of runs/search/rl_s<seed>
(weights and history features fixed) and 300 uniform random scenarios, both on the same
held-out map-seed list (split "heldout", never used before). Logs go to
runs/rq4/{rl_frozen,random}_s<seed>/ in the same format as search runs, so
scripts.report reads them directly.
"""
from __future__ import annotations

import argparse

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario
from avtest.drivers import qualified_sut
from avtest.records import JsonlLog, ROOT, write_json
from avtest.search import map_seed_list

from scripts.run_search import SEEDS  # noqa: E402  (one RQ4 comparison per search seed)
BUDGET = 300


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    from avtest.adversaries.rl import RLAdversary
    from avtest.evaluate import Evaluator
    sut, q = qualified_sut(ROOT)
    tier = q["tier"]
    with Evaluator(sut, tier, workers=args.workers) as evaluator:
        for seed in SEEDS:
            source = ROOT / f"runs/search/rl_s{seed}"
            if not (source / "done.json").exists():
                print(f"skip seed {seed}: {source} not finished")
                continue
            generator = RLAdversary(scenario.DIM, seed, workers=args.workers, cell=lambda t: scenario.cell(t, tier))
            generator.load(source / "rl", lambda t: [], 1)
            proposals = {"rl_frozen": generator.propose_frozen(BUDGET, rng_seed=seed + 4),
                         "random": np.random.default_rng([seed, 4]).random((BUDGET, scenario.DIM))}
            map_seeds = map_seed_list(seed, BUDGET, split="heldout")
            for arm, thetas in proposals.items():
                out = ROOT / f"runs/rq4/{arm}_s{seed}"
                write_json(out / "config.json", {"method": arm, "seed": seed, "tier": tier, "budget": BUDGET,
                                                 "split": "heldout", "sut_sha256": q["model_sha256"]})
                log = JsonlLog(out / "evaluations.jsonl")
                rows = log.read()
                for start in range(len(rows), BUDGET, args.workers):
                    batch = thetas[start:start + args.workers]
                    new = evaluator.evaluate(batch, map_seeds[start:start + len(batch)], split="heldout")
                    for i, r in enumerate(new):
                        r.update(index=start + i, method=arm, run_seed=seed, cell=list(scenario.cell(batch[i], tier)))
                    log.extend(new)
                write_json(out / "done.json", {"evaluations": BUDGET})
                print(f"rq4 {arm} seed {seed}: done", flush=True)


if __name__ == "__main__":
    main()
