"""One adversarial search run: method × seed × budget, resumable, every evaluation logged.

Layout of a run folder:
    config.json        method, seed, tier, budget, SUT hash; refuses to resume with other settings
    evaluations.jsonl  one row per evaluated scenario, in evaluation order (the budget ledger)
    state.pkl          adversary state after the last completed batch
    rl/                PPO generator checkpoint (RL adversary only)
    PAUSE / STOP       control files: stop after the current batch / stop now

Map seeds come from a list fixed by the run seed, so runs of different methods with
the same seed see the same seed sequence (matched nuisance), and a resumed run uses
exactly the seeds an uninterrupted one would.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import time

import numpy as np

from . import scenario, seeds
from .adversaries import make
from .records import JsonlLog, control_state, read_json, sha256_file, write_json


def map_seed_list(run_seed: int, budget: int, split: str = "search") -> list[int]:
    rng = np.random.default_rng([run_seed, 99])
    return [seeds.draw(rng, split) for _ in range(budget)]


class Paused(Exception):
    pass


def run(method: str, seed: int, tier: str, budget: int, out: Path, evaluator, sut_path: Path,
        batch: int = 4, deadline: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "seed": seed, "tier": tier, "budget": budget, "batch": batch,
              "sut_sha256": sha256_file(sut_path), "dim": scenario.DIM}
    if (out / "config.json").exists():
        if read_json(out / "config.json") != config:
            raise SystemExit(f"{out} was started with different settings; use a new folder")
    else:
        write_json(out / "config.json", config)
    log = JsonlLog(out / "evaluations.jsonl")
    rows = log.read()
    map_seeds = map_seed_list(seed, budget)
    cell = lambda theta: scenario.cell(theta, tier)

    def should_stop() -> str | None:
        return control_state(out) or ("session hours used" if deadline and time.time() > deadline else None)

    def evaluate(thetas: np.ndarray) -> list[dict]:
        start = len(rows)
        if start + len(thetas) > budget:
            thetas = thetas[: budget - start]
        if control_state(out) == "stop":
            raise Paused("STOP file")
        t0 = time.time()
        new = evaluator.evaluate(thetas, map_seeds[start:start + len(thetas)])
        for i, r in enumerate(new):
            r.update(index=start + i, method=method, run_seed=seed, cell=list(cell(thetas[i])),
                     batch_wall_s=time.time() - t0)
        log.extend(new)
        rows.extend(new)
        return new

    if method == "rl":
        from .adversaries.rl import RLAdversary
        adversary = RLAdversary(scenario.DIM, seed, workers=batch, cell=cell)
        if (out / "rl/generator.zip").exists():
            adversary.load(out / "rl", evaluate, budget)
            adversary.env.used = len(rows)             # a killed partial rollout still spent its evaluations
        else:
            adversary.build(evaluate, budget, used=len(rows))
        while len(rows) + adversary.evaluations_per_update() <= budget:
            if should_stop():
                raise Paused(should_stop())
            adversary.update()
            adversary.save(out / "rl")
            print(f"rl seed {seed}: {len(rows)}/{budget} evaluations, {adversary.updates} PPO updates", flush=True)
        # the last partial update would leave scenarios the policy cannot learn from; spend them frozen
        rest = budget - len(rows)
        if rest:
            evaluate(adversary.propose_frozen(rest, rng_seed=seed))
            adversary.save(out / "rl")
    else:
        adversary = make(method, scenario.DIM, seed)
        if (out / "state.pkl").exists():
            saved = pickle.loads((out / "state.pkl").read_bytes())
            adversary.load(saved["adversary"])
            told = saved["told"]
            if len(rows) > told:                       # killed after logging, before saving state
                extra = rows[told:]
                X = adversary.ask(len(extra))
                if not np.allclose(X, [r["theta"] for r in extra]):
                    raise RuntimeError("resume mismatch: re-asked proposals differ from the logged ones")
                adversary.tell(X, np.asarray([r["score"] for r in extra]))
        while len(rows) < budget:
            if should_stop():
                raise Paused(should_stop())
            X = adversary.ask(min(batch, budget - len(rows)))
            new = evaluate(X)
            adversary.tell(X[: len(new)], np.asarray([r["score"] for r in new]))
            (out / "state.pkl").write_bytes(pickle.dumps({"adversary": adversary.state(), "told": len(rows)}))
            if len(rows) % 50 < batch:
                print(f"{method} seed {seed}: {len(rows)}/{budget} evaluations, "
                      f"{sum(r['valid_failure'] for r in rows)} valid failures", flush=True)
    write_json(out / "done.json", {"evaluations": len(rows), "valid_failures": sum(r["valid_failure"] for r in rows)})
    return {"evaluations": len(rows)}
