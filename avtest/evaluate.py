"""Parallel scenario evaluation: one MetaDrive process per worker, one record per scenario.

Per scenario:
1. SUT episode.
2. If the SUT failed: replay the SUT once (reproduction check), then run the references in
   order (IDM, cautious IDM, expert) until one completes the scene (avoidability).
   With ``all_references`` (qualification) every reference runs on every scene.
3. Severity score (avtest.severity) and the valid-failure flag.

Record keys used downstream: theta, map_seed, tier, values, sut, replay, references,
sut_failed, outcome, avoidable, reproduced, valid_failure, score, episodes, wall_s.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from . import scenario as sc
from . import seeds, severity

_worker: dict = {}


def _init(sut_path: str | None):
    import torch  # noqa: F401  (before MetaDrive)
    torch.set_num_threads(1)
    from .sim import AVTestEnv
    from .drivers import SUTDriver, references
    _worker["env"] = AVTestEnv()
    _worker["sut"] = SUTDriver(Path(sut_path)) if sut_path else None
    _worker["references"] = references()


def evaluate_one(task: dict) -> dict:
    from .sim import run_episode
    started = time.time()
    env = _worker["env"]
    scenario = sc.decode(np.asarray(task["theta"]), seeds.check(task["map_seed"], task["split"]), task["tier"])
    if _worker["sut"] is None:                        # reference-only calibration
        refs = [run_episode(env, scenario, driver) for driver in _worker["references"]]
        return {"theta": list(map(float, task["theta"])), "map_seed": task["map_seed"], "tier": task["tier"],
                "values": scenario.values, "references": refs,
                "avoidable": any(r["outcome"] == "success" for r in refs), "wall_s": time.time() - started}
    sut = run_episode(env, scenario, _worker["sut"])
    failed = sut["outcome"] != "success"
    replay, refs, avoidable, reproduced = None, [], None, None
    if failed and task["references"]:
        replay = run_episode(env, scenario, _worker["sut"])
        reproduced = replay["outcome"] == sut["outcome"]
    if task["references"] and (failed or task["all_references"]):
        for driver in _worker["references"]:
            refs.append(run_episode(env, scenario, driver))
            if refs[-1]["outcome"] == "success" and not task["all_references"]:
                break
        avoidable = any(r["outcome"] == "success" for r in refs)
    valid = bool(failed and avoidable and reproduced)
    return {"theta": list(map(float, task["theta"])), "map_seed": task["map_seed"], "tier": task["tier"],
            "values": scenario.values, "sut": sut, "replay": replay, "references": refs,
            "sut_failed": failed, "outcome": sut["outcome"], "avoidable": avoidable, "reproduced": reproduced,
            "valid_failure": valid,
            "score": severity.score(failed, avoidable, reproduced, sut) if task["references"] else None,
            "episodes": 1 + (replay is not None) + len(refs), "wall_s": time.time() - started}


class Evaluator:
    def __init__(self, sut_path: Path | None, tier: str, workers: int = 4, references: bool = True,
                 all_references: bool = False):
        self.tier, self.references, self.all_references = tier, references, all_references
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init,
                                                 initargs=(str(sut_path) if sut_path else None,),
                                                 maxtasksperchild=200)

    def evaluate(self, thetas, map_seeds, split: str = "search") -> list[dict]:
        tasks = [{"theta": list(map(float, t)), "map_seed": int(s), "split": split, "tier": self.tier,
                  "references": self.references, "all_references": self.all_references}
                 for t, s in zip(np.asarray(thetas), map_seeds)]
        return self.pool.map(evaluate_one, tasks, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
