"""One adversarial search run: method × seed × budget, resumable, every evaluation logged.

Layout of a run folder:
    config.json        method, seed, tier, budget, SUT hash; refuses to resume with other settings
    evaluations.jsonl  one row per evaluated scenario, in evaluation order (the budget ledger)
    timing.jsonl       per batch: simulator seconds and adversary seconds (proposal/learning), separately
    state.pkl          adversary state after the last completed batch (ask/tell methods)
    rl/                PPO generator checkpoint and updates.jsonl, one learning-curve row per PPO update
    progress.png       discovery curve, severity and validity, redrawn every 50 evaluations
    recordings/        GIF + keyframes of new failures: the SUT failing and the reference that avoided it
    PAUSE / STOP       control files: stop after the current batch (RL: current update) / stop now

Map seeds come from a list fixed by the run seed, so runs of different methods with the
same seed see the same seed sequence (matched nuisance), and a resumed run uses exactly the
seeds an uninterrupted one would.

Recording policy (fixed, so it cannot favour a method): the first valid failure is recorded,
then at most one more per 100 evaluations, the first valid failure in a new failure cell.
Recording time is logged separately and never counted as the method's time.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import time

import numpy as np

from . import scenario, seeds
from .adversaries import make
from .records import JsonlLog, control_state, read_json, sha256_file, write_json
from .stats import failure_key

PROGRESS_EVERY = 50
RECORD_EVERY = 100


def map_seed_list(run_seed: int, budget: int, split: str = "search") -> list[int]:
    rng = np.random.default_rng([run_seed, 99])
    return [seeds.draw(rng, split) for _ in range(budget)]


class Paused(Exception):
    pass


class Recorder:
    """Applies the recording policy; renders through the evaluator's workers."""

    def __init__(self, out: Path, evaluator, enabled: bool):
        self.out, self.evaluator, self.enabled = out, evaluator, enabled and hasattr(evaluator, "record")
        done = sorted((out / "recordings").glob("eval_*_sut.gif")) if self.enabled else []
        self.last = int(done[-1].stem.split("_")[1]) if done else None
        self.seen: set = set()

    def observe(self, rows: list[dict], new: list[dict], log: JsonlLog) -> float:
        if not self.seen:
            self.seen = {failure_key(r) for r in rows[: len(rows) - len(new)] if r["valid_failure"]}
        started = time.time()
        for r in new:
            if not r["valid_failure"]:
                continue
            key = failure_key(r)
            fresh = key not in self.seen
            self.seen.add(key)
            due = self.last is None or (fresh and r["index"] >= self.last + RECORD_EVERY)
            if self.enabled and due:
                witness = next(ref["driver"] for ref in r["references"] if ref["outcome"] == "success")
                stem = self.out / "recordings" / f"eval_{r['index']:04d}"
                base = {"theta": r["theta"], "map_seed": r["map_seed"], "split": r.get("split", "search")}
                title = f"{self.out.name} eval {r['index']}"
                results = self.evaluator.record([dict(base, driver="sut", path=f"{stem}_sut", title=title),
                                                 dict(base, driver=witness, path=f"{stem}_{witness}", title=title)])
                log.append({"index": r["index"], "cell": r["cell"], "sut_outcome": results[0]["outcome"],
                            "witness": witness, "witness_outcome": results[1]["outcome"]})
                self.last = r["index"]
        return time.time() - started


def progress_chart(rows: list[dict], out: Path, title: str, updates: list[dict] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .stats import discovery_curve
    if not rows:
        return
    x = np.arange(1, len(rows) + 1)
    panels = 3 if updates else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 3.8))
    axes[0].plot(x, discovery_curve(rows), color="#2a78d6", lw=2, label="unique valid failure cells")
    axes[0].plot(x, np.cumsum([r["valid_failure"] for r in rows]), color="#999", lw=1, label="valid failures (all)")
    axes[0].set_xlabel("evaluations")
    axes[0].set_title(title, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=.3)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    window = min(50, len(scores))
    rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
    axes[1].plot(np.arange(window, len(scores) + 1), rolling, color="#eb6834", lw=2, label=f"mean severity (last {window})")
    failed = np.asarray([r["sut_failed"] for r in rows], dtype=float)
    invalid = np.asarray([bool(r["sut_failed"] and r["avoidable"] is False) for r in rows], dtype=float)
    axes[1].plot(np.arange(window, len(scores) + 1), np.convolve(failed, np.ones(window) / window, mode="valid"),
                 color="#d63a2a", lw=1, label="SUT failure rate")
    axes[1].plot(np.arange(window, len(scores) + 1), np.convolve(invalid, np.ones(window) / window, mode="valid"),
                 color="#888", lw=1, ls="--", label="unavoidable-failure rate")
    axes[1].set_xlabel("evaluations")
    axes[1].set_ylim(-0.55, 1.05)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=.3)
    if updates:
        u = [row["evaluations"] for row in updates]
        axes[2].plot(u, [row["mean_reward"] for row in updates], "o-", color="#1baf7a", ms=3, label="mean reward per update")
        if all("action_std" in row for row in updates):
            axes[2].plot(u, [row["action_std"] for row in updates], color="#555", lw=1, label="policy action std")
        axes[2].set_xlabel("evaluations")
        axes[2].set_title("RL generator learning", loc="left", fontsize=10)
        axes[2].legend(frameon=False, fontsize=8)
        axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "progress.png", dpi=120)
    plt.close(fig)


def run(method: str, seed: int, tier: str, budget: int, out: Path, evaluator, sut_path: Path,
        batch: int = 4, deadline: float | None = None, record: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "seed": seed, "tier": tier, "budget": budget, "batch": batch,
              "sut_sha256": sha256_file(sut_path), "dim": scenario.DIM}
    if (out / "config.json").exists():
        if read_json(out / "config.json") != config:
            raise SystemExit(f"{out} was started with different settings; use a new folder")
    else:
        write_json(out / "config.json", config)
    log = JsonlLog(out / "evaluations.jsonl")
    timing = JsonlLog(out / "timing.jsonl")
    recordings = JsonlLog(out / "recordings" / "index.jsonl")
    rows = log.read()
    map_seeds = map_seed_list(seed, budget)
    cell = lambda theta: scenario.cell(theta, tier)
    recorder = Recorder(out, evaluator, record)
    clock = {"eval_s": 0.0, "record_s": 0.0}
    last_chart = [len(rows)]

    def should_stop() -> str | None:
        return control_state(out) or ("session hours used" if deadline and time.time() > deadline else None)

    def evaluate(thetas: np.ndarray) -> list[dict]:
        start = len(rows)
        thetas = np.asarray(thetas)[: budget - start]
        if control_state(out) == "stop":
            raise Paused("STOP file")
        t0 = time.time()
        new = evaluator.evaluate(thetas, map_seeds[start:start + len(thetas)])
        clock["eval_s"] += time.time() - t0
        for i, r in enumerate(new):
            r.update(index=start + i, method=method, run_seed=seed, cell=list(cell(thetas[i])), split="search")
        log.extend(new)
        rows.extend(new)
        clock["record_s"] += recorder.observe(rows, new, recordings)
        return new

    def checkpoint_progress(updates=None):
        if len(rows) - last_chart[0] >= PROGRESS_EVERY or len(rows) == budget:
            progress_chart(rows, out, f"{method} seed {seed}", updates)
            last_chart[0] = len(rows)
            print(f"{method} seed {seed}: {len(rows)}/{budget} evaluations | valid failures "
                  f"{sum(r['valid_failure'] for r in rows)} | unique cells {len({failure_key(r) for r in rows if r['valid_failure']})}",
                  flush=True)

    if method == "rl":
        from .adversaries.rl import RLAdversary
        updates_log = JsonlLog(out / "rl" / "updates.jsonl")
        adversary = RLAdversary(scenario.DIM, seed, workers=batch, cell=cell)
        if (out / "rl/generator.zip").exists():
            adversary.load(out / "rl", evaluate, budget)
            adversary.env.used = len(rows)             # a killed partial rollout still spent its evaluations
        else:
            adversary.build(evaluate, budget, used=len(rows))
        while len(rows) + adversary.evaluations_per_update() <= budget:
            if should_stop():
                raise Paused(should_stop())
            before, clock["eval_s"], clock["record_s"] = len(rows), 0.0, 0.0
            t0 = time.time()
            adversary.update()
            elapsed = time.time() - t0
            adversary.save(out / "rl")
            batch_rows = rows[before:]
            stats = adversary.diagnostics()
            updates_log.append({"update": adversary.updates, "evaluations": len(rows),
                                "mean_score": float(np.mean([r["score"] for r in batch_rows])),
                                "valid_failures": int(sum(r["valid_failure"] for r in batch_rows)),
                                "sut_failures": int(sum(r["sut_failed"] for r in batch_rows)), **stats})
            timing.append({"evaluations": len(rows), "eval_s": clock["eval_s"], "record_s": clock["record_s"],
                           "adversary_s": elapsed - clock["eval_s"] - clock["record_s"]})
            checkpoint_progress(updates_log.read())
        rest = budget - len(rows)                       # a partial update cannot be learned from; spend it frozen
        if rest:
            clock["eval_s"], clock["record_s"] = 0.0, 0.0
            t0 = time.time()
            evaluate(adversary.propose_frozen(rest, rng_seed=seed))
            timing.append({"evaluations": len(rows), "eval_s": clock["eval_s"], "record_s": clock["record_s"],
                           "adversary_s": time.time() - t0 - clock["eval_s"] - clock["record_s"]})
            adversary.save(out / "rl")
            checkpoint_progress(updates_log.read())
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
            clock["eval_s"], clock["record_s"] = 0.0, 0.0
            t0 = time.time()
            X = adversary.ask(min(batch, budget - len(rows)))
            ask_s = time.time() - t0
            new = evaluate(X)
            t1 = time.time()
            adversary.tell(X[: len(new)], np.asarray([r["score"] for r in new]))
            (out / "state.pkl").write_bytes(pickle.dumps({"adversary": adversary.state(), "told": len(rows)}))
            timing.append({"evaluations": len(rows), "eval_s": clock["eval_s"], "record_s": clock["record_s"],
                           "adversary_s": ask_s + time.time() - t1})
            checkpoint_progress()
    write_json(out / "done.json", {"evaluations": len(rows), "valid_failures": sum(r["valid_failure"] for r in rows)})
    return {"evaluations": len(rows)}
