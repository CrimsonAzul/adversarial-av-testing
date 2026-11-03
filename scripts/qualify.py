"""SUT qualification gate (docs/DESIGN.md): widest passing ODD tier on a fresh held-out panel.

    .venv\\Scripts\\python.exe -m scripts.qualify --model runs\\driver\\d1\\model_best.zip --attempt 1

For each tier in T3, T2, T1 order: 500 scenarios with uniform θ and qualification-split
map seeds (panel seed 1000 + attempt, so every attempt uses a new panel). The panel is
written and hashed before any episode runs. Every scene runs the SUT and all three
reference drivers. Pass when, over scenes at least one reference completes:
    SUT success ≥ 98%, 95% Wilson lower bound ≥ 96%, collision + off-road ≤ 2%.
The first passing tier is frozen in models/QUALIFIED.json with the model hash and the SUT
controller it drives with (avtest.tactical.controller_for_model), and the model is copied to models/sut.zip.
A failed attempt's panel becomes development data; re-test only with a new --attempt number.
Results append to docs/QUALIFICATION.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import torch  # noqa: F401  (before MetaDrive)
import numpy as np

from avtest import scenario, seeds, stats
from avtest.records import JsonlLog, ROOT, read_json, sha256_file, sha256_json, write_json
from avtest.tactical import controller_for_model

PANEL = 500
SUCCESS_MIN, WILSON_MIN, CRASH_MAX = 0.98, 0.96, 0.02


def gate(rows: list[dict]) -> dict:
    solvable = [r for r in rows if any(ref["outcome"] == "success" for ref in r["references"])]
    n = len(solvable)
    ok = sum(r["sut"]["outcome"] == "success" for r in solvable)
    crashes = sum(r["sut"]["outcome"] in ("collision", "offroad") for r in solvable)
    low, high = stats.wilson(ok, n)
    result = {"scenes": len(rows), "reference_solvable": n, "sut_success": ok,
              "success_rate": ok / n if n else 0.0, "wilson_95": [low, high],
              "collision_offroad_rate": crashes / n if n else 1.0,
              "outcomes": {k: sum(r["sut"]["outcome"] == k for r in solvable)
                           for k in ("success", "collision", "offroad", "stuck", "timeout")},
              "unsolvable_scenes": len(rows) - n,
              "sut_success_on_unsolvable": sum(r["sut"]["outcome"] == "success" for r in rows) - ok}
    result["passed"] = bool(n and result["success_rate"] >= SUCCESS_MIN and low >= WILSON_MIN
                            and result["collision_offroad_rate"] <= CRASH_MAX)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--tiers", default="T3,T2,T1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--panel", type=int, default=PANEL, help="only for pipeline checks; the gate uses 500")
    args = p.parse_args()
    from avtest.evaluate import Evaluator

    from stable_baselines3 import PPO
    from avtest.sim import OBS_DIM
    obs_size = int(PPO.load(args.model, device="cpu").observation_space.shape[0])
    if obs_size != OBS_DIM:
        raise SystemExit("this driver uses the d1–d3 sensors; qualification is for today's sensors")
    controller = controller_for_model(args.model, obs_size)
    model_hash = sha256_file(args.model)
    panel_seed = 1000 + args.attempt
    map_seeds = seeds.panel_seeds("qual", args.panel, panel_seed)
    thetas = scenario.sample(np.random.default_rng(panel_seed), args.panel)
    lines = []
    for tier in args.tiers.split(","):
        out = ROOT / "runs/qualification" / f"attempt{args.attempt}_{tier}"
        # the SUT is its model (by hash, not path, so a resume may spell the path differently) plus its controller
        panel = {"tier": tier, "attempt": args.attempt, "panel_seed": panel_seed, "map_seeds": map_seeds,
                 "thetas": thetas.tolist(), "model_sha256": model_hash, "controller": controller}
        if (out / "panel.json").exists():
            if read_json(out / "panel.json")["sha256"] != sha256_json(panel):
                raise SystemExit(f"{out} holds a different panel or model; use a new --attempt")
        else:
            write_json(out / "panel.json", {"sha256": sha256_json(panel), "model": str(args.model), **panel})
        log = JsonlLog(out / "episodes.jsonl")
        done = log.read()
        if len(done) < len(map_seeds):
            with Evaluator(args.model, tier, workers=args.workers, references=True, all_references=True) as ev:
                for start in range(len(done), len(map_seeds), 20):     # resumable in blocks of 20 scenes
                    log.extend(ev.evaluate(thetas[start:start + 20], map_seeds[start:start + 20], split="qual"))
                    print(f"{tier}: {min(start + 20, len(map_seeds))}/{len(map_seeds)} scenes", flush=True)
        result = gate(log.read())
        write_json(out / "result.json", result)
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"{tier}: {verdict}  success {result['success_rate']:.1%} on {result['reference_solvable']} solvable scenes, "
              f"Wilson [{result['wilson_95'][0]:.3f}, {result['wilson_95'][1]:.3f}], "
              f"collision/off-road {result['collision_offroad_rate']:.1%}", flush=True)
        lines.append(f"| {args.attempt} | {tier} | `{model_hash[:12]}` | {result['reference_solvable']} | "
                     f"{result['success_rate']:.1%} | {result['wilson_95'][0]:.3f} | {result['collision_offroad_rate']:.1%} | "
                     f"{result['outcomes']} | **{verdict}** |")
        if result["passed"]:
            if args.panel != PANEL:
                print("pipeline check only (panel smaller than 500); nothing frozen", flush=True)
                break
            (ROOT / "models").mkdir(exist_ok=True)
            shutil.copy2(args.model, ROOT / "models/sut.zip")
            write_json(ROOT / "models/QUALIFIED.json", {
                "tier": tier, "attempt": args.attempt, "model_sha256": model_hash, "controller": controller,
                "obs_size": obs_size,
                "source": str(args.model),
                "result": result, "utc": datetime.now(timezone.utc).isoformat()})
            break
    report = ROOT / "docs/QUALIFICATION.md"
    if not report.exists():
        report.write_text("# Qualification attempts (appended by scripts/qualify.py)\n\n"
                          "| attempt | tier | model | solvable scenes | SUT success | Wilson low | collision/off-road | outcomes | verdict |\n"
                          "|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    if args.panel == PANEL:
        with report.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
