"""Tables and figures for RQ1–RQ4, rebuilt from evaluation logs only.

    .venv\\Scripts\\python.exe -m scripts.report                 # runs/ → reports/
    .venv\\Scripts\\python.exe -m scripts.report --runs some/dir --out some/report

Unfinished runs are reported with their evaluation count and are never extrapolated.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avtest import stats  # noqa: E402
from avtest.records import JsonlLog, ROOT, read_json  # noqa: E402

COLORS = {"rl": "#2a78d6", "cmaes": "#eb6834", "bo": "#1baf7a", "random": "#7a7a7a"}
ORDER = ("rl", "cmaes", "bo", "random")


def load_runs(folder: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for config_path in sorted(folder.glob("*/config.json")):
        config = read_json(config_path)
        rows = JsonlLog(config_path.parent / "evaluations.jsonl").read()
        timing = JsonlLog(config_path.parent / "timing.jsonl").read()
        runs.setdefault(config["method"], {})[config["seed"]] = {
            "config": config, "rows": rows, "timing": timing, "done": (config_path.parent / "done.json").exists()}
    return runs


def fmt(t) -> str:
    return "n/a" if np.isnan(t[0]) else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def search_tables(runs: dict, out: Path, budget: int) -> list[str]:
    lines = ["## RQ1–RQ3: search runs", "",
             "| method | seed | evaluations | valid failures | unique failure cells | per 1,000 evals | AUC | first failure | invalid share | entropy (bits) | hours (adversary share) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    table = []
    for method in [m for m in ORDER if m in runs]:
        for seed, run in sorted(runs[method].items()):
            rows = run["rows"]
            n = len(rows)
            unique = stats.unique_failures(rows)
            row = {"method": method, "seed": seed, "evaluations": n, "done": run["done"],
                   "valid_failures": sum(r["valid_failure"] for r in rows), "unique_cells": unique,
                   "per_1000": 1000 * unique / n if n else float("nan"), "auc": stats.auc(rows, budget),
                   "first_failure": stats.first_failure(rows), "invalid_share": stats.invalid_fraction(rows),
                   "entropy_bits": stats.cell_entropy(rows),
                   "simulator_hours": sum(t["eval_s"] for t in run["timing"]) / 3600,
                   "adversary_hours": sum(t["adversary_s"] for t in run["timing"]) / 3600}
            row["wall_hours"] = row["simulator_hours"] + row["adversary_hours"]
            table.append(row)
            hours = (f"{row['wall_hours']:.2f} ({row['adversary_hours'] / row['wall_hours']:.0%})"
                     if row["wall_hours"] else "n/a")
            lines.append(f"| {method} | {seed}{'' if run['done'] else ' (unfinished)'} | {n} | {row['valid_failures']} | "
                         f"{unique} | {row['per_1000']:.1f} | {row['auc']:.2f} | {row['first_failure'] or 'none'} | "
                         f"{row['invalid_share']:.2f} | {row['entropy_bits']:.2f} | {hours} |")
    if table:
        with (out / "search_runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines += ["", "### Differences against random and against the best baseline (matched seeds, bootstrap over seeds)", ""]
    if "random" in runs:
        for method in [m for m in ORDER if m in runs and m != "random"]:
            matched = sorted(set(runs[method]) & set(runs["random"]))
            diff_unique = [stats.unique_failures(runs[method][s]["rows"], budget)
                           - stats.unique_failures(runs["random"][s]["rows"], budget) for s in matched]
            diff_auc = [stats.auc(runs[method][s]["rows"], budget) - stats.auc(runs["random"][s]["rows"], budget)
                        for s in matched]
            lines.append(f"- {method} − random, seeds {matched}: unique cells {fmt(stats.bootstrap_mean(diff_unique))}; "
                         f"AUC {fmt(stats.bootstrap_mean(diff_auc))}")
    if "rl" in runs:
        baselines = [m for m in ("cmaes", "bo") if m in runs]
        if baselines:
            best = max(baselines, key=lambda m: np.mean([stats.auc(r["rows"], budget) for r in runs[m].values()]))
            matched = sorted(set(runs["rl"]) & set(runs[best]))
            diff = [stats.auc(runs["rl"][s]["rows"], budget) - stats.auc(runs[best][s]["rows"], budget) for s in matched]
            lines.append(f"- **Primary contrast** rl − {best} (best baseline by mean AUC), seeds {matched}: "
                         f"AUC {fmt(stats.bootstrap_mean(diff))}")
    lines.append("\nWith 3 seeds the bootstrap intervals are unstable; per-seed values above are the primary evidence.")
    return lines


def discovery_figure(runs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in [m for m in ORDER if m in runs]:
        for i, run in enumerate(runs[method].values()):
            curve = stats.discovery_curve(run["rows"])
            ax.plot(np.arange(1, len(curve) + 1), curve, color=COLORS[method], alpha=.85, lw=1.6,
                    label=method if i == 0 else None)
    ax.set_xlabel("SUT evaluations")
    ax.set_ylabel("unique valid failure cells")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "discovery.png", dpi=160)
    fig.savefig(out / "discovery.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=ROOT / "runs/search")
    p.add_argument("--rq4", type=Path, default=ROOT / "runs/rq4")
    p.add_argument("--out", type=Path, default=ROOT / "reports")
    p.add_argument("--budget", type=int, default=1000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# Adversarial search results (generated by scripts/report.py)", ""]
    runs = load_runs(args.runs) if args.runs.exists() else {}
    if runs:
        lines += search_tables(runs, args.out, args.budget)
        discovery_figure(runs, args.out)
        lines += ["", "![discovery curves](discovery.png)"]
    else:
        lines.append("No search runs yet.")
    rq4 = load_runs(args.rq4) if args.rq4.exists() else {}
    lines += ["", "## RQ4: frozen RL generator vs random on held-out map seeds", ""]
    if {"rl_frozen", "random"} <= set(rq4):
        matched = sorted(set(rq4["rl_frozen"]) & set(rq4["random"]))
        diff = [stats.unique_failures(rq4["rl_frozen"][s]["rows"]) - stats.unique_failures(rq4["random"][s]["rows"])
                for s in matched]
        lines.append(f"Seeds {matched}: unique valid failure cells, frozen RL − random: {fmt(stats.bootstrap_mean(diff))}")
    else:
        lines.append("Not run yet.")
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
