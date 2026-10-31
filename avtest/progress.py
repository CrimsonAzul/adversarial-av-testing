"""Driver-training progress: validation on a fixed panel, showcase recordings, progress chart.

Imitation (scripts/pretrain_bc.py) and PPO (scripts/train_driver.py) both call
``checkpoint_report`` after every round/chunk, so their numbers are directly comparable:
- validation: the same 200 scenarios (val split, uniform θ over T3), SUT driving
  deterministically, no references (this panel selects checkpoints; the qualification
  panel is separate and untouched);
- recordings: the same three showcase scenarios (one per event type), so the GIFs show the
  same situations getting better over training;
- chart: success overall and per event type against the 98% gate, regenerated each time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import scenario as sc
from . import seeds

TIER = "T3"
VAL_PANEL = 200
VAL_PANEL_SEED = 7
OUTCOMES = ("success", "collision", "offroad", "stuck", "timeout")
JUNCTIONS = set("XTO")


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.0%}"


def headline(report: dict) -> str:
    """One-line validation summary for console progress."""
    by = report["val_success_by_event"]
    return (f"val {pct(report['val_success'])} (" + ", ".join(f"{e} {pct(by[e])}" for e in by) + ")")


def validation_panel(size: int = VAL_PANEL) -> tuple[np.ndarray, list[int]]:
    return (sc.sample(np.random.default_rng(VAL_PANEL_SEED), VAL_PANEL)[:size],
            seeds.panel_seeds("val", VAL_PANEL, VAL_PANEL_SEED)[:size])


def summarise(rows: list[dict]) -> dict:
    ok = lambda rs: float(np.mean([r["sut"]["outcome"] == "success" for r in rs])) if rs else None
    by_event = {e: ok([r for r in rows if r["values"]["event"] == e]) for e in sc.TIERS[TIER]["events"]}
    plain = [r for r in rows if not JUNCTIONS & set(r["values"]["map"])]
    return {"val_success": ok(rows), "val_scenes": len(rows),
            "val_outcomes": {k: sum(r["sut"]["outcome"] == k for r in rows) for k in OUTCOMES},
            "val_success_by_event": by_event,
            "val_success_no_junctions": ok(plain),
            "val_success_with_junctions": ok([r for r in rows if r not in plain]),
            "val_mean_success_time_s": float(np.mean([0.1 * r["sut"]["steps"] for r in rows
                                                      if r["sut"]["outcome"] == "success"] or [np.nan]))}


def checkpoint_report(model_path: Path, run_dir: Path, label: str, workers: int = 4,
                      panel: int = VAL_PANEL, record: bool = True) -> dict:
    """Validate ``model_path`` and record the showcase scenarios under run_dir/recordings/<label>/."""
    from .evaluate import Evaluator
    from .recording import showcase
    thetas, map_seeds = validation_panel(panel)
    with Evaluator(model_path, TIER, workers=workers, references=False) as evaluator:
        rows = evaluator.evaluate(thetas, map_seeds, split="val")
        recorded = []
        if record:
            items = [{"theta": list(s.theta), "map_seed": s.map_seed, "split": "val", "driver": "sut",
                      "path": str(run_dir / "recordings" / label / s.values["event"]),
                      "title": f"{run_dir.name} {label}"} for s in showcase(TIER)]
            recorded = [{"event": Path(r["gif"]).stem, "outcome": r["outcome"]} for r in evaluator.record(items)]
    summary = summarise(rows)
    summary["showcase"] = recorded
    return summary


def chart(history: list[dict], x_key: str, x_label: str, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [h for h in history if h.get("val_success") is not None]
    if not rows:
        return
    x = [h[x_key] for h in rows]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
    ax.plot(x, [h["val_success"] for h in rows], "o-", color="#111", lw=2, label=f"all {rows[-1]['val_scenes']} scenes")
    for event, colour in zip(sc.TIERS[TIER]["events"], ("#2a78d6", "#eb6834", "#1baf7a")):
        ax.plot(x, [h["val_success_by_event"][event] for h in rows], ".--", color=colour, label=event)
    ax.axhline(0.98, color="#c00", lw=1, ls=":", label="qualification gate 98%")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel("validation success")
    ax.set_title(title, loc="left")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    bottom = np.zeros(len(rows))
    for outcome, colour in zip(OUTCOMES, ("#1baf7a", "#d63a2a", "#eda100", "#8e7cc3", "#999")):
        values = np.asarray([h["val_outcomes"][outcome] for h in rows], dtype=float)
        bx.bar(range(len(rows)), values, bottom=bottom, color=colour, label=outcome)
        bottom += values
    bx.set_xticks(range(len(rows)), [str(v) if not isinstance(v, (int, float)) or v < 1e5 else f"{v / 1e6:.1f}M"
                                     for v in x], rotation=45, fontsize=7)
    bx.set_xlabel(x_label)
    bx.set_ylabel("validation outcomes (scenes)")
    bx.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
