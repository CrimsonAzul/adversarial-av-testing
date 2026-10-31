"""Search runner on a fake evaluator: budget ledger, matched seeds, pause/resume, report."""
import subprocess
import sys

import numpy as np
import pytest

from avtest import scenario, search
from avtest.records import JsonlLog, ROOT


class FakeEvaluator:
    """Failure when the first two continuous coordinates are both high: a small target region."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, thetas, map_seeds):
        rows = []
        for theta, seed in zip(np.asarray(thetas), map_seeds):
            self.calls += 1
            hit = bool(theta[4] > .8 and theta[5] > .8)
            score = 1.0 if hit else 0.5 * max(0.0, theta[4] + theta[5] - 1.0)
            rows.append({"theta": theta.tolist(), "map_seed": seed, "score": score, "valid_failure": hit,
                         "sut_failed": hit, "avoidable": True, "outcome": "collision" if hit else "success",
                         "wall_s": 0.01})
        return rows


@pytest.fixture
def sut(tmp_path):
    path = tmp_path / "sut.zip"
    path.write_bytes(b"model")
    return path


@pytest.mark.parametrize("method", ["random", "cmaes", "bo"])
def test_ask_tell_run_logs_exact_budget_and_resumes_identically(tmp_path, sut, method):
    whole = tmp_path / "whole"
    search.run(method, 101, "T2", 40, whole, FakeEvaluator(), sut)
    split = tmp_path / "split"
    (split).mkdir()
    (split / "PAUSE").touch()
    with pytest.raises(search.Paused):
        search.run(method, 101, "T2", 40, split, FakeEvaluator(), sut)
    (split / "PAUSE").unlink()
    search.run(method, 101, "T2", 40, split, FakeEvaluator(), sut)
    a, b = JsonlLog(whole / "evaluations.jsonl").read(), JsonlLog(split / "evaluations.jsonl").read()
    assert len(a) == len(b) == 40 and [r["theta"] for r in a] == [r["theta"] for r in b]
    assert [r["map_seed"] for r in a] == search.map_seed_list(101, 40)
    assert [r["index"] for r in a] == list(range(40))


def test_resume_refuses_changed_settings(tmp_path, sut):
    search.run("random", 101, "T2", 8, tmp_path, FakeEvaluator(), sut)
    with pytest.raises(SystemExit):
        search.run("random", 101, "T2", 12, tmp_path, FakeEvaluator(), sut)


def test_rl_adversary_learns_online_within_budget_and_resumes(tmp_path, sut):
    out = tmp_path / "rl"
    evaluator = FakeEvaluator()
    budget = 3 * 24 + 5                              # three PPO updates, then a frozen remainder
    out.mkdir()
    (out / "PAUSE").touch()
    with pytest.raises(search.Paused):
        search.run("rl", 202, "T2", budget, out, evaluator, sut)
    assert evaluator.calls == 0
    (out / "PAUSE").unlink()
    search.run("rl", 202, "T2", budget, out, evaluator, sut)
    rows = JsonlLog(out / "evaluations.jsonl").read()
    assert len(rows) == budget == evaluator.calls
    assert all(0 <= x <= 1 for r in rows for x in r["theta"]) and len(rows[0]["theta"]) == scenario.DIM
    assert (out / "rl/generator.zip").exists() and (out / "done.json").exists()


def test_report_regenerates_from_logs(tmp_path, sut):
    runs = tmp_path / "runs"
    for method in ("random", "cmaes"):
        for seed in (101, 202):
            search.run(method, seed, "T2", 24, runs / f"{method}_s{seed}", FakeEvaluator(), sut)
    out = tmp_path / "report"
    subprocess.run([sys.executable, "-m", "scripts.report", "--runs", str(runs), "--rq4", str(tmp_path / "none"),
                    "--out", str(out), "--budget", "24"], check=True, cwd=ROOT)
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "cmaes − random, seeds [101, 202]" in text and (out / "discovery.png").exists()
