"""Pure-Python rules: seeds, scenario decoding, severity, statistics, adversary interfaces."""
import numpy as np
import pytest

from avtest import scenario, seeds, severity, stats
from avtest.adversaries import make


def test_seed_splits_are_disjoint_and_enforced():
    ranges = sorted(seeds.SPLITS.values())
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    panel = seeds.panel_seeds("qual", 500, 1)
    assert len(set(panel)) == 500 and all(seeds.check(s, "qual") for s in panel)
    with pytest.raises(ValueError):
        seeds.check(5, "qual")


@pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
def test_decode_respects_tier_and_plausible_ranges(tier):
    rng = np.random.default_rng(0)
    for theta in scenario.sample(rng, 300):
        v = scenario.decode(theta, 7, tier).values
        assert set(v["map"]) <= set(scenario.TIERS[tier]["blocks"]) and len(v["map"]) == 3
        assert v["event"] in scenario.TIERS[tier]["events"]
        assert 0 <= v["traffic_density"] <= scenario.TIERS[tier]["max_traffic"]
        assert 2 <= v["brake_decel_mps2"] <= 6 and 1.5 <= v["cut_in_duration_s"] <= 4
        assert v["lane_num"] in (2, 3) and 3.0 <= v["lane_width"] <= 4.0
        if v["event"] == "lead_brake":
            assert v["event_lane"] == "same"
        if v["event"] == "cut_in":
            assert v["event_lane"] in ("left", "right")
    with pytest.raises(ValueError):
        scenario.decode(np.full(scenario.DIM, 1.2), 7, tier)


def test_cells_cover_categories_and_three_continuous_bins():
    c = scenario.cell(np.zeros(scenario.DIM), "T3")
    assert c == (0,) * scenario.DIM
    top = scenario.cell(np.ones(scenario.DIM), "T3")
    assert top[0] == 6 and top[3] == 1 and top[4] == 2


def test_severity_orders_outcomes_and_punishes_unavoidable():
    calm = {"min_ttc_s": None, "min_edge_margin_m": 2.0}
    close = {"min_ttc_s": 0.6, "min_edge_margin_m": 2.0}
    assert severity.score(False, None, None, calm) == 0
    assert 0 < severity.score(False, None, None, close) < 0.5
    assert severity.score(True, False, None, calm) == -0.5
    assert severity.score(True, True, False, calm) == 0.6
    assert severity.score(True, True, True, calm) == 1.0


def row(cell, valid, outcome="collision", failed=None, avoidable=True):
    return {"cell": cell, "valid_failure": valid, "outcome": outcome,
            "sut_failed": valid if failed is None else failed, "avoidable": avoidable}


def test_discovery_counts_unique_cells_once():
    rows = [row((1,), False), row((1,), True), row((1,), True), row((2,), True, "offroad")]
    assert stats.discovery_curve(rows).tolist() == [0, 1, 1, 2]
    assert stats.unique_failures(rows) == 2 and stats.first_failure(rows) == 2
    assert stats.auc(rows, 4) == pytest.approx(1.0) and stats.auc(rows[:2], 4) == pytest.approx(0.75)
    assert stats.invalid_fraction([row((1,), False, failed=True, avoidable=False), row((2,), True)]) == 0.5
    assert stats.bootstrap_mean([2, 2, 2]) == (2.0, 2.0, 2.0)
    low, high = stats.wilson(490, 500)
    assert low < 0.98 < high


@pytest.mark.parametrize("name", ["random", "cmaes", "bo"])
def test_ask_tell_adversaries_stay_in_bounds_and_resume_exactly(name):
    dim = scenario.DIM
    objective = lambda X: -np.sum((X - 0.7) ** 2, axis=1)
    a = make(name, dim, 101)
    for _ in range(9):
        X = a.ask(4)
        assert X.shape[1] == dim and X.min() >= 0 and X.max() <= 1
        a.tell(X, objective(X))
    b = make(name, dim, 101)
    b.load(a.state())
    assert np.allclose(a.ask(4), b.ask(4))
