"""The single severity score every adversary maximises (docs/DESIGN.md).

    valid failure (SUT fails, scene avoidable, failure reproduced)     1.0
    SUT failure not reproduced on replay                               0.6
    SUT failure no reference could avoid (invalid)                    -0.5
    SUT success: near-miss shaping                                     0 .. 0.5

Near-miss shaping = 0.5 · max(TTC term, edge term, progress term) with
    TTC term      = clip(1 - min_ttc / 3 s, 0, 1)       (0 when nothing closed in)
    edge term     = clip(1 - min_edge_margin / 1 m, 0, 1)
    progress term = 0 (a success has completed the route)
so a success that never came within 3 s of a collision or 1 m of the road edge
scores 0, and the score rises smoothly towards the failure boundary. The -0.5 for
unavoidable failures stops any method from earning reward with impossible scenes.
"""
from __future__ import annotations

TTC_HORIZON_S = 3.0
EDGE_HORIZON_M = 1.0


def near_miss(episode: dict) -> float:
    ttc = episode.get("min_ttc_s")
    ttc_term = 0.0 if ttc is None else min(max(1 - ttc / TTC_HORIZON_S, 0.0), 1.0)
    edge = episode.get("min_edge_margin_m")
    edge_term = 0.0 if edge is None else min(max(1 - edge / EDGE_HORIZON_M, 0.0), 1.0)
    return max(ttc_term, edge_term)


def score(sut_failed: bool, avoidable: bool | None, reproduced: bool | None, episode: dict) -> float:
    if not sut_failed:
        return 0.5 * near_miss(episode)
    if avoidable is False:
        return -0.5
    if reproduced:
        return 1.0
    return 0.6
