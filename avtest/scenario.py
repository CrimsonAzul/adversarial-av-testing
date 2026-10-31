"""The scenario space: θ ∈ [0,1]^12 plus a map seed, decoded into one MetaDrive episode.

The same decoder serves driver training, qualification, random search and every
adversary. Ranges are bounded to plausible values, so a decoded scenario is
plausible by construction (no post-hoc filter):

- NPC decelerations ≤ 6 m/s² (hard but ordinary braking; emergency braking reaches ~8–9);
- cut-in lane changes take 1.5–4 s (≈ 0.9–2.3 m/s lateral speed for a 3.5 m lane);
- the event is defined relative to the ego, as in OpenSCENARIO: until the trigger time the
  event vehicle drives ``event_ahead_m`` ahead of the ego along the route (centre to
  centre) at the ego's speed, then takes the ego's speed plus ``event_rel_speed_mps``.
  Every driver therefore meets the same event geometry, whatever its own cruise speed;
  this is what makes "a reference driver avoided it" a fair test of avoidability;
- the offset range depends on the event: a lead vehicle in the ego's lane is 12–45 m
  ahead (at least a 7.5 m bumper gap); a cut-in vehicle in the adjacent lane is 3–30 m
  ahead (it may start alongside); a broken-down vehicle (obstacle) stands 40–160 m
  ahead of the ego's start;
- background traffic is MetaDrive's own IDM traffic at the tier's density.

Categorical parameters use equal-width bins of their coordinate. ``cell`` maps θ to
a parameter-grid cell (categories as-is, continuous values in 3 bins); stats and the
RL novelty bonus count failures per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIERS = {
    "T1": {"blocks": "SC", "max_traffic": 0.10, "events": ("lead_brake", "obstacle")},
    "T2": {"blocks": "SCrR", "max_traffic": 0.15, "events": ("lead_brake", "obstacle", "cut_in")},
    "T3": {"blocks": "SCrRXTO", "max_traffic": 0.20, "events": ("lead_brake", "obstacle", "cut_in")},
}
LANES = (2, 3)
EVENT_LANES = ("same", "left", "right")
AHEAD_M = {"lead_brake": (12.0, 45.0), "cut_in": (3.0, 30.0), "obstacle": (40.0, 160.0)}


@dataclass(frozen=True)
class Param:
    name: str
    low: float = 0.0
    high: float = 1.0
    categories: tuple | None = None   # set → categorical; values depend on tier for blocks/events

    def bins(self, tier: str) -> int:
        return len(self.options(tier)) if self.categorical else 3

    @property
    def categorical(self) -> bool:
        return self.categories is not None

    def options(self, tier: str) -> tuple:
        if self.categories == ("tier_blocks",):
            return tuple(TIERS[tier]["blocks"])
        if self.categories == ("tier_events",):
            return TIERS[tier]["events"]
        return self.categories


PARAMS = (
    Param("block_1", categories=("tier_blocks",)),
    Param("block_2", categories=("tier_blocks",)),
    Param("block_3", categories=("tier_blocks",)),
    Param("lane_num", categories=LANES),
    Param("lane_width", 3.0, 4.0),
    Param("traffic_density", 0.0, 1.0),               # scaled by the tier's maximum
    Param("event", categories=("tier_events",)),
    Param("event_lane", categories=EVENT_LANES),
    Param("event_ahead", 0.0, 1.0),                  # offset ahead of the ego; metres depend on the event
    Param("event_rel_speed_mps", -4.0, 2.0),         # event vehicle speed minus ego speed at the trigger
    Param("event_trigger_s", 3.0, 15.0),             # seconds after the start
    Param("event_intensity", 0.0, 1.0),              # braking 2–6 m/s², or cut-in duration 4–1.5 s
)
DIM = len(PARAMS)


@dataclass(frozen=True)
class Scenario:
    theta: tuple
    map_seed: int
    tier: str
    values: dict = field(compare=False)

    def to_dict(self) -> dict:
        return {"theta": list(self.theta), "map_seed": self.map_seed, "tier": self.tier, "values": self.values}


def _category(x: float, options: tuple):
    return options[min(int(x * len(options)), len(options) - 1)]


def decode(theta, map_seed: int, tier: str) -> Scenario:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (DIM,) or not np.all(np.isfinite(theta)) or theta.min() < 0 or theta.max() > 1:
        raise ValueError(f"theta must be {DIM} finite values in [0, 1]")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier}")
    values = {}
    for p, x in zip(PARAMS, theta):
        values[p.name] = _category(x, p.options(tier)) if p.categorical else p.low + x * (p.high - p.low)
    values["traffic_density"] *= TIERS[tier]["max_traffic"]
    intensity = values["event_intensity"]
    values["brake_decel_mps2"] = 2.0 + 4.0 * intensity
    values["cut_in_duration_s"] = 4.0 - 2.5 * intensity
    low, high = AHEAD_M[values["event"]]
    values["event_ahead_m"] = low + values.pop("event_ahead") * (high - low)
    if values["event"] == "lead_brake":
        values["event_lane"] = "same"               # a braking lead vehicle is in the ego's lane by definition
    elif values["event"] == "cut_in" and values["event_lane"] == "same":
        values["event_lane"] = "left"               # a cut-in starts from an adjacent lane
    values["map"] = values["block_1"] + values["block_2"] + values["block_3"]
    values["lane_num"] = int(values["lane_num"])
    return Scenario(tuple(float(x) for x in theta), int(map_seed), tier, values)


def cell(theta, tier: str) -> tuple:
    theta = np.asarray(theta, dtype=float)
    return tuple(min(int(x * p.bins(tier)), p.bins(tier) - 1) for p, x in zip(PARAMS, theta))


def with_event(theta, event: str, tier: str) -> np.ndarray:
    """θ with its event coordinate moved to the centre of ``event``'s bin (other values unchanged)."""
    theta = np.array(theta, dtype=float)
    events = TIERS[tier]["events"]
    theta[[p.name for p in PARAMS].index("event")] = (events.index(event) + 0.5) / len(events)
    return theta


def sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform θ: the qualification distribution and the random-search proposal."""
    return rng.random((n, DIM))
