# Scenario space

Source of truth: `avtest/scenario.py` (decoder) and `avtest/sim.py` (how events are executed). A scenario is θ ∈ [0,1]^12 plus a MetaDrive map seed. The decoder is shared by driver training, qualification, random search and every adversary.

| # | Parameter | Type | Range / values | Notes |
|---|---|---|---|---|
| 1–3 | block_1..3 | categorical | T1: S C · T2: + r R · T3: + X T O | After MetaDrive's start block. S straight, C curve, r on-ramp, R off-ramp, X intersection, T T-junction, O roundabout. Curve radii and similar block details come from the map seed. |
| 4 | lane_num | categorical | 2, 3 | |
| 5 | lane_width | continuous | 3.0 – 4.0 m | |
| 6 | traffic_density | continuous | 0 – tier max (0.10 / 0.15 / 0.20) | MetaDrive IDM background traffic, deterministic per map seed |
| 7 | event | categorical | T1: lead_brake, obstacle · T2/T3: + cut_in | Exactly one scripted event per scenario |
| 8 | event_lane | categorical | same, left, right | lead_brake is always same lane; cut_in starts from an adjacent lane; mirrored when the ego is at the road edge |
| 9 | event_ahead | continuous | lead_brake 12–45 m · cut_in 3–30 m · obstacle 40–160 m | Centre-to-centre offset ahead of the ego along the route (obstacle: ahead of the start) |
| 10 | event_rel_speed_mps | continuous | −4 … +2 m/s | Event vehicle speed minus ego speed at the trigger |
| 11 | event_trigger_s | continuous | 3 – 15 s | Seconds after the start |
| 12 | event_intensity | continuous | 0 – 1 | lead_brake: deceleration 2–6 m/s² · cut_in: lane change over 4.0–1.5 s |

**How an event runs.** Until the trigger, the event vehicle drives at the ego's speed, `event_ahead` ahead of it along the route. At the trigger its speed becomes the ego's speed plus `event_rel_speed_mps`, and from then on it moves on its own:

- a lead vehicle brakes to a stop, waits 2 s, then accelerates at 1.5 m/s² back to its trigger speed;
- a cut-in vehicle moves into the ego's lane along a smooth cosine profile;
- an obstacle is a stopped vehicle with a warning triangle 8 m behind it, there from the start.

Every driver, the SUT and each reference, therefore meets the same relative event geometry whatever its own cruise speed. Event vehicles move kinematically, so their speeds and decelerations are exactly the scenario's bounded values. Their lane localisation is refreshed every step, so IDM-based drivers and background traffic can see them.

**Episode outcome.** One of:

- success: the destination is reached;
- collision: with a vehicle, an object or a building;
- offroad: off every lane, onto the sidewalk, or across the yellow centre line. Solid white lines are counted in telemetry but don't end the episode;
- stuck: less than 1 m of progress in 10 s;
- timeout: the 150 s horizon is reached.

**Plausibility by construction.** Decelerations are capped at 6 m/s² (ordinary hard braking; emergency braking reaches about 8–9). Lateral speed during a cut-in stays within about 0.9–2.3 m/s, and a same-lane lead vehicle keeps at least a 7.5 m bumper gap before the trigger. Scenes that are still too hard are caught by the avoidability check: if no reference driver completes the scene, a SUT failure there never counts.
