# Project log

Every step that changes the study: date, what was done, exact command, outcome, decision and reason. Newest entries at the bottom. Results are never copied in by hand; they point to generated reports.

## 2026-10-31 — project start

- New project, started from an empty folder. Requirements from the author: the SUT is the author's own trained driver, and it must be strong on random MetaDrive worlds, since a weak driver makes adversarial search meaningless. The adversary is the contribution. Comparisons are RL vs CMA-ES vs Bayesian optimisation vs random. Final results are due by the end of November 2026. Training and long runs happen overnight, launched by the author.
- Decision: "perfect" is made testable as a preregistered qualification gate inside a declared ODD: ≥ 98% success on 500 held-out random in-ODD scenarios, with a 95% Wilson lower bound ≥ 96%, at the widest of three nested tiers that passes. Reason: no learned policy is perfect on every random world. An ODD plus a random-testing gate is how AV validation states competence, and it makes "rare failures that random testing misses" the adversary's target.
- Decision: plausibility is enforced by construction, through bounded parameter ranges, and validity by avoidability: at least one of three independent reference drivers completes the scene. Reason: a post-hoc plausibility filter needs its own calibration study, which this timeline cannot afford.
- Environment: Python 3.11 venv, CPU PyTorch. The driver and the generator are small MLPs, so the GPU is not needed. MetaDrive, Stable-Baselines3, pycma and scikit-learn. Exact versions are frozen in `requirements.lock.txt`.

## 2026-10-31 — simulator probe, scenario space and pipeline checks

Everything below was measured on the author's laptop (i7-11370H, 16 GB) while it was in normal use. No driver has been trained yet; every run so far is a pipeline check.

- Installed MetaDrive 0.4.3, Stable-Baselines3 2.9.2, CPU torch 2.15.0, cma 4.5.0 and scikit-learn 1.9.1 (`requirements.lock.txt`).
- Probe results:
  - Road layout, lane count and width can change on every reset, with `store_map=False` and an updated `map_config`.
  - IDM and the bundled expert can drive the ego from outside the environment.
  - Extra vehicles can be spawned and moved by a custom manager.
- Speed: about 80 env steps/s per process with traffic density 0.1, versus 281 without traffic. Lidar size makes no difference. The cost is MetaDrive's per-vehicle navigation and physics, so nothing tunable. Consequences: about 10M driver-training steps per 8-hour night on 6 processes; a 1,000-evaluation search run takes about 45–60 minutes on 4 workers; a 500-scene qualification tier, with all references, takes about 1 hour.
- **Decision: events are relative to the ego.** The first version gave event vehicles absolute speeds. A driver slower than the event vehicle never caught up, and the event fired at second 51, at the route's end. Slow references would have "avoided" every event by never meeting it. Now, until the trigger time, the event vehicle drives a set offset ahead at the ego's speed, then takes ego speed + offset (OpenSCENARIO-style). Offset ranges depend on the event type, keeping at least a 7.5 m bumper gap for a same-lane lead vehicle.
- **Decision: off-road no longer includes solid white lines.** MetaDrive ended IDM episodes as "off-road" in mid-lane, 5.7 m from either edge, when a lane change crossed a solid white line near a ramp. Off-road is now: off every lane, onto the sidewalk, or across the yellow centre line. White-line crossings are counted in telemetry.
- **Fix: event vehicles' lane localisation.** IDM finds the vehicle ahead through each object's lane, which the event vehicle never updated. IDM therefore drove into lead vehicles that started 44 m ahead. The event manager now refreshes the vehicle's localisation each step. Reference calibration on 48 random T2 scenes went from 75.0% union success to 97.9% (cautious IDM 95.8%, IDM 83.3%, expert 52.1%). Output: `runs/calibration/T2/summary.json`. Not committed; re-run with `python -m scripts.calibrate_references --tier T2 --scenes 48`.
- **Decision: driver training starts with a DAgger imitation warm start from cautious IDM, then PPO.** Reason: PPO from scratch on lidar is unlikely to reach the 98% gate in about 4 nights. The teacher-reference overlap is disclosed in DESIGN.md. It cannot make failures look valid, because avoidability needs any one of three references.
- Pipeline checks (tiny budgets, untrained check models; results meaningless by design):
  - `pytest`: 16 passed.
  - `scripts.train_driver --check`: 2 chunks with validation; 136–190 steps/s on 2 envs.
  - `scripts.pretrain_bc --check`: 2 rounds; the checkpoint loads into the real training env and PPO continues from it.
  - `scripts.run_search --check`: the RL generator and CMA-ES ran 24 evaluations each with references and replays.
  - `scripts.report`: regenerated tables and figure from the logs.
  - `scripts.qualify --panel 8 --attempt 99`: FAIL, as expected for an untrained model; nothing frozen.
- Published at the author's request: public repository https://github.com/CrimsonAzul/adversarial-av-testing (`main`). Run outputs under `runs/` are not committed. The qualified model and generated reports will be committed when they exist.

## 2026-10-31 — review of all training code, progress recordings

The author asked for every training component to be reviewed and verified, with regular progress recordings for driver training and adversary search. Every pipeline was run end to end on small budgets after the changes; the outputs are under `runs/` and are not committed.

Fixes:
- **Imitation warm start.** Data was kept at full precision on every step: about 2.2M × 259 floats, with peaks near 5 GB of RAM. It now keeps every 2nd step (consecutive 0.1 s frames are nearly identical), stores float16 shards per round on disk, and resumes at the last finished round.
- **Imitation objective.** The negative log-likelihood objective learned the noise level slowly and plateaued at an action error of about 0.10 on a synthetic test. It was replaced by mean-squared-error regression of the mean action, the standard behaviour cloning for a deterministic teacher. Held-out action error on the same test fell to 0.027 (`tests/test_training.py`).
- **PPO after the warm start.** It began with an untrained value network that can wreck the imitated policy. Added: a 300k-step critic-only warm-up with the actor frozen (tested: actor parameters unchanged, critic changed), and `target_kl = 0.02`.
- **PPO STOP.** STOP did not work mid-chunk. A callback now checks STOP every 200 steps, saves `model_last.zip` and exits.
- **Training scenario streams.** Every session restarted the same training scenario sequence. The seeds are now offset by the step count.
- **Validation cost.** Validation used 2 workers per 500k steps, about 13 minutes (≈1/3 overhead). Now: 4 workers every 1M steps.
- **Search timing.** Adversary compute time was never logged, and the report summed per-scene times as wall time. `timing.jsonl` now records simulator seconds and adversary seconds per batch; recording time is kept separate and never charged to a method.
- **BO speed.** BO re-optimised GP hyperparameters before every proposal: about 70–80 s per batch of 4 at 600 points, roughly 4–5 extra hours per run. That was also not what DESIGN.md specifies ("every 10 evaluations"). Now hyperparameters are re-optimised every 10 evaluations and the GP is refit with them fixed in between: 3.8 s per batch on average, and proposals still converge (measured at 600 points).
- **Simulator.** White-line crossings were counted twice per step, because MetaDrive calls the off-road check from both reward and termination. They are now counted once in `step`. Top-down frames were transposed (rotated); recordings are now ego-centred, heading up, at 6 px/m.
- **Report.** A formatting bug would have cut table rows short when timing was missing.

Findings:
- Bullet physics in MetaDrive is not bit-exact across resets. Outcome, step count and trigger step replay identically; minimum TTC differs around 1e-6 relative. The replay check compares outcomes, so it is unaffected (`tests/test_sim.py`).
- MetaDrive allows one engine per process. Production code respects this (one environment per worker), and the tests now keep simulator test modules separate.

New progress records:
- Driver: after every imitation round and every 1M PPO steps, validation on the fixed 200-scene panel (overall, per event type, with and without junctions, outcome counts) and GIF + keyframe recordings of the same 3 showcase scenarios. Outputs: `recordings/<checkpoint>/`, `bc_progress.png` / `progress.png`, `bc_log.jsonl` / `train_log.jsonl`.
- Search: a progress chart every 50 evaluations (discovery curve, severity, failure and invalid rates; RL learning curve), and the RL generator's per-update statistics in `rl/updates.jsonl`. Recordings follow a fixed policy: the first valid failure, then at most one new failure cell per 100 evaluations, each recorded as the SUT failing plus the reference driver that completed the same scene.

Checks: 23 tests pass, including real-simulator tests for replay determinism, event geometry, recording fidelity and the training environment. End-to-end checks ran `pretrain_bc --check` (with a resume), `train_driver --check` (warm-start path with critic warm-up), `run_search --check` (all four methods with recordings) and `report`.

## 2026-11-01 — first overnight training (run d1): far below the gate; reward design error found

Command (author, overnight): `scripts.pretrain_bc --name d1`, then `scripts.train_driver --name d1 --hours 6`. Logs: `runs/driver/d1/bc_log.jsonl`, `train_log.jsonl`, `bc_progress.png`, `progress.png` and `recordings/`. Not committed.

- **Imitation (DAgger, 6 rounds, 740k samples, ~1 h).** Validation success on the 200-scene panel by round: 0%, 29%, 7%, 23%, **52%**, 41.5%. Best is round 4: 71% on scenes without junctions, 49% with. Action error on the training data rose from 0.05 to 0.17 as DAgger added harder states.
- **PPO (7.3M steps, ~6 h).** PPO started from the last imitation round (41.5%), not the best. Validation fell to about 20% and stayed between 5.5% and 23.5% for 7M steps, with no upward trend. Failures were split between collision and off-road. Throughput fell from 559 to about 260 steps/s as the curriculum added traffic, so plan for about 7–8M steps per night, not 11M.
- **Diagnosis: reward design error (mine).** MetaDrive's driving reward is per metre of progress, so a full route is worth hundreds of reward points, while crash, off-road and stuck penalties were only 10. Reckless progress therefore paid more than caution. PPO learned to drive fast: mean time to complete a validation route fell from about 66 s after imitation to about 30 s after PPO, and success fell with it. Recordings show the policy reaching 40 km/h within 4.6 s and touching the centre line.
- **Contributing.** Every 1M-step chunk was evaluated, but PPO always continued from its latest weights; `model_best.zip` kept the best checkpoint (23.5% at 5.3M). The imitation student sees lidar only. The teacher's labels depend on lane-change intentions the student cannot observe, which makes them ambiguous, and imitation plateaued around 50%.
- **Ceiling check needed.** The calibration on 48 T2 scenes gave cautious IDM, the teacher, 95.8% raw, about 97.9% on reference-solvable scenes. That is about the 98% gate itself. Whether 98% is reachable for any driver on this distribution must be measured on a larger panel before the gate decision.
- No qualification attempt was made, and nothing is frozen.

## 2026-11-01 — fixes after run d1 (driver d2 from scratch)

Decisions made after seeing run d1's results. None of them touch the qualification gate, the scenario space or the adversary study, which have no results yet.

- **Reward rebalanced (fixes d1's error).**
  - Progress: 0.05 per metre with lane centring (a whole route is about 15–20).
  - Speed bonus: 0.02 per step at maximum speed.
  - Success: +20.
  - Every failure: −30, now including stuck and timeout, which MetaDrive did not penalise.

  Checked on 3 real lead-brake scenes: cautious IDM succeeds with return ≈ +45.7, full throttle crashes or leaves the road at −17 to −27, and a frozen car is stuck at −30. The same check is now a regression test (`tests/test_sim.py::test_reward_makes_safety_dominate`).
- **Driver input: 4 nearest vehicles added** (relative position and velocity; MetaDrive's `num_others = 4`). The observation goes from 259 to 275 values. Reason: d1's lidar-only student plateaued around 50% and mostly crashed into event vehicles. An object list is what a perception stack hands a real planner, and MetaDrive's own expert policy uses it. Verified that the event vehicle appears in it. This needs a new driver: d2 is trained from scratch, and d1 is kept for comparison.
- **Imitation:** 8 rounds instead of 5. PPO starts from the round with the best validation (d1 started from round 5 at 41.5% instead of round 4 at 52%). Initial PPO noise is σ ≈ 0.22 instead of 0.37.
- **PPO:** learning rate 3e-5 (was 1e-4), 5 epochs (10), clip 0.1 (0.2), target_kl 0.01 (0.02), so PPO refines the warm start instead of overwriting it.
- **Reference ceiling check started:** 200 validation scenes each for T3, T2 and T1, all three references on every scene (`scripts.calibrate_references --scenes 200`). It measures whether the 98% gate is reachable by any driver on this distribution. Results are logged when it finishes.
- Checks: 24 tests pass. `pretrain_bc --name d2 --check` and `train_driver --name d2 --check` ran end to end with the 275-value input and the best-round handover.

## 2026-11-01 — reference ceiling check (200 validation scenes per tier)

Command: `scripts.calibrate_references --tier {T3,T2,T1} --scenes 200` (all three references on every scene). Outputs: `runs/calibration/{T3,T2,T1}_200/summary.json`.

| Tier | Solved by at least one reference | IDM | Cautious IDM | Expert | Best single reference on solvable scenes |
|---|---|---|---|---|---|
| T3 | 77.5% | 48.5% | 63.0% | 46.5% | cautious IDM, 81.3% |
| T2 | 95.0% | 86.0% | 88.0% | 54.5% | cautious IDM, 92.6% |
| T1 | 100% | 99.0% | 96.0% | 62.5% | IDM, 99.0% |

Unsolved scenes by event: T3 cut-in 20/69, lead-brake 11/62, obstacle 14/69; T2 cut-in 9/69, lead-brake 1/62, obstacle 0/69.

Reading:
- On T3, even the privileged rule-based drivers fail often: junctions combined with events defeat them.
- Only on T1 does any driver reach the 98% gate on solvable scenes (IDM 99%).
- The imitation teacher, cautious IDM, completes 81.3% of solvable T3 scenes and 92.6% of solvable T2 scenes. A student imitating it is therefore capped below the gate on T3 and T2.
- Training on T3, as d1 did, imitates a teacher that fails on 37% of the training scenes. The demonstrations include its crashes.

No decision is taken in this entry.

## 2026-11-01 — decision: d2 trains on T2 and drops failed teacher demonstrations

The author chose this after seeing the ceiling check.
- **Training tier T3 → T2** for imitation, PPO and validation, including the showcase recordings. Reason: on T3 the privileged references fail on 22.5% of scenes, and the imitation teacher completes only 81.3% of the solvable ones. A student imitating it cannot approach the 98% gate, and d1's training data included the teacher's crashes on 37% of T3 scenes.
- **Unchanged:** the qualification rule (still T3, then T2, then T1, first pass wins), the gate, the scenario space and the adversary study. On current evidence T1 is the likely qualifying tier: IDM completes 99% of T1 scenes. T2's best reference completes 92.6% of solvable scenes, below the gate.
- **Episodes the teacher drove and failed are dropped** from the imitation data; student-driven DAgger episodes are kept. Tested by forcing a crashing teacher (`tests/test_collect.py`).
- **Validation** now also reports success on its T1-like subset (S/C roads, traffic ≤ 0.10, lead-brake or obstacle), the slice of the panel most comparable to the likely qualifying tier.
- Checks: 25 tests pass; `pretrain_bc --check` and `train_driver --check` ran on T2.
