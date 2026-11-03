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

## 2026-11-01 — daytime pilot before any further overnight run (criteria declared before running)

The author asked for every component to be double or triple checked before another overnight run. Checks already done:
- Both training scripts re-read line by line. Validation now takes the training tier explicitly and refuses a mismatch.
- 25 tests pass.
- The teacher was checked through the exact imitation path (clipped labels, T2 training seeds): 40/42 = 95.2% success, labels plausible (steering std 0.12, braking in 5.5% of steps), observations within [0, 1].

Pilot (production settings, scaled down in time):
- `pretrain_bc --name d2 --pilot`: 3 rounds × 240 episodes, validation on the first 100 scenes of the fixed panel.
- Then `train_driver --name d2 --pilot`: production configuration and curriculum, 100k critic warm-up, 2 × 300k PPO chunks, the same validation.

Pass criteria, declared now:
- **C1, DAgger works:** the best validation of rounds 1–2 is higher than round 0's.
- **C2, lane keeping works:** off-road in ≤ 10% of validation scenes in the best round.
- **C3, on track:** best round ≥ 60% validation success on T2 and ≥ 75% on its T1-like subset. Basis: d1, lidar only on T3, peaked at 52% after 5 rounds. A driver that cannot pass 60% after 3 rounds is unlikely to approach the gate overnight.
- **C4, PPO does not collapse:** validation after each PPO chunk is no more than 5 points below the warm start's.
- **C5, no reckless shift:** mean completion time on successful validation scenes stays ≥ 80% of the warm start's (d1's dropped from about 66 s to about 30 s).

Rule: the overnight command is given only if C1–C5 pass. Otherwise the cause is diagnosed and fixed, and the pilot is re-run, during the day.

## 2026-11-01 — pilot 1 result: FAILED (C2, C3); diagnosis

`pretrain_bc --name d2 --pilot` (log: `runs/driver/d2_pilot/bc_log.jsonl`). Validation success by round: 10% → 30% → 31%. Best round: off-road 40/100, collision 25, stuck 4. T1-like subset (7 scenes): 2, 2, then 4 of 7.
- C1 PASS (improves across rounds). **C2 FAIL** (off-road 40% > 10%). **C3 FAIL** (31% < 60%). PPO pilot not run. No overnight run.

Diagnosis (`scripts/diagnose_driver.py`, 30 validation scenes plus an empty-road variant of each; outputs in `runs/diagnose/`):
- Every off-road was the yellow-centre-line flag, with the car near its lane centre and drifting laterally at about 1 m/s. The corner crosses the line at an angle: the student weaves.
- On empty roads (no traffic, event out of reach) the round-2 student completed only 18/30. The teacher completed 30/30 on both the normal and the empty roads. The environment is fine; the student cannot hold its lane.
- Compared with the teacher, the student steered twice as hard (mean |steer| 0.079 vs 0.038), flipped steering direction nearly twice as often (1.0 vs 0.57 per s), and changed lanes on empty roads, where the teacher never does.
- **Wrong hypothesis, corrected.** I first thought the observation lacked lane-relative position. MetaDrive's default observation does contain the offset from the current lane centre; it lacks lane boundaries and any view of the road ahead. Adding a 24-beam lane-line detector and a 24-beam road-edge detector (observation 275 → 320) only moved round-0 empty-road success from 3/30 to 6/30. That is not the main cause. Kept anyway, as a realistic perception input.
- **Undertraining ruled out.** The same round-0 data fitted for 60 instead of 10 epochs changed held-out steering error only from 0.049 to 0.040, and added a steering bias of +0.021.
- **Conclusion.** An MLP copying the teacher's steering feedback loop from sensor features makes steering errors (about 0.04) as large as the corrections lane keeping needs (about ±0.02), and in closed loop these compound into weaving. DAgger corrects this only slowly.

Candidate fix under test (prototype, production code unchanged): **tactical interface** (`avtest/tactical.py`). The policy outputs a lane command (left/keep/right) and throttle/brake; steering comes from the lane-tracking PID controller that MetaDrive's IDM and `LaneChangePolicy` use, with the same gains. This is the learned-planner / classical-controller split of modular driving stacks. The learned part still makes every decision the adversarial events probe.
- Check: the teacher driving only through this interface, using its own labels, succeeds on 40/42 T2 training scenes, identical to the 40/42 it gets with full control. The labels capture its behaviour completely.
- Round-0 student with the interface: **0 off-road** on 30 validation and 30 empty-road scenes (vs 12 and 7 before), but mostly stuck. It under-predicts throttle when slow: about 0.0 at 7 km/h where the teacher applies 1.0. Those states are rare in teacher-driven data, and DAgger is the standard fix.
- Running: the 3-round DAgger pilot with the interface, same budget and same C1–C3 criteria as pilot 1 (`runs/driver/tactical_pilot/pilot_log.jsonl`).

## 2026-11-01 — pilot 2 (tactical interface prototype): imitation criteria PASS

Same budget as pilot 1: 3 rounds × 240 T2 episodes, teacher crashes dropped. The script is a prototype outside `scripts/`; production code is unchanged. Logs: `runs/driver/tactical_pilot/`.
- **My bug, caught before reporting.** The prototype's in-loop validation evaluated only 10 of the 100 scenes (wrong list chunking), so its 60% / 100% figures are void. Round 1 and 2 checkpoints were re-validated on all 100 scenes with an assertion that every scene runs once (`validation_100.jsonl`). The empty-road test (30 scenes) was unaffected.
- **Results (100 validation scenes, T2):**
  - Round 1: 83% (collision 13, stuck 4).
  - Round 2: **85%**. Collision 14, **off-road 1**. Lead-brake 36/37, obstacle 33/36, cut-in 16/27. T1-like subset 7/7. Mean completion time on successes 72 s (the teacher drives at 20 km/h).
  - Empty roads: 29/30 and 27/30. The old design's round-0 student managed 3/30, its round-2 student 18/30.
- **Criteria:** C1 PASS (0% → 83% → 85%), C2 PASS (off-road 1%), C3 PASS (85% ≥ 60%; T1-like 7/7 ≥ 75%, though only 7 scenes).
- The student is near its teacher (cautious IDM completes 88% of random T2 scenes). Cut-ins are the main remaining failure, and PPO has to improve beyond imitation there.
- **Not yet done:** adopting the interface in production code (needs the author's approval, because it changes the SUT's architecture), then the PPO pilot (C4, C5) through the production scripts.

## 2026-11-01 — design adopted; pilot 3 (production scripts): C1, C2, C5 pass; C3 (T1-like part) and C4 fail

The author approved the tactical design (learned lane command + throttle/brake, PID steering). It is now in `avtest/drivers.py` (SUTDriver), `avtest/sim.py` (TrainingEnv) and `scripts/pretrain_bc.py` (tactical labels, lane-change samples weighted 10×, 20 epochs).

New tests (28 pass):
- the controller holds the lane centre and a held "left" command changes lane;
- the teacher driving through its own tactical labels succeeds;
- failed teacher demonstrations are dropped, using a teacher that never moves;
- a reused environment gives identical results before and after other scenarios (no leak between episodes).

Two intermediate test failures were my test's fault: constant throttle reached 64 km/h on a curve, and a lane-change command was held for only one step. Both were traced, not assumed.

Pilot 3 (`pretrain_bc --name d2 --pilot`, then `train_driver --name d2 --pilot`; logs in `runs/driver/d2_pilot/`; pilot 1's folder kept as `d2_pilot1_endtoend`):

| Stage | Validation (100 T2 scenes) | Off-road | Collisions | T1-like (7 scenes) | Mean success time |
|---|---|---|---|---|---|
| Imitation round 0 | 3% (60 stuck) | 0 | 16 | 0/7 | 108 s |
| Imitation round 1 | 77% | 1 | 21 | 7/7 | 78 s |
| Imitation round 2 | 81% | 0 | 18 | 4/7 | 73 s |
| PPO 405k steps | 75% | 0 | 25 | 5/7 | 62 s |
| PPO 707k steps | 75% | 0 | 25 | 6/7 | 60 s |

- C1 PASS, C2 PASS. C3 PASS on T2 (81%) but **FAIL on the T1-like part** (4/7 < 75%). That subset is only 7 scenes, which I under-sized when declaring it; the criterion still stands as written.
- **C4 FAIL**: PPO is 6 points below the warm start, more than the 5 allowed. C5 PASS narrowly: success time is 83% of the warm start's, the limit being 80%.
- **Diagnosis of C4.** PPO trained at curriculum difficulty 0.30–0.32, with gentle events and light traffic. Its training success was 93–95% while validation, at full difficulty, fell. On easy scenes driving faster is safe and earns progress reward sooner, so it sped up (73 → 60 s) and collided more on hard scenes (18 → 25). The curriculum was designed for PPO from scratch; after a competent warm start it trains on the wrong distribution.
- No overnight run.

## 2026-11-01 — pilot 3b: PPO without curriculum after the warm start — C4, C5 pass

- **Fix.** After an imitation warm start, PPO trains at full difficulty from the first step (`scripts/train_driver.py`). From scratch it still uses the curriculum.
- **Re-run from the identical warm start** (imitation round 2, byte-identical file; `runs/driver/d2b_pilot/`):

| Checkpoint | Validation (100 T2 scenes) | Collisions | Training success | Mean success time | T1 (100 scenes) |
|---|---|---|---|---|---|
| Warm start (imitation round 2) | 81% | 18 | — | 73 s | 87% (Wilson 79–92%) |
| PPO 405k | **86%** | 14 | 85% | 66 s | **92%** (85–96%) |
| PPO 707k | **84%** | 15 | 89% | 65 s | **92%** (85–96%) |

- **C4 PASS** (+5 and +3 points, not below −5). **C5 PASS** (success time 89–90% of the warm start's; limit 80%). Training success rose across chunks. PPO diagnostics were stable: approx KL 0.002–0.004, clip fraction 0.11–0.13, explained variance 0.84–0.86.
- **C3's T1-like part remains a declared failure** on its 7-scene subset (imitation round 2: 4/7). The 100-scene T1 measurement made since (87% after imitation, 92% after PPO) is far above the 75% that part of the criterion aimed at. The author is told this explicitly with the overnight command; the criterion is not rewritten.
- **T1 failures after PPO:** 7–8 collisions, mostly broken-down obstacles (37/44 succeed; lead-brake 55/56). Still short of the 98% gate; overnight training is needed to see how far it goes.

Changes for the overnight run, decided now:
- Validation also reports T1 on 100 scenes (the same θ decoded as T1), since T1 is the tier most likely to qualify.
- Imitation rounds 8 → 5 (6 collections of 600 episodes, about 3 h). With the tactical interface, imitation reached 77% after one round, while PPO is adding the most per hour.

Checks before handing over: all tests, plus `pretrain_bc --check` and `train_driver --check` through the new T1 reporting path.

## 2026-11-01 — pre-overnight check: `train_driver --check` did not stop (my bug, fixed)

- **What happened.** The final check (all tests, then both training scripts in `--check` mode) ran for about two hours instead of a few minutes. Tests passed (28), and `pretrain_bc --check` finished normally. `train_driver --check` should stop at 6,144 steps but ran 82,944 steps (41 chunks) until the 40-minute timeout killed it. The killed process's validation workers kept the output pipe open, so the check never reported back.
- **Cause.** When I added `--pilot`, I read the stop condition (`stop_at = total_steps`) *before* `--check` shrank `total_steps` to 6,144. So `--check` ran to the session limit. The pilot set its own limit and the production run uses the full 40M target, so neither was affected; the check was.
- **Fix.** Mode handling moved into `run_settings()` in `scripts/train_driver.py`, which computes the stop condition after every override and refuses `--check` with `--pilot`. New test `test_each_mode_stops_at_its_own_step_count` checks output folder, step limit and panel for production, pilot and both check paths. Restoring the old ordering on purpose makes the test fail (40,000,000 ≠ 6,144), so it does detect this bug.
- **What the overlong run did show.** The new T1 reporting worked on every validation (`val_t1_success` in every row), and recordings and charts were written. Checks now log to files rather than pipes, so leftover workers cannot hide a result.
- **Re-check after the fix (18:34–18:42).** 29 tests pass. `pretrain_bc --check`: teacher round plus one student round, then stops. `train_driver --check`: critic warm-up to 1,024 steps, PPO chunks at 3,072, 5,120 and 7,168, then stops by itself and writes `done.json` (3.4 minutes). Every validation row includes T1 results, there are no errors in any log, and no processes are left. The 0–25% success rates are expected for check-sized training (4 episodes, 1 epoch); the check tests the pipeline, not learning.

## 2026-11-01 — checks of what only the full-size run exercises

`--check` runs the whole pipeline, but small. Before handing over the overnight command I checked what changes with size:
- **Full-size validation (200 T2 + 100 T1 scenes, recordings)**, run once on pilot 3b's best model: 200 and 100 scenes evaluated, T2 86.5% (173/200; 27 collisions, no off-road, stuck or timeout), T1 92% (identical to the pilot's, same 100 scenes, deterministic simulator), all 3 showcase recordings written, 5.6 minutes. By event on T2: lead brake 93%, obstacle 86%, cut-in 79%.
- **Memory:** pilot imitation data was about 50 MB per 240 episodes, so 6 × 600 episodes is about 750 MB (about 1.5 GB at peak while the rounds are concatenated); the laptop has 15.7 GB plus a 16 GB page file. Disk: 47 GB free; the pilots used 256 MB.
- **Hand-off:** both scripts use `runs/driver/d2`, which does not exist yet, so the run starts clean. In PowerShell, PPO starts only if imitation exits with code 0; tested with a clean exit, an exit with stderr output, and an uncaught error.
- **Laptop:** High performance plan, sleep and hibernate set to Never on AC, on AC power, no update restart pending. Risk found: Windows Update may restart between 2:00 and 8:00 (active hours 8:00–2:00). The author is asked to pause updates; the RUNBOOK now lists this.
- **Expected duration:** imitation about 2.5 h (about 12 min collection, up to 10 min fitting and 6 min validation per round); PPO `--hours 5` ends at the first 1M-step boundary after 5 h, so 5–6 h.

## 2026-11-01 — GitHub repository recreated

The repository was deleted and recreated with the same name and the same history (7 commits). Early commits had carried a co-author line naming the AI assistant; rewriting the history removed it from `main`, but GitHub kept the old commits and kept listing the assistant as a contributor. Code and history are unchanged. The federated-learning repository was recreated the same way.

## 2026-11-01/02 — run d2, night 2 (interim: paused at the author's request)

Started 19:52 with the command from RUNBOOK (imitation, then `train_driver --hours 5`). Paused at 00:41 with a `PAUSE` file; the trainer finished validating the 4.3M checkpoint, logged it and exited cleanly. The `PAUSE` file must be removed before resuming with `train_driver --name d2` alone: `pretrain_bc` refuses a folder that already has PPO training, so the chained command would stop at once.

Imitation (6 rounds × 600 episodes, 1 h 36 min in total; validation on the fixed panel, T2 200 scenes, T1 100):

| Round | Samples | T2 | T1 | Lane-change recall |
|---|---|---|---|---|
| 0 (teacher drives) | 194,750 | 67.5% | 76% | 0.82 |
| 1 | 390,463 | 82.0% | 83% | 0.85 |
| 2 | 585,583 | 85.0% | 97% | 0.91 |
| 3 | 775,706 | **91.5%** | 97% | 0.93 |
| 4 | 983,947 | 89.5% | 100% | 0.88 |
| 5 | 1,182,966 | 90.5% | 97% | 0.88 |

Round 3 had the best T2 validation and became PPO's starting point. The best round is chosen on T2, as declared.

PPO (started 21:29; about 50 min per 1M steps plus 5.6 min of validation):

| Steps | Training success | T2 | T1 | T1 failures |
|---|---|---|---|---|
| 301k (critic warm-up) | 88% | — | — | — |
| 1.30M | 89% | 87.5% | 97% | 3 collisions |
| 2.30M | 91% | **91.0%** (best, `model_best.zip`) | 99% | 1 stuck |
| 3.31M | 90% | 89.0% | 99% | 1 stuck |
| 4.31M | 90% | 89.0% | 93% | 5 collisions, 2 stuck |

- T2 has not improved on the warm start (91.5% → 87.5–91.0%). T1 reached 99% at 2.3M and 3.3M, then fell to 93% at 4.3M.
- These are validation results on 100 T1 scenes, not qualification. 99/100 has a Wilson 95% lower bound of about 94.5%, below the gate's 96%. Qualification uses 500 new held-out scenes.

## 2026-11-02 — run d2, PPO session 2: validation fell after 2.3M steps; stopped at 11.8M

Resumed at 01:34 with `train_driver --name d2 --hours 6` after removing `PAUSE`. Seven checkpoints were validated. The 11.3M checkpoint finished validating at 07:33, 5 h 59 min into the session, just inside the 6-hour budget, so a further chunk began (`--hours` is checked only at chunk boundaries). At 08:01 the author needed the laptop, so the run was stopped with a `STOP` file: `model_last.zip` was saved at 11,827,248 steps (510k into the unvalidated chunk) and the trainer exited cleanly. `STOP` was then removed.

| Steps | Training success | Mean return | T2 | T1 | T1 failures | Policy std |
|---|---|---|---|---|---|---|
| 2.30M (session 1, best) | 91% | 35.2 | **91.0%** | 99% | 1 stuck | 0.216 |
| 4.31M (session 1) | 90% | 34.5 | 89.0% | 93% | 5 collisions, 2 stuck | 0.209 |
| 5.31M | 89% | 33.6 | 87.5% | 95% | 4 collisions, 1 stuck | 0.207 |
| 6.31M | 89% | 33.8 | 81.5% | 91% | 7 collisions, 2 stuck | 0.204 |
| 7.31M | 87% | 32.5 | 80.5% | 90% | 8 collisions, 2 stuck | 0.202 |
| 8.31M | 88% | 33.2 | 84.0% | 91% | 7 collisions, 2 off-road | 0.201 |
| 9.31M | 86% | 32.1 | 76.5% | 74% | 24 collisions, 2 off-road | 0.197 |
| 10.32M | 85% | 31.0 | 76.5% | 77% | 21 collisions, 2 off-road | 0.194 |
| 11.32M | 85% | 31.7 | 80.5% | 82% | 17 collisions, 1 off-road | 0.192 |

- PPO is making the driver worse. After 2.3M steps, every measure declines together: T2 91% → 76–80%, T1 99% → 74–82%, training success 91% → 85%, and the mean training return 35.2 → 31.0. Because the return PPO maximises is also falling, this is not the policy exploiting the reward at the expense of success; the updates themselves are degrading it.
- The extra failures are mostly collisions. On T2 cut-ins, success fell from 77% to 68–69%.
- PPO diagnostics stayed within limits: approx KL 0.001–0.015 against `target_kl` 0.01, clip fraction 0.07–0.22, and explained variance 0.78–0.94. Meanwhile the action std fell steadily from 0.223 to 0.192, and the value loss rose from 0.64 to around 2.8–3.3.
- `model_best.zip` is still the 2.3M checkpoint (91.0% T2, 99% T1 on the validation panels), saved 2026-11-01 23:10 and unchanged.
- Continuing this PPO configuration is not worth another night. The next step needs a decision: qualify `model_best.zip` as it is, or first find out why PPO degrades the warm-started policy.

## 2026-11-02 — checks before qualification attempt 1

Decision (author): qualify `runs/driver/d2/model_best.zip`, the 2.3M PPO checkpoint (validation 91.0% T2, 99% T1), rather than train further first.

- **Model identity:** `model_best.zip` and `checkpoints/steps_0002304000.zip` have the same policy weights (compared tensor by tensor; the 3.3M checkpoint differs). The zip files themselves hash differently only because they were written at different times.
- **Same driver code as validation:** qualification and validation both evaluate through `avtest.evaluate.Evaluator` and `drivers.SUTDriver` (tactical policy, deterministic). Qualification also runs all three references on every scene and replays every SUT failure.
- **Gate test added** (`tests/test_core.py::test_qualification_gate_counts_only_solvable_scenes`):
  - 490/500 passes, 489/500 fails;
  - scenes no reference completes are excluded;
  - 49/50 fails on the Wilson bound;
  - no solvable scenes means no pass.
  Loosening the success threshold to 97% or the Wilson threshold to 0.85 makes the test fail. A first version missed the 97% change because its 489/500 case also broke the crash limit; that case now fails only on the success rule.
- **Resume trap fixed in `scripts/qualify.py`:**
  - The panel fingerprint included the model path as typed, so resuming an interrupted attempt with the path spelled differently (`runs/driver/...` vs `runs\driver\...`) would have been refused with "use a new --attempt", wasting the attempt.
  - The fingerprint now identifies the model by its SHA-256 only; the path is still written to `panel.json` for reference.
- **Dry run** (`--attempt 98 --panel 8 --workers 2`, a check panel that is not a qualification attempt; with 8 scenes no tier can pass, so nothing is frozen or appended): 248 s for all three tiers, no errors.

  | Tier | SUT success on solvable scenes | Unsolvable | Wall time per scene |
  |---|---|---|---|
  | T3 | 5/7 (3 collisions in all, 1 on an unsolvable scene) | 1 | 19.5 s |
  | T2 | 7/7 | 1 (SUT collided there) | 17.3 s |
  | T1 | 8/8 | 0 | 9.8 s |

  Rerunning the same command with backslashes in the path continued without rerunning any scene. Running the 3.3M checkpoint on the same attempt was refused ("holds a different panel or model"), so the guard still works. `runs/qualification/attempt98_*` and `attempt99_T2` are pipeline checks, not attempts.
- **Expected duration of attempt 1:** about 40 min (T3) + 35 min (T2) + 20 min (T1) with 4 workers, about 1.5–2 h in all.

## 2026-11-02 — qualification attempt 1: FAIL at every tier

`scripts.qualify --model runs\driver\d2\model_best.zip --attempt 1` (the 2.3M PPO checkpoint, sha256 `035452fb8908…`). It ran from 09:02 to 10:57 with no errors: T3 took 37 min, T2 42 min and T1 36 min. No tier passed, so nothing was frozen (`models/` was not created). Under the DESIGN.md rule, the attempt-1 panel is now development data, and the next test needs a new panel (`--attempt 2`).

| Tier | Solvable scenes | SUT success | Wilson low (gate 0.96) | Collision/off-road (gate 2%) | Failures |
|---|---|---|---|---|---|
| T3 | 417 | 80.3% | 0.763 | 19.2% | 66 collisions, 14 off-road, 2 stuck |
| T2 | 468 | 94.0% | 0.915 | 4.7% | 20 collisions, 2 off-road, 6 stuck |
| T1 | 498 | 95.6% | 0.934 | 4.2% | 21 collisions, 1 stuck |

**What fails on T1** (22 failures, 20 of them reproduced exactly on replay):

- **Static obstacles cause most failures:** 18 of the 22 (17 collisions, 1 stuck). The SUT succeeds on 92.4% of obstacle scenes and 98.5% of lead-brake scenes.
- **Nearly all failures are on curves.** On maps that start with a curve, the SUT succeeds 222/241 times (92.1%), against 254/257 (98.8%) on the other maps. 19 of the 22 failures are on curve-first maps, including all 4 lead-brake collisions. The SSS and SSC maps had no failures.
- **Obstacles in failures are further ahead:** a median of 81 m in failed scenes against 41 m in successes, so the obstacle is typically around the curve. In 11 of the 18 obstacle failures it is in the SUT's own lane.
- **References on T1 obstacle scenes:** IDM succeeds on 97.0%, cautious IDM (the imitation teacher) on 86.9%, the expert on 63.3%. The SUT, at 92.4%, beats its teacher but not IDM. On all solvable T1 scenes, IDM alone succeeds on 98.4%, so the T1 gate is reachable.
- **Validation was optimistic on T1.** It gave 99/100, but the Wilson interval for that result is [0.945, 0.998], which includes the qualification result of 95.6%. 100 validation scenes cannot tell 96% from 99%.

**Next:**
- Continue training and re-test on a new panel. If T1 still fails on 2026-11-14, narrow its ranges in a logged decision.
- PPO in its current configuration degrades the driver (see the earlier 2026-11-02 entry), so the next training step must address both the curve and obstacle weakness and that degradation.

## 2026-11-02 — diagnosis of the attempt-1 T1 failures: the driver gets out of the obstacle's lane too late

Data: the attempt-1 T1 panel, which is development data after the failed attempt. Tools: step-by-step traces of two failures, and `scripts/diagnose_obstacles.py` (added) on all 77 T1 scenes where the obstacle stands in the ego's lane. The diagnosis run, `--model runs\driver\d2\model_best.zip --attempt 1 --tier T1`, took 229 s with 3 workers; its output is in `runs/diagnose/attempt1_T1_obstacles.jsonl`.

**Traces**

- **Row 102** (CCC, 2 lanes, obstacle 81 m ahead in the ego's lane): the SUT approaches at about 20 km/h. Its lane command stays between −0.3 and +0.1 until 13 m before the stopped vehicle and passes −0.5 only at 12.8 m. It hits the warning triangle, which stands 8 m before the vehicle, while only 0.65 m out of its lane (`crash_object`). IDM starts moving over about 36 m before the vehicle and passes.
- **Row 0** (CSS, 3 lanes): the command passes −0.5 at 13 m, then drops back inside ±0.5 twice during the lane change, which steers the car back toward the obstacle's lane. It hits the triangle.

**All 77 scenes**: distance to the stopped vehicle where each driver finally leaves the obstacle's lane, i.e. its centre moves more than 1 m from that lane's centre line:

| Driver | Straight (43): median | 10th pct | Curve (34): median | 10th pct | Success |
|---|---|---|---|---|---|
| SUT (d2, 2.3M) | 16.9 m | 12.9 m | **11.6 m** | 10.2 m | 64/77 |
| IDM | 32.6 m | 27.6 m | 32.9 m | 29.9 m | 75/77 |
| Cautious IDM (the imitation teacher) | 34.3 m | 30.6 m | 34.3 m | 32.1 m | 68/77 |

- **Every SUT failure is a late exit.** All 13 failed SUT runs left the lane only 9.9–13.6 m before the vehicle, so just 2–6 m before the triangle. 10 hit the triangle and 3 the vehicle.
- **Curves make it worse.** On curves the SUT leaves about 5 m later than on straights, which is the margin between scraping past and hitting the triangle. This explains why the failures concentrate on curved maps.
- **The command flickers.** In 75 of 77 scenes it drops back inside ±0.5 at least once while the car is still in the obstacle's lane and already more than 0.3 m off its centre. This happens in successful runs too, so it is general indecision, not something specific to failures.
- **The teacher is not the cause.** Cautious IDM leaves at a median of 34 m on straights and curves alike; its failures here are "stuck", not collisions.

**Cause (inferred).**

- Imitation fits the lane command with a weighted mean-squared error against labels of −1, 0 and +1 (`pretrain_bc.fit`), and the controller acts only when the command passes ±0.5.
- In similar states the teacher's labels differ. Its overtaking starts only when its speed has dropped more than 3 km/h below its 20 km/h target and an internal timer has passed 50 steps (MetaDrive `IDMPolicy.lane_change_policy`); the student cannot observe either. In the middle lane of a 3-lane road it may also go left or right.
- A least-squares fit predicts the average of such labels. That average stays inside ±0.5 until the obstacle is close and then hovers near the threshold, which gives both the late start and the flicker.
- Why curves make the start later is not established. A plausible reason: on a curve an obstacle in the ego's lane appears off to the side in the ego-centred lidar and vehicle list.

**Run-to-run variation.** Success counts on these 77 scenes differed between three runs: SUT 64–67, cautious IDM 68–76. The exit distances agreed within 0.5 m. Outcomes depend slightly on which worker process runs a scene and what it ran before; qualification's immediate replays reproduced 20 of 22 SUT failures. This is an open issue for the reproducibility checks in the search runs.

**Fix: not yet decided.** Two candidates:

1. **Change the controller only:** commit to a started lane change and use a lower decision threshold. This needs no retraining and is tested on this development panel.
2. **Retrain imitation with a categorical lane decision:** left, keep or right as a classification, so labels are not averaged.

Either way, re-qualify on a new panel (`--attempt 2`).

## 2026-11-02 — controller fix tested and rejected; the real cause is the teacher's overtaking rule

**Author's instruction (afternoon):** take charge of the project while the author is away; decide and proceed without asking. The decisions below are mine (Claude's), logged for review.

**Experiment.** Added two options to `TacticalController`:

- `commit`: once a lane change starts, keep steering for that lane until the ego is in it; only a command the other way cancels it.
- `threshold`: the lane-command magnitude that starts a change.

`scripts/evaluate_controller.py` drives the d2 2.3M policy with each setting on the 500 attempt-1 T1 scenes (development data). It counts success over the 498 scenes a reference completed, as the gate does. Two settings were run twice, to measure run-to-run noise. Total: 3,500 episodes, 62 min with 4 workers.

| Setting (threshold, commit) | Success | Collision/off-road | Lead-brake | Obstacle | Lane changes per episode |
|---|---|---|---|---|---|
| 0.5, off (current) | 95.4% / 95.6% (replicate) | 4.6% / 4.2% | 256 / 257 of 261 | 219 / 219 of 237 | 0.69 |
| 0.5, on | 90.4% | 9.2% | 231 | 219 | 1.70 |
| 0.35, on | 87.8% | 12.0% | 226 | 211 | 2.08 |
| 0.25, on | 86.3% / 85.9% (replicate) | 13.3% / 13.9% | 230 / 231 | 200 / 197 | 2.56 |
| 0.15, on | 68.5% | 31.5% | 179 | 162 | 4.92 |

- Replicates agree within 2 scenes, so the differences are real. Every change to the controller makes the driver worse.
- The policy's lane command has many brief spikes past the threshold. Without commitment they are harmless, because the car steers straight back. With commitment each spike becomes a full lane change, often into traffic: lead-brake success falls by about 25 scenes.
- **Decision:** the SUT keeps the original controller (`SUT_CONTROLLER = {"threshold": 0.5, "commit": False}`). The options remain in the code, defaulting to off, so the experiment can be reproduced.

**Cause of the late lane changes.** I replayed the d2 driver on failed scene 102 with the cautious-IDM teacher labelling every visited state, exactly as the imitation step does (DAgger).

- The teacher calls for full braking from 37 m, when the warning triangle enters its 30 m view.
- It labels "change lane" only once the student's speed has dropped below 17 km/h, at 25 m.
- Its label then flips between "change" and "keep" as the student's speed crosses 17 km/h.

MetaDrive's IDM overtakes only when its own speed differs from its 20 km/h target by more than 3 km/h, and only after an internal timer has passed 50 steps (`IDMPolicy.lane_change_policy`). When the teacher drives, it brakes early and so changes lanes at about 34 m. In DAgger, the student's speed decides the label. The student therefore learns a lane change that depends on its own slowing down, and that flips at a speed boundary. This matches the late (median 15.5 m) and flickering lane changes found in the diagnosis.

**Next.** Give the teacher a scene-based overtaking rule: pass a stopped or near-stopped object in its lane as soon as it is within the teacher's 30 m view and the adjacent lane is safe, whatever its own speed. Check this teacher on the development panels, then retrain imitation (run d3).

**Also in this commit:**

- **Controller settings are part of the SUT's identity.** `avtest.tactical.SUT_CONTROLLER` is used by `SUTDriver`. It is included in the qualification panel fingerprint and recorded in `QUALIFIED.json`. `avtest.drivers.qualified_sut` refuses to start the search runs or RQ4 if `models/sut.zip` or the controller settings differ from what qualified. Test added.
- **`JsonlLog` fix.** A hard kill can leave a torn last line. A resumed run appended after it, which made every later row unreadable; qualification resumes through the same class. `append` now drops a torn tail first. The new test fails without the fix.
- **`evaluate_controller` resumes.** It appends each result as it arrives; checked with a simulated torn write, after which each of the 6 results was present exactly once.
- Full test suite: 33 passed.

## 2026-11-02 — a scene-based teacher, and a simulator leak between episodes

**New imitation teacher** (`avtest.tactical.ScenePassingTeacher`, now the default of `scripts.pretrain_bc --teacher`). It is cautious IDM with two changes:

1. It passes a stopped or near-stopped object (below 5 km/h) in its lane as soon as the object is within its 30 m view and an adjacent lane is clear (front and back gaps over 15 m, no blocked object ahead), whatever its own speed.
2. On entering a new road it follows the lane that continues its current one. MetaDrive takes the first connected lane (lane 0) and corrects a step later, which produced a one-step "change lane" label at every road boundary.

Run d2's teacher stays available as `--teacher cautious_idm`.

**Simulator leak found and fixed** (`avtest/sim.py`).

- MetaDrive recycles objects between episodes and keeps their velocity. A warning triangle hit in an earlier episode reported about 5.4 km/h in the next one.
- IDM-based drivers read object speeds: the references, background traffic and the imitation teacher. So their results depended on what the worker process had run before. This explains the run-to-run variation reported earlier (cautious IDM 68 vs 76 of 77 scenes).
- It also affected qualification attempt 1's reference runs, and therefore which scenes counted as solvable. Driven through its own labels with the fix in place, cautious IDM completes all 498 T1 scenes that attempt 1 counted as solvable; as a reference in attempt 1, under the leak, it completed 93.8% of them.
- The SUT's observation contains no object speeds, so its own results should not be directly affected.
- The new teacher first showed the leak: its rule fired only in the first scene of each worker.
- **Fix:** the triangle's velocity is set to zero when it is spawned. `test_recycled_obstacle_objects_start_at_rest` hits a triangle, starts a new episode and checks the recycled triangle is at rest; it fails without the fix.

**Teacher labels along the d2 driver's own drive** (`scripts/diagnose_labels.py`, 77 T1 own-lane obstacle scenes, fix in place):

| Teacher | First sustained "change lane" label | Scenes where the label flips back | One-step label blips per scene |
|---|---|---|---|
| cautious IDM (d2) | median 26.7 m (10th–90th pct 16.6–28.5 m); none within 50 m in 12 scenes | 47 | 3.5 |
| scene-passing (new) | median 37.7 m (37.5–38.0 m), in all 77 scenes | 4 | 0.1 |

**Teachers driving through their own tactical labels**, i.e. what a perfect student would achieve (`evaluate_controller --settings teacher:...`, attempt-1 dev panels; 2,000 episodes, 46 min):

| Teacher | T1 (498 solvable) | T2 (468 solvable) |
|---|---|---|
| cautious IDM | 100.0% | 97.0% (12 collisions, 2 stuck) |
| scene-passing | 100.0% | 97.2% (10 collisions, 3 stuck) |

- **T1 is reachable:** a student that learns the new labels well can pass the T1 gate.
- **T2 is not:** neither teacher reaches the 98% gate there.
- **Validation is unaffected:** each scene is evaluated once (no replays) and training does not use these panels.

**Also:**

- **Collector fix:** `pretrain_bc._collect` now closes its simulator if an error occurs. A failing collection test had left MetaDrive's one engine per process open and made 9 later tests error.
- **New test:** `test_scene_passing_teacher_moves_over_as_soon_as_it_sees_the_obstacle` checks a success and a first "change lane" label 30–40 m before the stopped car.
- Full test suite: 35 passed.

**Next:** run d3, which is imitation with the new teacher and otherwise d2's settings (6 rounds × 600 episodes, T2), about 1.6 h. Then evaluate on the T1 development panel and, if strong, qualify as attempt 2. PPO is skipped for now, because it degraded d2.

## 2026-11-02 (evening) — d3 shows the label fix works; the driver must now cruise at 60 km/h

**d3 stopped after round 2.** It was imitation with the scene-passing teacher at 20 km/h.

| d3 round | T2 validation | T1 validation | Time |
|---|---|---|---|
| 0 | 41% | 60% | 27 min |
| 1 | 77.5% | 85% | 24 min |
| 2 | 90.0% | 99% | 28 min |

- **Speed of the run:** rounds take about 24–28 min against d2's 15–18, because collection with the new teacher is slower.
- **On the 500 T1 dev scenes, round 2 reaches 97.4%** (485/498, Wilson low 0.956, 13 collisions), against d2's 95.5%. Obstacle scenes improve from 219 to 227 of 237.
- **Lane changes come earlier and no longer waver.** The driver leaves an obstacle's lane at a median of 26.4 m (curves 22.5 m), against d2's 15.5 m (11.6 m). It wavers in 4 of 77 scenes, against 75.
- **Why it was stopped:** the author decided the driver must cruise at at least 60 km/h. Rounds 3–5 at 20 km/h would not produce the final SUT, and the CPU was needed for the 60 km/h work. d3 resumes with the same command.

**Decision (author): cruise at 60 km/h, slowing for curves.** MetaDrive curves have radii of 25–60 m, with inner lanes down to about 18 m. At 60 km/h that would need up to 1.1 g of lateral acceleration, so the drivers slow to a 0.3 g advisory speed in curves (31 km/h at 25 m, 48 km/h at 60 m).

- **`avtest.drivers.RoadIDMPolicy`:** MetaDrive's IDM at 60 km/h.
  - The target speed is capped at sqrt(3 m/s² × radius) in curves, and the driver slows at 2 m/s² before each one.
  - It looks 80 m ahead, against MetaDrive's 30 m.
  - Its following gap is computed in SI units. MetaDrive multiplies km/h by seconds, which at 60 km/h demands 100–165 m behind any vehicle.
  - `CautiousRoadIDMPolicy` uses a 15 m standstill gap and a 2 s time gap.
- **References from now on:** `RoadIDMDriver` (`idm60`), `CautiousRoadIDMDriver` (`idm_cautious60`) and the expert. A scene counts as avoidable only if a driver at the SUT's own cruise speed completes it. The 20/30 km/h references remain available, to reproduce attempt 1.
- **Teacher `scene_passing_60`:** the scene-passing rules on top of the cautious 60 km/h IDM. It is now the default of `pretrain_bc`.
- **Sensors:** 720 lidar beams to 100 m, against 240 to 50 m. A 1 m warning triangle was reliably hit only within about 38 m, 1.4 s at 60 km/h; at 0.5° spacing it is hit out to about 100 m. The observation grows from 320 to 800 values. `sim.sensor_config` gives each driver the sensors matching its input size, so d1–d3 keep running.
- **Lane-change steering:** at 60 km/h the 60 km/h teacher, driving through its labels on the T1 dev scenes, completed only 96.0%: 14 off-road and 6 collisions. On T2 it completed 82.3%. Eleven of the 14 off-road failures happened during a lane change, 1.3–1.8 m off the lane centre.

  A test of a single lane change on an empty straight road (overshoot past the target lane's centre, right change / left change):

  | Steering | 30 km/h | 45 km/h | 60 km/h |
  |---|---|---|---|
  | Current (MetaDrive's lane-tracking PID) | 0.23 / 0.17 m | 0.00 / 0.36 m | off the road |
  | Gains scaled by 30 / speed | 0.00 / 0.16 m | 0.00 / 0.51 m | 0.82 m, then off the end of the route |
  | Lateral error capped at 1 m | 0.00 / 0.08 m | 0.00 / 0.16 m | 0.27 / 0.30 m |

  The controller of the 60 km/h generation (`SUT_CONTROLLER`), its training environments and the 60 km/h references now cap the lateral error at 1 m. d1–d3 keep `LEGACY_CONTROLLER`. Qualification refuses a model with the old sensors, because it records today's controller and sensors.

**Tests:** 39 pass. New ones cover the legacy sensors, cruising at 60 km/h within the curve limit, the 60 km/h teacher passing from about 80 m, and a 60 km/h lane change staying on the road.

**Next:** the capped 60 km/h teacher and the three 60 km/h references on the T1 and T2 dev panels (running). If the teacher completes about 98% or more of the scenes some 60 km/h reference completes, train d4 overnight.

## 2026-11-02 (night) — the 60 km/h teacher clears the gate on T1, but only just

- **Evaluation interrupted:** the first run of the capped 60 km/h teacher and references was killed when the Claude Code session restarted (background jobs die with it). Results already appended were kept: T1 complete, T2 partial. Long jobs now run in a terminal tab.
- **60 km/h references on the 500 T1 dev scenes** (leak fixed; mean speed includes starting from standstill, curves and following traffic):

  | Driver | Raw success | Mean speed |
  |---|---|---|
  | `idm60` | 489/500 | 40 km/h |
  | `idm_cautious60` | 495/500 | 39 km/h |
  | expert | 299/500 | 30 km/h |

  499 of the 500 scenes are avoidable at 60 km/h.
- **60 km/h teacher, through its own labels with capped steering:** 490/499 avoidable scenes, **98.2%** (Wilson low 0.966). Off-road fell from 14 to 2.
- **Passing restricted to static objects:** the teacher had passed lead cars that braked to a stop and lost 7 of 261 lead-brake scenes, against 2 for the cautious reference. It now passes only static objects, i.e. warning triangles (`ScenePassingTeacher60.STATIC_ONLY`); the 20 km/h teacher is unchanged.
- **Result after the change:** 490/498 attempt-1-solvable scenes, **98.4%** (Wilson low 0.969). Lead-brake 255/261, obstacle 235/237. No real change: the difference is within run-to-run noise.
- **What the remaining failures share:** all 6 lead-brake collisions happen during lane changes in traffic. At 60 km/h the drivers keep overtaking MetaDrive's background traffic, which drives at 30 km/h; both 60 km/h IDMs average about 1 lane change per scene.
- **Consequence:** the teacher sits right at the gate, and a student usually lands a little below its teacher. A realistic background traffic speed would remove most forced overtakes. That is the next lever if d4 falls short.

## 2026-11-03 (night) — d4, the first 60 km/h driver: 87% on T1, undone by lane changes

The author paused d4 after round 2 and resumed it at 01:31. Round 3 had been trained but not validated, so it is redone on resume. Before resuming, `model_last.zip` was reset to round 2's network (round 3's kept as `checkpoints/bc_round_03_unvalidated.zip`), so the redone round 3 collects with round 2's student, as an uninterrupted run would. The author had also overclocked the CPU. Under sustained full load it ran at about 3.5 GHz, 6% above base, so the effect on long runs is probably small.

| d4 round | T2 val | T1 val | T2 failures |
|---|---|---|---|
| 0 | 50% | 45% | 63 stuck, 33 collisions |
| 1 | 12% | 23% | 148 collisions, 28 off-road |
| 2 | 48% | 63% | 87 collisions |
| 3 | 47% | 71% | 101 collisions |
| **4 (best)** | **77%** | **89%** | 44 collisions |
| 5 | 72% | 75% | 49 collisions |

- **Imitation fits worse at 60 km/h:** the action error rose from 0.13 to 0.29 over the rounds. In round 0, 112 of 600 teacher demonstrations on T2 failed and were dropped, against 43 for d3.
- **On the 500 T1 dev scenes, round 4 completes 434/499 avoidable scenes (87.0%,** Wilson low 0.837). Lead-brake 232/261, obstacle 202/238.
- **62 of its 65 failures involve lane changes.** It changes lanes 4.4 times per scene, against its teacher's 1.1, mostly overtaking MetaDrive's 30 km/h background traffic. No qualification attempt.

**Two changes for d5:**

1. **Background traffic at 50–60 km/h** (`sim.ROAD_TRAFFIC`, `drivers.TrafficIDMPolicy`: RoadIDMPolicy at a seeded per-vehicle speed, 50 m look-ahead).
   - Traffic crawling at 30 km/h on a 60 km/h road forced about one overtake per scene. Every lead-brake collision of the teacher, and almost every d4 failure, happened in a lane change.
   - This changes the environment for every driver; results before 2026-11-03 used 30 km/h traffic. Result files from the new environment carry the tag `traffic50-60`.
   - Checked: traffic is deterministic, and on a curvy map traffic tops out at 45–55 km/h. Test added.
2. **`pretrain_bc --tier T1`:** train on the tier being qualified. The validation panel stays (200 T2 scenes); the T1 part grows from 100 to 200 scenes, and the best round is chosen on it.
   - At 60 km/h the teacher completes about 82% of T2 scenes (measured with uncapped steering), so T2 demonstrations are often dropped failures and its cut-in labels are poor.

**Tests:** 40 pass.

**Next:** the teacher and the references in the new traffic on the 500 T1 dev scenes (running), then d5 if the teacher improves.

## 2026-11-03 (night) — the 60 km/h teacher's overtaking, made scene-based; d5 on T1

All results below are on the 500 attempt-1 T1 scenes, in 50–60 km/h traffic (`traffic50-60` files). 499 of the scenes are avoidable at 60 km/h.

| 60 km/h teacher | Success (Wilson low) | Lead-brake | Obstacle | Lane changes per scene | Mean speed |
|---|---|---|---|---|---|
| passing static objects, plus MetaDrive's overtaking | 98.6% (0.971) | 257/260 | 235/239 | 1.22 | 42 km/h |
| passing static objects only, otherwise following | 79.4% (0.756) | 167/260 (57 stuck, 42 timeouts) | 229/239 | 0.14 | 27 km/h |
| **passing static, stopped (< 5 km/h) or much slower (< 40 km/h) objects, otherwise following** | **98.6% (0.971)** | 258/260 | 234/239 | 2.30 | 43 km/h |
| references: `idm_cautious60` / `idm60` / expert | 98.4% / 97.6% / 58.5% | | | 1.03 / 0.99 / 0 | |

- **Why d4 wove.** I traced d4 on its most-weaving scenes: it often changed lanes while the teacher said "keep". MetaDrive's IDM overtakes any slower car whenever its own speed is more than 3 km/h below target and an internal timer has run out. Under DAgger, "its own speed" is the student's. d4 drove slower than its teacher (36 km/h average), so the teacher kept labelling "overtake".
  - Along d4's own drive, that teacher gave 1.23 sustained "change lane" labels per scene. A teacher that only follows gave 0.17.
- **Following alone fails.** A lead car that brakes resumes at its trigger speed (ego speed −4 to +2 m/s), which is often 0 or slow when the ego is still accelerating, and a teacher that never passes waits behind it until it is stuck or times out. So the teacher must pass by the scene.
- **New rule (`ScenePassingTeacher60`, `FOLLOW_TRAFFIC`, `PASS_SLOWER_BY_KMH = 20`):** pass a static object, a stopped vehicle, or one at least 20 km/h slower than its 60 km/h cruise; follow all other traffic. The labels now depend on what the student can see (the other car's speed) and on no hidden timer.
- **Cost:** 2.3 lane changes per scene. Before its trigger, the event's lead car drives at the ego's speed, so while the ego is still accelerating it looks slow and gets passed; traffic starting from rest is passed too.
- **`STATIC_ONLY` dropped for the 60 km/h teacher** (passing stopped cars again): restricting passing to static objects had made no measurable difference.
- **d5:** `pretrain_bc --name d5 --tier T1`. It uses this teacher and 50–60 km/h traffic, with otherwise d4's settings (6 rounds × 600 episodes, lane-change weight 10). The best round is chosen on 200 T1 validation scenes.
- **Checks:** pipeline check passed (`tier: T1` recorded, selection on `val_t1_success`); 40 tests pass.

## 2026-11-03 (early morning) — d5, an evaluation bug, and a teacher that passes only when closing in

**d5** (`--tier T1`, the teacher passing anything under 40 km/h, 50–60 km/h traffic) reached its best T1 validation, 87.5% of 200 scenes, at round 5.

| d5 round | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| T1 val | 70.0% | 85.5% | 84.5% | 71.0% | 86.0% | **87.5%** |

**Bug in my evaluation script, fixed.**

- `evaluate_controller` built the SUT's controller from threshold and commit alone, which dropped the 60 km/h generation's 1 m lateral cap. d4 and d5 were first evaluated with steering that leaves the road at 60 km/h. d5 scored 73.3% that way, against 87.5% in training validation, which uses the driver's own controller; that disagreement exposed the bug.
- Now only threshold and commit are replaced (`controller_settings`), and a test was added. The wrong result files are kept with the suffix `.uncapped_bug`.
- d4's dev result (87.0%, 4.4 lane changes) came from the buggy script. My separate trace of d4's weaving used the correct controller, so that weaving is real.

**d5 with the correct controller:** 444/499 avoidable scenes (89.0%, Wilson low 0.859), 49 collisions, but 10.6 lane changes per scene, against its teacher's 2.3.

- Lead-brake scenes: its lane command passes ±0.5 on 43–92% of steps, and the car rides along lane lines.
- Obstacle scenes: 27%, median |command| 0.07.
- **Cause:** before its trigger, the lead car drives at the ego's speed, and 60% of lead-brake events trigger before 10 s, while the ego is still accelerating. "Pass anything under 40 km/h" therefore said "pass" while the ego was slow and "keep" once it was fast. To the student these are the same situation: a car ahead at its own speed.
- 16.5% of d5's training samples were lane-change labels.

**Network size is not the bottleneck** (`capacity_test.py` in the session scratchpad). Fitting on d5's rounds 0–4 and testing on round 5:

| Network | Lane-command error | Lane-label agreement | False lane changes |
|---|---|---|---|
| 256 × 256 | 0.307 | 78.3% | 17.2% |
| 512 × 512 | 0.296 | 79.8% | 16.2% |
| 512 × 512 × 512 | 0.279 | 77.3% | 18.8% |

**Teacher, closing condition added (`CLOSING_KMH = 10`):** pass a moving vehicle only if it is under 40 km/h *and* at least 10 km/h slower than the ego.

- The lead car before its trigger never qualifies, because it drives at the ego's own speed.
- A lead that brakes to a stop is passed by the stopped-vehicle rule during its 2 s hold.
- On the 500 T1 dev scenes: **489/499 (98.0%, Wilson low 0.964)**, 1.6% crashes, **1.08 lane changes per scene** (lead-brake 1.33, obstacle 0.81), 2 stuck.

**Next:** d6, the same as d5 with this teacher.

## 2026-11-03 (early morning) — d6; why the 60 km/h students weave; d7

**d6** (closing-speed teacher, T1, 50–60 km/h traffic, lane-change weight 10): best T1 validation 91.5% at round 4.

- **On the 500 T1 dev scenes:** 457/499 avoidable scenes, **91.6%** (Wilson low 0.888). Lead-brake 244/260, obstacle 213/239, 36 collisions.
- **Lane changes:** 5.75 per scene, against the teacher's 1.08. Every one of the 42 failures involves lane changes.

**The weaving, traced on two obstacle scenes.**

- d6 switches between two lanes every 0.5–1 s, often while its command is near 0 and the teacher says "keep". It rides the lane line.
- A lane command is relative to the lane the ego's centre is in, so while it straddles a line, "the lane it is in", and with it the steering target, flips with every wobble.
- DAgger then puts the student in these straddling states, where the teacher's corrections point both ways, and the student learns ±0.8 commands there.
- **Network size is not the cause** (the capacity test in the previous entry); the 1024 × 512 × 256 net also stays at 79.7% lane agreement.

**Tried: a controller that holds its own target lane** (`hold_lane`): "keep" means stay in the held lane, a change starts only on a fresh command, and the opposite command cancels it.

- Models now record the controller they were trained with inside their zip (`tactical.stamp_controller`, `controller_for_model`), so it travels with copies and is covered by the hash. Qualification records the model's own controller and its input size. PPO refuses a warm start trained with another controller, because that path is not wired up yet.
- **Teacher through its labels with `hold_lane`: 96.4%** (481/499), 3.0% crashes, against 98.0% with lane-relative commands. Its labels are relative to the current lane, so a second change it wants needs a fresh command it never gives; fixing that would need the student to see the controller's hidden target.
- **Not adopted.** It remains available as `--controller hold`; the default is `level`.

**d7:** d6's settings with the lane-change weight lowered from 10 to 3 (`--lane-change-weight 3`). The ±0.8 commands in straddling states come from a small number of lane-change samples weighted 10×.

**Tests:** 43 pass. New ones cover the `hold_lane` logic and the controller recorded in the model file.

## 2026-11-03 (morning) — d7, and where imitation at 60 km/h stands

**d7** (d6 with lane-change weight 3): best T1 validation 88.0% at round 4.

| On the 499 avoidable T1 dev scenes | d6 (weight 10) | d7 (weight 3) | 60 km/h teacher |
|---|---|---|---|
| Success | **91.6%** (Wilson low 0.888) | 90.2% (0.873) | 98.0% (0.964) |
| Failures | 36 collisions, 6 off-road | 36 collisions, 13 stuck | 6 collisions, 2 off-road, 2 stuck |
| Lead-brake / obstacle | 244/260, 213/239 | 229/260, 221/239 | 257/260, 232/239 |
| Lane changes per scene | 5.75 | **1.73** | 1.08 |

- **The lower weight removed most of the weaving and helped on obstacles, but the student became hesitant:**
  - in 13 lead-brake scenes it stays behind a stopped lead car until it counts as stuck;
  - 9 of 14 traced collisions are with the warning triangle, at 30–45 km/h, without braking and with a late lane change.
- **d6 was too eager and d7 too hesitant.** Both learn a yes/no lane decision as a least-squares fit to −1/0/+1 that only acts above ±0.5; the weight only moves where the averaged value lands.

**Where it stands.**

- No 60 km/h student has passed 92% on T1, against its teacher's 98%. At 20 km/h, d3 reached 97.4% after two rounds.
- Not the cause, each measured: network size (capacity test), traffic speed (d5), the training tier (d5–d7 on T1), overtaking labels that depend on the student's own speed (fixed in the teacher), and the lane-change weight (d6 vs d7).
- **Next proposal:** learn the lane decision as a classification (left/keep/right, cross-entropy, act on the most likely class), with throttle and brake regressed as now. This replaces the policy's lane output, so it needs its own network, about 2–3 h of work plus training.
- No qualification attempt was spent: every driver tonight was clearly below the gate on the development scenes.

## 2026-11-03 — "it vibrates a lot": what it is, and smoothing

The author watched d6 in the 3D viewer: it vibrates.

**Measured on 12 T1 scenes, d6 against its teacher:**

| | d6 | Teacher | Ratio |
|---|---|---|---|
| Steering change per step | 0.081 | 0.030 | 2.7× |
| Throttle change per step | 0.084 | 0.039 | 2.2× |
| Lane switches per second | 0.22 | 0.04 | 6× |
| Standard deviation of the lateral offset | 0.68 m | 0.27 m | 2.5× |

The vibration is mainly side-to-side wobble: the lane command crossing ±0.5 makes the steering target jump. Noisy throttle and brake add to it.

**Controller options added** (`TacticalController(smooth=, debounce=)`, off by default; `evaluate_controller` settings such as `0.5:off+smooth=0.5`, and it now also reports steering and throttle change per step). d6 on the 499 avoidable T1 dev scenes:

| Setting | Success | Lane changes per scene | Steering change per step | Throttle change per step |
|---|---|---|---|---|
| As trained (repeat; first run 91.6%) | 90.4% | 5.70 | 0.073 | 0.077 |
| Throttle smoothing 0.5 | 90.6% | 5.76 | 0.073 | **0.045** |
| Lane debounce 3 steps | 82.2% | 3.72 | 0.047 | 0.075 |
| Both | 82.4% | 3.64 | 0.047 | 0.045 |

- **Throttle smoothing: free.** 42% less change per step, at the same success. Adopted for viewing: `runs/driver/d6/model_best_bc_smooth.zip` is d6 with `smooth: 0.5` recorded in the file, so the viewer and evaluations use it.
- **Lane debounce: rejected.** It cuts the wobble by about a third, but collisions double, because lane decisions come too late.
- **Conclusion:** the wobble needs the classification-based lane decision; smoothing cannot remove it safely.

**Tests:** 44 pass.

## 2026-11-03 — the lane decision as a classification (d8)

**New student** (`avtest/policy.py`, `TacticalModel`), the default of `pretrain_bc --head classify`:

- a shared 256 × 256 Tanh trunk;
- a 3-way lane head (left/keep/right) trained with cross-entropy;
- a throttle/brake head trained with mean-squared error.

Its lane command is P(left) − P(right), acted on above the controller's threshold of 0.5, so a change needs one side to be the clear majority. The threshold can be tuned after training without retraining. The tactical action has the same form as before, so the controller, the evaluator and qualification are unchanged. `load_policy` reads either kind of model file. PPO refuses a `TacticalModel` warm start. The lane-change weight does not apply to this head.

**Offline check on d6's data** (rounds 0–4 for training, round 5 for testing, same network size and epochs):

| | Lane agreement | Recall of teacher lane-change steps | False lane changes per step | Throttle error |
|---|---|---|---|---|
| Weighted mean (d6's method) | 82.4% | 74.0% | 14.9% | 0.383 |
| Classification | **91.2%** | 39.6% | **2.5%** | lower (combined action error 0.199) |

- **Fewer false changes:** six times fewer, which is the wobble.
- **Risk:** it commits to a change only when confident, so it may start late.

**d8:** `pretrain_bc --name d8 --tier T1 --controller smooth`. It uses the classification head, the closing-speed teacher, 50–60 km/h traffic and throttle smoothing (`SMOOTH_CONTROLLER`), and is otherwise as d6/d7.

**Checks:** pipeline check passed (the model saves, loads and records its controller; round 1 drove with the classifier as the student); 45 tests pass.

## 2026-11-03 — d8, the classifier: as safe as d6, and smooth

**d8** (classification head, throttle smoothing, T1): best T1 validation 79.0% at round 5, still rising (72.5% at round 4). On its training data it matches the teacher's lane class 96% of the time, with 1.4% false lane changes.

**On the 499 avoidable T1 dev scenes:**

| Driver, setting | Success (Wilson low) | Failures | Lane changes per scene | Steering change per step | Throttle change per step |
|---|---|---|---|---|---|
| d6 smoothed (previous best) | 90.6% (0.877) | 39 collisions, 8 off-road | 5.76 | 0.073 | 0.045 |
| d8, threshold 0.5 (as trained) | 83.0% (0.794) | 80 collisions, 5 off-road | 0.84 | 0.035 | 0.041 |
| **d8, threshold 0.3** | **90.4% (0.875)** | 45 collisions, 3 off-road | **1.42** | **0.039** | 0.042 |
| d8, threshold 0.3 with commit | 88.8% (0.857) | 50 collisions, 5 off-road, 1 stuck | 1.97 | 0.028 | 0.044 |

- **The vibration is fixed.** At threshold 0.3, d8 matches d6's success with 75% fewer lane changes (the teacher makes 1.08) and about half the steering change.
- **As trained it is too hesitant:** at 0.5 it collides twice as often.
- **Tuning:** the threshold was tuned on the development panel, which is allowed. `tactical.copy_with_controller` writes `runs/driver/d8/model_best_bc_t03.zip`: the same network, recording threshold 0.3 (test added).
- **Still far from the 98% gate.** d8 was still improving at round 5, so more DAgger rounds are the cheapest next step.

**Tests:** 46 pass.

## 2026-11-03 — d8 extended to round 9

`pretrain_bc --name d8 --tier T1 --controller smooth --rounds 9`. A run may now add rounds (`same_run`: every other setting must match; test added). It resumed from round 5's network, which was also its best round. Round 5's `bc_done.json` and `model_best_bc.zip` are kept as `*_after_round5.json` and `model_best_bc_round5.zip`.

| d8 round | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| T1 validation (200 scenes, threshold 0.5) | 79.0% | 76.0% | **90.5%** | 81.0% | 90.0% |
| T1 collisions | 40 | 45 | 18 | 35 | 18 |

- **Best is round 7,** now `model_best_bc.zip`. At the trained threshold of 0.5, rounds 7 and 9 halve round 5's collisions.
- **Not yet done:** the dev-panel test at threshold 0.3, which lifted round 5 from 83.0% to 90.4%. The author will run it later.

## 2026-11-03 — d8 round 7 on the dev panel: no gain; the remaining collisions are missing brakes

**d8 round 7** (best on T1 validation, 90.5%) on the 499 avoidable T1 dev scenes:

| Setting | Success (Wilson low) | Crash rate | Lead-brake | Obstacle | Lane changes per scene |
|---|---|---|---|---|---|
| threshold 0.5 | 88.4% (0.853) | 11.6% | 239/260 | 202/239 | 1.03 |
| threshold 0.4 | 88.6% (0.855) | 11.4% | 234/260 | 208/239 | 1.34 |
| threshold 0.3 | 88.6% (0.855) | 11.4% | 240/260 | 202/239 | 1.81 |
| *round 5, threshold 0.3* | *90.4% (0.875)* | *9.6%* | *237/260* | *214/239* | *1.42* |

- **Selection noise.** Round 7 is not better than round 5. Choosing the best round on 200 validation scenes (±4 points) picks partly by chance; on the 500 dev scenes the extension brought no gain.
- **The plateau.** Five students in a row (d5–d8) landed between 88% and 91%, though each fix solved its targeted behaviour (weaving, smoothness).
- **The collisions.** Of 20 traced collisions, 16 repeated on replay, and 16 of those happened without braking: 9 with the warning triangle (7 without braking), 4 with a braking lead car and 3 with traffic.
  - Hard braking is rare in the teacher's data, so regressing throttle and brake as one averaged number pulls it toward zero, as happened with the lane decision.
  - Throttle smoothing may add some lag; d6's test showed no measurable cost.
- **Next candidate:** classify the longitudinal action as well, over a few levels (hard brake, brake, coast, accelerate).

## 2026-11-03 — throttle and brake as a classification too (d9)

- **Data:** the teacher's throttle labels in d8's data (820k samples) span the whole range. 6.7% are at or near full braking, and 24% are braking of any strength.
- **New `TacticalModel(throttle_bins=THROTTLE_BINS)`:** throttle/brake becomes the most likely of 9 levels (−1, −0.7, −0.35, −0.12, 0, 0.12, 0.35, 0.7, 1), trained with cross-entropy. It is the default of `pretrain_bc --throttle-head bins`; d8's regression remains as `--throttle-head regress`. Test added.

**Offline check on d8's data** (rounds 0–8 for training, round 9 for testing, 10.7% hard-brake labels):

| Throttle head | Hard-brake recall | Throttle error | Lane agreement | False lane changes |
|---|---|---|---|---|
| Regressed (d8) | **0.6%** | 0.349 | 90.7% | 4.3% |
| 9 levels (d9) | **42.4%** | 0.272 | 91.6% | 2.2% |

- **d8 almost never braked hard,** which matches its collisions without braking.
- **d9:** `pretrain_bc --name d9 --tier T1 --controller smooth`. It is d8's setup with the binned throttle head.
- **Tests:** 48 pass.

## 2026-11-03 — d9, the binned throttle: no gain on the road

The author paused d9 at 13:25 and resumed it at 13:45. It was stopped after round 3 was logged, while round 4 was collecting; `model_last.zip` matched round 3's checkpoint, so the resume was exact.

| d9 round | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| T1 validation (threshold 0.5) | 48.0% | 36.0% | 77.5% | 82.5% | 77.5% | **83.0%** |

**On the 499 avoidable T1 dev scenes:**

| Driver, setting | Success (Wilson low) | Failures | Lead-brake / obstacle | Lane changes per scene |
|---|---|---|---|---|
| d8 round 5, threshold 0.3 | 90.4% (0.875) | 45 collisions, 3 off-road | 237/260, 214/239 | 1.42 |
| d9, threshold 0.5 | 86.8% (0.835) | 62 collisions, 4 off-road | 232/260, 201/239 | 0.85 |
| d9, threshold 0.3 | 89.0% (0.859) | 54 collisions, 1 off-road | 241/260, 203/239 | 1.57 |

- **No gain on the road.** The 9-level throttle brakes hard far more often offline (42% against 0.6% recall), but collisions on the dev panel did not fall.
- **Six students in a row** (d5–d9, two output types, three throttle/lane variants) land at 87–91% at 60 km/h. The teacher reaches 98.0%, and at 20 km/h d3 reached 97.4% after two rounds.
- **The decision rule stated before d9 applies:** if it did not clearly break 93%, change approach.

## 2026-11-03 — an emergency-braking layer: crashes down to 1.4%

**`TacticalController(aeb_ttc=)`:** an independent emergency-braking layer, as in production cars.

- **What it watches:** every object the ego's lidar detects (vehicles and warning triangles) whose position, measured along the ego's current lane, overlaps the ego's path.
- **When it acts:** if the time to collision is under `aeb_ttc`, or anything is closer than 1.5 m in the path, it brakes fully.
- **What stays with the planner:** steering and lane choice.
- **First version, discarded:** it also braked half below twice `aeb_ttc`. Braking and the planner's throttle then alternated every step, and near standstill the car crept into the triangle.
- **Test:** driving straight at a warning triangle at full throttle, the car hits it without the layer and stops in front of it with it.

**On the 499 avoidable T1 dev scenes** (threshold 0.3):

| Driver + braking layer | Success (Wilson low) | Crash rate | Failures | Lead-brake | Obstacle |
|---|---|---|---|---|---|
| d8 round 5 alone | 90.4% (0.875) | 9.6% | 45 collisions, 3 off-road | 237/260 | 214/239 |
| d8 + 1.5 s | 88.6% (0.855) | 2.4% | 45 stuck, 6 collisions, 6 off-road | 239/260 | 203/239 |
| d8 + 2.5 s | 89.6% (0.866) | 1.4% | 45 stuck, 2 collisions, 5 off-road | 243/260 | 204/239 |
| d9 alone | 89.0% (0.859) | 11.0% | 54 collisions, 1 off-road | 241/260 | 203/239 |
| d9 + 1.5 s | 91.8% (0.890) | 2.8% | 27 stuck, 10 collisions, 4 off-road | 249/260 | 209/239 |
| **d9 + 2.5 s** | **94.4% (0.920)** | **1.4%** | 21 stuck, 5 collisions, 2 off-road | **256/260** | 215/239 |

- **d9 with braking at 2.5 s is the best 60 km/h driver so far.** Its crash rate is within the gate's 2% limit, and lead-brake scenes are nearly solved.
- **Remaining failures are "stuck":** after an emergency stop behind a static obstacle, the planner does not pull out around it. It was trained without the layer and never saw those states, because the teacher passes long before.
- **Next: d10**, trained with the layer in the DAgger loop. The student then reaches those stops and is labelled by the teacher, which passes a stopped or static object when the adjacent lane is clear.

**Tests:** 49 pass.
