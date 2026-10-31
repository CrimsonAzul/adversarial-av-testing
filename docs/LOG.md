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
