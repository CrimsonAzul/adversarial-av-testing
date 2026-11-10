# Design: adversarial scenario generation against a qualified driving policy

Status: preregistered on 2026-10-31, before any driver training or adversarial run. Later changes go to the decision log in [LOG.md](LOG.md) with a date and a reason. A change made after looking at results is labelled as such.

## Thesis claim

A learned driving policy can pass a random-testing qualification inside its operational design domain (ODD) and still have reproducible failures that random testing rarely finds. An adversarial scenario generator trained with reinforcement learning finds these failures more efficiently than black-box optimisers (CMA-ES, Bayesian optimisation) and random sampling at an equal simulator budget. Every failure counted is a *valid* failure: inside the ODD, physically plausible by construction, and avoidable, meaning a reference driver completes the same scenario.

The driver is the system under test (SUT), not the contribution. It has to be strong, so that failures are rare under random testing. Without that, any search finds failures and the comparison between adversaries says nothing.

## Research questions

- **RQ1, efficiency (primary).** At an equal number of SUT evaluations, does the RL adversary find more unique valid failures than CMA-ES, BO and random? Primary metric: unique valid failures per 1,000 evaluations. Secondary: area under the discovery curve, evaluations to the first failure, and wall-clock time including the adversary's own training and optimisation.
- **RQ2, validity.** What fraction of each method's proposed failures is invalid (not avoidable by any reference driver)? An adversary that wins RQ1 only by producing unavoidable crashes has not won.
- **RQ3, diversity.** How many distinct failure modes does each method find? Measured as clusters in scenario-parameter space plus failure type, and as coverage of the parameter grid.
- **RQ4, generalisation of the generator.** Frozen after training, does the RL adversary keep its advantage on map seeds it never saw, compared with random at an equal budget?

No direction is assumed. A negative or inconclusive answer is reported as such.

## Operational design domain and scenario space

A scenario is a vector θ ∈ [0,1]^d plus a map seed. A single decoder (`avtest/scenario.py`) turns it into a MetaDrive configuration. The same decoder is used for driver training (θ sampled uniformly), qualification (uniform θ, held-out seeds), random search (uniform θ) and every adversary (chosen θ). Uniform random search is therefore exactly the qualification distribution, and its failure rate is the rate a random-testing campaign would see.

Parameter ranges are bounded to physically plausible values, so no post-hoc plausibility filter is needed. Events are specified relative to the ego (OpenSCENARIO-style), so every driver meets the same event geometry. The 12 parameters, their ranges and the outcome definitions are in [SCENARIO_SPACE.md](SCENARIO_SPACE.md).

The ODD has three nested tiers, declared now:

| Tier | Road blocks | Traffic density | Events |
|---|---|---|---|
| T1 | straight, curve | 0 – 0.10 | lead-vehicle braking, static obstacle |
| T2 | + on/off ramps, merges | 0 – 0.15 | + cut-in |
| T3 | + intersections, roundabouts | 0 – 0.20 | all |

The driver was first trained on T2 (decision 2026-11-01, after the reference ceiling check in LOG.md: the privileged reference drivers of the time together completed only 77.5% of random T3 scenes). From run d13 on it trains on all three tiers, each episode's tier drawn uniformly (the author requires a driver that is good on every tier; LOG.md, 2026-11-05). Qualification tests T3, then T2, then T1. The thesis uses the **widest tier that passes qualification**, chosen by the rule below before any adversary is run.

**Result (qualification attempt 3, 2026-11-10; [QUALIFICATION.md](QUALIFICATION.md)):** T3 failed (90.1% of 382 solvable scenes, 8.9% crashes) and T2 failed (96.7% of 455, 3.3% crashes). T1 passed: 99.4% of 496, Wilson lower bound 0.982, 0.4% crashes. **The ODD is T1**, and the adversary study runs on T1.

## SUT qualification gate

A held-out panel of 500 scenarios per tier, drawn uniformly from θ with map seeds disjoint from training and validation (`avtest/seeds.py`). The panel is generated and hashed before training ends.

- **Pass:** the SUT completes ≥ 98% of the panel scenarios that at least one reference driver completes, with a 95% Wilson lower bound ≥ 96%, and has a collision or off-road rate ≤ 2%.
- **Tier rule:** test T3, then T2, then T1; the first passing tier becomes the ODD. Freeze the driver checkpoint and its hash.
- **If no tier passes:** continue training (from the last checkpoint, more steps, same reward) and re-test on a *new* panel, since a failed panel has become development data. Re-tests are logged. If T1 still fails by 2026-11-14, narrow T1's ranges in a logged decision. The adversary study is not run on an unqualified driver.

## Driver (SUT)

The qualified SUT is **d17, imitation round 11** (`models/sut.zip`, sha256 `ec548f5347f8…`; `models/QUALIFIED.json`). It is an **end-to-end neural network**: its steering and throttle/brake go to the simulator unchanged, with no controller or rule acting on them (author's decision, LOG.md 2026-11-05: "fully AI network"). Its network and training are in `avtest/policy.py` (`EndToEndModel`, `fit_end_to_end`) and `scripts/pretrain_bc.py --action e2e`.

**Observation (813 values, each in [0, 1]):**
- MetaDrive's 720-beam lidar (100 m);
- a 24-beam road-edge detector (50 m) and a 24-beam lane-line detector (20 m);
- the 4 nearest vehicles' relative position and velocity;
- 10 navigation values;
- 6 ego values: heading to the lane, speed, steering, last applied steering and throttle, yaw rate;
- a 13-value **lane model** (`avtest/sim.lane_model`): the lane centre 0, 5, 10, 20 and 40 m ahead in the ego's frame, the lane width, and the lanes to each side. This is what a lane-detection camera reports, and it does not flicker as the lane-line beams do between dashes.

The network sees each observation together with its change over the last 0.2 s (`FrameHistory`), so 1,626 inputs, standardised with the first round's data.

**Network** (1,470,735 parameters):
- A SiLU trunk of 3 × 512 with two heads:
  - lane intent (left / keep / right). A side is chosen when P(left) − P(right) passes ±0.3.
  - throttle/brake as 9 levels (−1, −0.7, −0.35, −0.12, 0, 0.12, 0.35, 0.7, 1). The most likely level is used.
- Steering comes from a separate 2 × 256 network on the 77 ego-state and lane values (and their changes), not on the lidar. It gives one steering value per intent, and the chosen intent picks which is used, so a state in which the teacher sometimes changes lane is not averaged into steering between lanes.

**Teacher.** The labels come from a rule-based teacher that reads privileged simulator state: the map, the route, and every vehicle's exact position and velocity.
- **Planner:** `tactical.ScenePassingTeacher60`, MetaDrive's IDM at a 60 km/h cruise with curve slowing. It passes stopped objects, and vehicles far below its cruise speed, as soon as the next lane is clear.
- **Controller:** `tactical.YIELD_PD_CONTROLLER`:
  - PD steering along the route, with no integral term, whose sum the student could not observe;
  - full braking when the time to collision is below 2.5 s or the gap below 1.5 m;
  - full braking for a vehicle predicted to cross the ego's path within 3 s, and the escape forward when only accelerating avoids it;
  - lane-change guards for blocked lanes and vehicles alongside;
  - throttle smoothing 0.5.

Its labels are its final steering and throttle and the side it steered for. On the attempt-1 dev panels it completes 100 / 98.0 / 95.6% on T1 / T2 / T3.

Disclosure: the teacher is built on the same IDM as two of the reference drivers. Avoidability needs any one of three references, so this cannot make a failure look valid; it can make the SUT share some of the teacher's blind spots.

**Training: DAgger imitation** (LOG.md, 2026-11-05 to 2026-11-06):
- Round 0: the teacher drives 600 episodes. Rounds 1–12: 600 episodes each, the teacher driving a share 0.5^round of them and the student the rest. The teacher labels every state either way.
- In half of the teacher's episodes its steering gets correlated noise (DART, σ = 0.02), so the data shows its corrections. Episodes the teacher fails are discarded. Every 2nd step is kept, and the data of all rounds is aggregated (1.4M samples by round 12).
- Each round fits 20 epochs: Adam, learning rate 1e-3 falling to 5e-5 along a cosine, batch 512.
- Loss: cross-entropy on the intent; squared steering error in units of each intent's label spread; cross-entropy on the throttle level, with the hard-braking levels weighted 3.
- Every episode's tier is drawn from T1 / T2 / T3.
- After every round the student is validated on the fixed 200-scene validation panel (val split) as T1, T2 and T3. The round with the best mean is kept, never chosen on a qualification panel. Training took 11.6 h, resumable after any round.

**What was tried and not adopted** (LOG.md):
- **The earlier designs.**
  - d1–d7: Stable-Baselines3 PPO policies after an imitation warm start; from d4 on, their lane command and throttle went through the PID tracking controller.
  - d8–d13: an imitation-trained tactical planner (classified lane command and throttle) over the PID tracking controller, with an emergency-braking layer from d10, lane-change guards from d11 and predictive braking for side traffic from d13. d13 reached 100 / 97.6 / 93.1% on the dev panels, but with rules acting on the network's outputs.
- **More inputs, a learnable brake label, brake weighting:** d18 added an object list of the 10 nearest objects with headings and predicted paths; d19 imitated a teacher whose braking holds until 1 s past its thresholds. Neither improved T3 beyond 86.9–89.4%.
- **PPO fine-tuning** of the end-to-end network's throttle policy, at two step sizes (r1, r2; `avtest/finetune.py`, `scripts/finetune_rl.py`): 10.6M steps without a gain on T3.

**Environment reward** (used by the PPO experiments; the qualified driver is trained by imitation alone):
- 0.05 per metre of progress, scaled by a lane-centring factor;
- 0.02 per step at maximum speed;
- +20 for arriving;
- −30 for every failure (collision, off-road, stuck, timeout).

A crash therefore always outweighs the progress made before it (`tests/test_sim.py`).

**Environment version.** From 2026-11-10 (environment v3), MetaDrive destroys every object between episodes instead of recycling it. Recycled vehicles carried physics state into the next episode, so the same driver on the same scene could end differently depending on what the process had run before. Every episode now starts from the same simulator state (`tests/test_sim.py`).

## Reference drivers and validity

The reference drivers, from 2026-11-02 when the driver's cruise speed became 60 km/h, are:
- MetaDrive's IDM with lane changes at 60 km/h, slowing for curves (`drivers.RoadIDMDriver`);
- a cautious variant of it, with a larger gap and headway (`drivers.CautiousRoadIDMDriver`);
- MetaDrive's bundled PPO expert.

Qualification attempt 1 used the original 30 and 20 km/h IDM drivers with the same expert. Calibration of those on 48 random T2 scenes (`scripts/calibrate_references.py`; [LOG.md](LOG.md), 2026-10-31): at least one completed 97.9% of them.

A scenario is **avoidable** if any reference completes it (arrives at the destination without collision or leaving the road). A **valid failure** is a SUT failure (collision, off-road, or no progress for 10 s) in an avoidable scenario, reproduced on one replay with the same seed. In environment v3 every episode starts from the same simulator state, so a replay repeats the episode exactly. Unavoidable SUT failures are counted in RQ2 and never as discoveries.

## Adversaries (equal budget)

All methods search the same θ space through the same evaluator and are charged the same way. Budget unit: one *evaluation* = one SUT episode. Reference and replay episodes are logged and reported separately as total simulator episodes.

- **Random:** θ ~ U[0,1]^d.
- **CMA-ES:** `cma`, population 12, σ₀ = 0.3, restarts on stagnation.
- **BO:** Gaussian process (Matérn 5/2 + noise) with expected improvement, 20 random initial points, kernel hyperparameters re-optimised every 10 evaluations (fixed in between), the GP refit on at most 400 points before every proposal, batch proposals via kriging believer.
- **RL adversary (contribution):** PPO generator that proposes θ given a coverage-and-history observation. Reward: a bounded failure-severity score (valid failure = 1, otherwise near-miss shaping from minimum time-to-collision and route completion), a novelty bonus for unexplored parameter cells, and a penalty for unavoidable proposals. The same continuous severity score is the objective for CMA-ES and BO, so all methods optimise the same quantity.

Budgets: 1,000 evaluations per run, 10 seeds per method (101, 202, …, 909, 1010). The plan was 3 seeds (101, 202, 303), sized for overnight sessions, to be revised only from measured timing before any comparison run. Revised on 2026-11-10 (author's decision; [LOG.md](LOG.md)): on T1 with the frozen SUT, one seed of the four methods takes about 70 minutes (BO's own computation about 20 of them), and the SUT's T1 failures are rare, so 10 seeds fit the schedule and narrow the intervals. Budget per run unchanged.

## Statistics

Per seed: the discovery curve (unique valid failures vs. evaluations). Report mean and 95% bootstrap intervals over seeds for differences against random and between adversaries. Individual seeds are always shown alongside the intervals. The primary contrast is RL vs. the best baseline on RQ1. Uniqueness and clustering rules are fixed in `avtest/analysis.py` before results exist.

## Integrity rules

- Qualification and RQ4 panels are hashed when created; any later change is detected.
- The frozen SUT is checked against its recorded hash and controller before every search run (`drivers.qualified_sut`).
- Every episode starts from the same simulator state (objects destroyed between episodes, environment v3), so an episode's outcome does not depend on what ran before it in the same process.
- Every episode is a JSON record (scenario, driver, outcome, metrics, timing). All tables and figures regenerate from these records with one command.
- Long jobs are resumable, with a PAUSE file (stop at the next safe point) and a STOP file (stop now).
- Simulator results do not establish real-world safety.
