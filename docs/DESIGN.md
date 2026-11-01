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

The driver is trained on T3. The thesis uses the **widest tier that passes qualification**, chosen by the rule below before any adversary is run.

## SUT qualification gate

A held-out panel of 500 scenarios per tier, drawn uniformly from θ with map seeds disjoint from training and validation (`avtest/seeds.py`). The panel is generated and hashed before training ends.

- **Pass:** the SUT completes ≥ 98% of the panel scenarios that at least one reference driver completes, with a 95% Wilson lower bound ≥ 96%, and has a collision or off-road rate ≤ 2%.
- **Tier rule:** test T3, then T2, then T1; the first passing tier becomes the ODD. Freeze the driver checkpoint and its hash.
- **If no tier passes:** continue training (from the last checkpoint, more steps, same reward) and re-test on a *new* panel, since a failed panel has become development data. Re-tests are logged. If T1 still fails by 2026-11-14, narrow T1's ranges in a logged decision. The adversary study is not run on an unqualified driver.

## Driver (SUT)

The SUT is an MLP policy (256×256) with continuous steering and throttle. Its observation (275 values) is MetaDrive's lidar-and-state observation (240 lidar beams, ego state, navigation) plus an object list of the 4 nearest vehicles, with their relative position and velocity. That is the object-level output a real perception stack gives a planner. Added after run d1 (LOG.md, 2026-11-01). It is trained in two stages:

1. **Imitation warm start (DAgger).** The cautious-IDM reference labels every visited state; the student drives from round 1 on (8 rounds × 600 episodes over T3, every 2nd step kept). The actor's mean action is regressed onto the teacher's actions (mean squared error). PPO starts from the round with the best validation, not the last one, with exploration noise σ = e^−1.5 ≈ 0.22. Reason: PPO from scratch on lidar typically reaches 60–80% success in MetaDrive with traffic, far below the gate, and the deadline allows about four nights of training. Disclosure: the teacher is also a reference driver. Avoidability needs any one of three references, so this cannot make a failure look valid; it can make the SUT share some of the teacher's blind spots.
2. **PPO fine-tuning** (Stable-Baselines3) with uniform θ over T3, train-split seeds, and a curriculum on event intensity, offset and relative speed during the first 30% of the step target. After the warm start, the first 300k steps train only the value network with the policy frozen, because updates driven by an untrained critic can undo imitation; afterwards small, conservative updates (learning rate 3e-5, 5 epochs, clip 0.1, `target_kl` 0.01) keep PPO refining the warm start instead of overwriting it. Run d1's larger steps (1e-4, 10 epochs, clip 0.2, 0.02) did not.

Reward: 0.05 per metre of progress, scaled by a lane-centring factor (about 15–20 for a whole route); 0.02 per step at maximum speed, so the driver does not creep into timeouts; +20 for arriving; −30 for every failure (collision, off-road, stuck, timeout). A crash therefore always outweighs the progress made before it. Run d1 used 1 point per metre against a 10-point crash penalty, so speed paid more than safety (LOG.md, 2026-11-01). `tests/test_sim.py` checks on a real scene that a safe success out-scores a reckless crash and a stuck driver. Checkpoints are selected on a fixed validation panel (val split, 200 scenes, every 1M steps and after every imitation round), never on the qualification panel. Training writes resumable checkpoints so it runs in overnight sessions. Progress is recorded on three fixed showcase scenarios (one per event type) at every validation, so the recordings show the same situations improving.

## Reference drivers and validity

The reference drivers are MetaDrive's IDM policy (with lane changes, 30 km/h), a cautious IDM variant (20 km/h, larger gap and headway), and MetaDrive's bundled PPO expert. Calibration on 48 random T2 scenes (`scripts/calibrate_references.py`; [LOG.md](LOG.md), 2026-10-31): at least one reference completed 97.9% of them. Individually, cautious IDM completed 95.8%, IDM 83.3% and the expert 52.1%. A scenario is **avoidable** if any reference completes it (arrives at the destination without collision or leaving the road). A **valid failure** is a SUT failure (collision, off-road, or no progress for 10 s) in an avoidable scenario, reproduced on one replay with the same seed. Unavoidable SUT failures are counted in RQ2 and never as discoveries.

## Adversaries (equal budget)

All methods search the same θ space through the same evaluator and are charged the same way. Budget unit: one *evaluation* = one SUT episode. Reference and replay episodes are logged and reported separately as total simulator episodes.

- **Random:** θ ~ U[0,1]^d.
- **CMA-ES:** `cma`, population 12, σ₀ = 0.3, restarts on stagnation.
- **BO:** Gaussian process (Matérn 5/2 + noise) with expected improvement, 20 random initial points, kernel hyperparameters re-optimised every 10 evaluations (fixed in between), the GP refit on at most 400 points before every proposal, batch proposals via kriging believer.
- **RL adversary (contribution):** PPO generator that proposes θ given a coverage-and-history observation. Reward: a bounded failure-severity score (valid failure = 1, otherwise near-miss shaping from minimum time-to-collision and route completion), a novelty bonus for unexplored parameter cells, and a penalty for unavoidable proposals. The same continuous severity score is the objective for CMA-ES and BO, so all methods optimise the same quantity.

Budgets: 1,000 evaluations per run, 3 seeds per method (101, 202, 303). This is sized for overnight sessions before the end-of-November deadline and is revised only from measured timing, before any comparison run.

## Statistics

Per seed: the discovery curve (unique valid failures vs. evaluations). Report mean and 95% bootstrap intervals over seeds for differences against random and between adversaries. With 3 seeds the intervals are wide, so individual seeds are always shown. The primary contrast is RL vs. the best baseline on RQ1. Uniqueness and clustering rules are fixed in `avtest/analysis.py` before results exist.

## Integrity rules

- Qualification and RQ4 panels are hashed when created; any later change is detected.
- Every episode is a JSON record (scenario, driver, outcome, metrics, timing). All tables and figures regenerate from these records with one command.
- Long jobs are resumable, with a PAUSE file (stop at the next safe point) and a STOP file (stop now).
- Simulator results do not establish real-world safety.
