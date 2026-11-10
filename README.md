# Adversarial scenario generation for a qualified driving policy

Capstone project. A driving policy, an end-to-end neural network whose steering and throttle/brake drive the car with no hand-written rules in between, is trained in [MetaDrive](https://github.com/metadriverse/metadrive) by imitating a privileged rule-based teacher (DAgger). It then has to pass a strict random-testing qualification inside its operational design domain (ODD). The driver d17 qualified on tier T1 (99.4% of 496 solvable held-out scenes, Wilson lower bound 0.982, 0.4% crashes) and is frozen. Four search methods then hunt for its remaining failures at equal simulator budgets: a reinforcement-learning scenario generator (the contribution), CMA-ES, Bayesian optimisation and random search. Only *valid* failures count: inside the ODD, physically plausible by construction, avoidable by an independent reference driver, and reproduced on replay.

- Design and preregistered rules: [docs/DESIGN.md](docs/DESIGN.md)
- Scenario parameters and ranges: [docs/SCENARIO_SPACE.md](docs/SCENARIO_SPACE.md)
- What was run, when and why: [docs/LOG.md](docs/LOG.md)
- Qualification attempts: docs/QUALIFICATION.md (generated)
- Results: reports/RESULTS.md (generated)

## Pipeline

```
1  driver: end-to-end network trained by DAgger imitation of a privileged teacher, 13 rounds (scripts.pretrain_bc --action e2e)
2  qualification gate: ≥98% on 500 held-out scenes per tier, T3 → T2 → T1, first pass frozen
   (attempt 3, 2026-11-10: T1 passed with d17; models/QUALIFIED.json)
3  search runs: rl / cmaes / bo / random × 10 seeds (101, 202, …, 1010), 1,000 evaluations each, on T1
4  RQ4: frozen RL generator vs random on held-out map seeds
5  scripts.report: tables and figures rebuilt from the logs
```

PPO training (`scripts.train_driver`, the early drivers) and PPO fine-tuning of the end-to-end network (`scripts.finetune_rl`, runs r1–r2) were tried; the qualified driver is trained by imitation alone ([docs/LOG.md](docs/LOG.md)).

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Run commands are listed in [docs/RUNBOOK.md](docs/RUNBOOK.md). Every long job resumes when run again with the same command. To stop one, create `PAUSE` in its output folder (it stops at the next safe point) or `STOP` (it stops immediately).

## Layout

```
avtest/scenario.py     θ ∈ [0,1]^12 → MetaDrive scenario (roads, lanes, traffic, one scripted event)
avtest/sim.py          MetaDrive environment (v3: objects destroyed between episodes), the event vehicle/obstacle,
                       lane model, object list and episode telemetry
avtest/policy.py       the driver's network (EndToEndModel), its frame history and the imitation fit
avtest/drivers.py      SUT (the frozen model, end-to-end or with its recorded controller) and the reference drivers
                       (60 km/h IDM, cautious IDM, MetaDrive expert)
avtest/tactical.py     the teacher (60 km/h scene-passing IDM) and its controller: PD steering, emergency and
                       predictive braking, lane-change guards; also the earlier drivers' tactical interface
avtest/finetune.py     PPO fine-tuning of the end-to-end network (tried, not adopted)
avtest/evaluate.py     parallel evaluator: SUT episode, references, replay → one record per scenario
avtest/severity.py     the shared objective every adversary maximises
avtest/adversaries/    random, CMA-ES, Bayesian optimisation, RL (PPO scenario generator)
avtest/search.py       resumable search run with an exact evaluation budget
avtest/stats.py        discovery curves, unique failure cells, bootstrap and Wilson intervals
avtest/recording.py    top-down GIF + keyframe recordings with outcome overlays
avtest/progress.py     driver validation panel, showcase recordings and progress charts
scripts/               pretrain_bc, qualify, run_search, rq4, report, evaluate_controller, watch_driver,
                       inspect_scenario, calibrate_references; train_driver and finetune_rl (PPO, not adopted)
```

Simulator outcomes do not establish real-world safety.
