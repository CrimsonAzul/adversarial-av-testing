# Adversarial scenario generation for a qualified driving policy

Capstone project. A PPO driver is trained in [MetaDrive](https://github.com/metadriverse/metadrive) until it passes a strict random-testing qualification inside its operational design domain (ODD). Four search methods then hunt for its remaining failures at equal simulator budgets: a reinforcement-learning scenario generator (the contribution), CMA-ES, Bayesian optimisation and random search. Only *valid* failures count: inside the ODD, physically plausible by construction, avoidable by an independent reference driver, and reproduced on replay.

- Design and preregistered rules: [docs/DESIGN.md](docs/DESIGN.md)
- Scenario parameters and ranges: [docs/SCENARIO_SPACE.md](docs/SCENARIO_SPACE.md)
- What was run, when and why: [docs/LOG.md](docs/LOG.md)
- Qualification attempts: docs/QUALIFICATION.md (generated)
- Results: reports/RESULTS.md (generated)

## Pipeline

```
1  imitation warm start (DAgger from cautious IDM)
2  PPO training, overnight sessions
3  qualification gate: ≥98% on 500 held-out scenes, widest passing ODD tier, frozen SUT
4  search runs: rl / cmaes / bo / random × seeds 101, 202, 303, 1,000 evaluations each
5  RQ4: frozen RL generator vs random on held-out map seeds
6  scripts.report: tables and figures rebuilt from the logs
```

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
avtest/sim.py          MetaDrive environment with the event vehicle/obstacle and episode telemetry
avtest/drivers.py      SUT (PPO checkpoint) and the reference drivers (IDM, cautious IDM, MetaDrive expert)
avtest/evaluate.py     parallel evaluator: SUT episode, references, replay → one record per scenario
avtest/severity.py     the shared objective every adversary maximises
avtest/adversaries/    random, CMA-ES, Bayesian optimisation, RL (PPO scenario generator)
avtest/search.py       resumable search run with an exact evaluation budget
avtest/stats.py        discovery curves, unique failure cells, bootstrap and Wilson intervals
avtest/recording.py    top-down GIF + keyframe recordings with outcome overlays
avtest/progress.py     driver validation panel, showcase recordings and progress charts
scripts/               pretrain_bc, train_driver, qualify, run_search, rq4, report, inspect_scenario, calibrate_references
```

Simulator outcomes do not establish real-world safety.
