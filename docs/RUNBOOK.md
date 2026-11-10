# Runbook: what to run, in order

Run each command from the project folder in PowerShell when the laptop is free (close heavy apps; every job uses all CPU cores). Every job resumes from where it stopped if you repeat the same command. To end a job early, create `PAUSE` in its run folder (it finishes the current chunk or batch, saves, and exits) or `STOP` (it exits immediately). After each step, add a line to [LOG.md](LOG.md), or ask Claude to read the generated files and log it.

Before an overnight run: charger in, lid open, sleep set to Never on AC power, and Windows Update paused (Settings → Windows Update → Pause updates). Windows may restart for updates outside active hours, which on this laptop are 8:00–2:00, so between 2:00 and 8:00, in the middle of a run. A restart loses at most the current chunk; the same command resumes.

Times are measured on this laptop (about 80 simulator steps/s per process with traffic).

## Schedule (deadline: end of November)

| When | Step | Status |
|---|---|---|
| Nov 1–9 | 1 driver training (runs d1–d19, r1–r2) | done: d17, imitation round 11 |
| Nov 10 | 2 qualification, attempt 3 | done: T3 and T2 failed, T1 passed; d17 frozen |
| From Nov 11 | 3 search runs, 40 runs (10 seeds × 4 methods) | about 12–20 h, several sessions |
| Then | 4 RQ4 + report | about 1.5–2 h |

## Where to watch progress

| Stage | Look at | Updated |
|---|---|---|
| Driver imitation | `runs/driver/<name>/bc_progress.png`, `recordings/bc_round_XX/*.gif` | every round |
| Search runs | `runs/search/<method>_s<seed>/progress.png`, `recordings/eval_XXXX_sut.gif` + the reference that avoided it | every 50 evaluations / each new kind of failure |
| RL generator learning | `runs/search/rl_s<seed>/rl/updates.jsonl` (also the 3rd panel of its progress.png) | every PPO update (24 evaluations) |

The three driver showcase GIFs (lead_brake, obstacle, cut_in) are the same scenarios at every checkpoint, so they show the same situations getting better.

## 0. Check the pipeline (about 5 minutes)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 1. Train the driver (done: d17)

The qualified driver was trained with:

```powershell
.\.venv\Scripts\python.exe -m scripts.pretrain_bc --name d17 --tier all --controller yield_pd --action e2e --rounds 12 --brake-weight 3
```

It runs DAgger imitation of the 60 km/h teacher: round 0 and 12 student rounds of 600 episodes, each episode's tier drawn from T1/T2/T3. Each round prints:
- the steering error;
- validation success on the 200-scene panel as T2, overall and per event;
- validation success on the same panel as T1 and as T3.

If it is interrupted, the same command resumes after the last finished round. At the end the round with the best mean validation is saved as `runs/driver/d17/model_best_bc.zip`. d17 took 11.6 h. PPO training (`scripts.train_driver`) and PPO fine-tuning (`scripts.finetune_rl`) were tried and not adopted (LOG.md).

## 2. Qualify the driver (done: attempt 3)

```powershell
.\.venv\Scripts\python.exe -m scripts.qualify --model runs\driver\d17\model_best_bc.zip --attempt 3 --workers 8
```

Attempt 3, with 8 workers, took 57 min for T3, 25 min for T2 and 17 min for T1.

This prints PASS or FAIL for T3, then T2, then T1, stopping at the first PASS, and appends to `docs/QUALIFICATION.md`. Progress is printed every 20 scenes. If it is interrupted, repeat the same command with the same `--attempt`: it continues where it stopped. A PASS writes `models/QUALIFIED.json` and `models/sut.zip`, which become the frozen SUT for everything that follows. On a FAIL at every tier, train more (step 1) and re-test with a new `--attempt`; never reuse an attempt number.

## 3. Search runs (40 runs: 10 seeds × 4 methods, about 70 min per seed on T1)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_search --hours 8
```

It runs seed 101 for random, cmaes, bo and rl, then seed 202, and so on to 1010, skipping finished runs and continuing unfinished ones. Measured on 2026-11-10: about 11 min for random, 9 for CMA-ES, 34 for BO (20 of them its own computation) and 15–19 for RL per 1,000-evaluation run, more when a method finds many failures. `--hours 8` stops after about 6–7 seeds; repeat the command to continue.

## 4. RQ4 and the report

```powershell
.\.venv\Scripts\python.exe -m scripts.rq4
.\.venv\Scripts\python.exe -m scripts.report
```

`reports/RESULTS.md` and the figures are rebuilt from the logs every time.

## Useful extras

```powershell
# watch the AI drive in MetaDrive's 3D view: 20 random T1 scenes with a running tally (--speed 2 = twice as fast)
.\.venv\Scripts\python.exe -m scripts.watch_driver --model runs\driver\d11\model_best_bc.zip --tier T1
# the same scenes with a reference driver or the 60 km/h teacher, to compare
.\.venv\Scripts\python.exe -m scripts.watch_driver --driver idm60 --tier T1
.\.venv\Scripts\python.exe -m scripts.watch_driver --driver teacher60 --tier T1
# a 20 km/h driver (d2, d3) in the 30 km/h traffic it was measured in
.\.venv\Scripts\python.exe -m scripts.watch_driver --model runs\driver\d3\checkpoints\bc_round_02.zip --tier T1 --traffic30
# replay development-panel scenes by row (the failed attempt-1 T1 panel, rows as in LOG.md), e.g. where a driver failed
.\.venv\Scripts\python.exe -m scripts.watch_driver --model runs\driver\d11\model_best_bc.zip --row 134,302
# the same model with controller settings changed, e.g. d9 without and with the braking layer; --headless prints outcomes only
.\.venv\Scripts\python.exe -m scripts.watch_driver --model runs\driver\d9\model_best_bc.zip --row 28 --set threshold=0.3
.\.venv\Scripts\python.exe -m scripts.watch_driver --model runs\driver\d9\model_best_bc.zip --row 28 --set threshold=0.3,aeb_ttc=2.5
# every driver, its problem, its fix and a replay that shows it: LOG.md, 2026-11-04, "the project page"
# watch a scenario (GIF in runs\inspect); --driver sut uses models\sut.zip
.\.venv\Scripts\python.exe -m scripts.inspect_scenario --event cut_in --driver idm --seed 120005
# how often the reference drivers solve random scenes, per tier
.\.venv\Scripts\python.exe -m scripts.calibrate_references --tier T3 --scenes 48
```
