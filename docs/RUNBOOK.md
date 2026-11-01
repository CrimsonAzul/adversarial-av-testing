# Runbook: what to run, in order

Run each command from the project folder in PowerShell when the laptop is free (close heavy apps; every job uses all CPU cores). Every job resumes from where it stopped if you repeat the same command. To end a job early, create `PAUSE` in its run folder (it finishes the current chunk or batch, saves, and exits) or `STOP` (it exits immediately). After each step, add a line to [LOG.md](LOG.md), or ask Claude to read the generated files and log it.

Before an overnight run: charger in, lid open, sleep set to Never on AC power, and Windows Update paused (Settings → Windows Update → Pause updates). Windows may restart for updates outside active hours, which on this laptop are 8:00–2:00, so between 2:00 and 8:00, in the middle of a run. A restart loses at most the current chunk; the same command resumes.

Times are measured on this laptop (about 80 simulator steps/s per process with traffic).

## Suggested schedule (deadline: end of November)

| When | Step | Time |
|---|---|---|
| Night 1 (d1, done) | first attempt: far below the gate; reward error fixed | — |
| Night 2 | 1a imitation warm start (d2, 5 rounds), then 1b PPO `--hours 5` | ~2.5 h + 5–6 h |
| Nights 2–4 | 1b PPO (same command each night) | 8 h each |
| ~Nov 5 | 2 qualification | 1 h per tier tested |
| If it fails | more 1b, then 2 with `--attempt 2` | until Nov 14 at the latest (DESIGN.md rule) |
| Next 2 nights | 3 search runs | ~12 h total |
| Then | 4 RQ4 + report | ~1.5 h |

## Where to watch progress

| Stage | Look at | Updated |
|---|---|---|
| Imitation warm start | `runs/driver/d2/bc_progress.png`, `recordings/bc_round_XX/*.gif` | every round |
| PPO training | `runs/driver/d2/progress.png`, `recordings/steps_XXXXXXXXXX/*.gif` | every 1M steps |
| Search runs | `runs/search/<method>_s<seed>/progress.png`, `recordings/eval_XXXX_sut.gif` + the reference that avoided it | every 50 evaluations / each new kind of failure |
| RL generator learning | `runs/search/rl_s<seed>/rl/updates.jsonl` (also the 3rd panel of its progress.png) | every PPO update (24 evaluations) |

The three driver showcase GIFs (lead_brake, obstacle, cut_in) are the same scenarios at every checkpoint, so they show the same situations getting better.

## 0. Check the pipeline (about 5 minutes)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 1a. Imitation warm start (about 3 hours, once)

```powershell
.\.venv\Scripts\python.exe -m scripts.pretrain_bc --name d2
```

Each round prints the action error and validation success on the 200-scene T2 panel, overall and per event, plus success on 100 T1 scenes (the tier most likely to qualify). Rising validation means the warm start is working. If it is interrupted, the same command resumes at the last finished round. At the end the best round is saved as `runs/driver/d2/model_last.zip`, PPO's starting point.

## 1b. PPO training (several nights)

```powershell
.\.venv\Scripts\python.exe -m scripts.train_driver --name d2 --hours 8
```

It continues from the warm start: first 300k steps of value-network warm-up (printed as "critic warm-up"), then PPO. Every 1M steps it prints training and validation success and keeps the best checkpoint as `runs/driver/d2/model_best.zip`. Measured in pilot 3b (warm start, full difficulty, 6 environments): about 356 steps/s, so 1M steps take about 47 minutes plus a few minutes of validation, and a 5-hour session is about 5–6M steps. Repeat each night until validation success is clearly above 98% or has levelled off.

## 2. Qualify the driver (about 1 hour per tier)

```powershell
.\.venv\Scripts\python.exe -m scripts.qualify --model runs\driver\d2\model_best.zip --attempt 1
```

This prints PASS or FAIL for T3, then T2, then T1, stopping at the first PASS, and appends to `docs/QUALIFICATION.md`. A PASS writes `models/QUALIFIED.json` and `models/sut.zip`, which become the frozen SUT for everything that follows. On a FAIL at every tier, train more (1b) and re-test with `--attempt 2`; never reuse an attempt number.

## 3. Search runs (12 runs, about 45–60 min each)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_search --hours 8
```

It runs seed 101 for rl, cmaes, bo and random, then seed 202, then 303, skipping finished runs and continuing unfinished ones.

## 4. RQ4 and the report

```powershell
.\.venv\Scripts\python.exe -m scripts.rq4
.\.venv\Scripts\python.exe -m scripts.report
```

`reports/RESULTS.md` and the figures are rebuilt from the logs every time.

## Useful extras

```powershell
# watch a scenario (GIF in runs\inspect); --driver sut uses models\sut.zip
.\.venv\Scripts\python.exe -m scripts.inspect_scenario --event cut_in --driver idm --seed 120005
# how often the reference drivers solve random scenes, per tier
.\.venv\Scripts\python.exe -m scripts.calibrate_references --tier T3 --scenes 48
```
