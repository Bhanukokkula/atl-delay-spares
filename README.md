# Delay risk and spare aircraft placement: DL departures from ATL

> **Independent portfolio project on public data.** Not affiliated with or endorsed by Delta Air Lines. Uses no Delta internal data. All data comes from the U.S. DOT Bureau of Transportation Statistics (BTS).

## Problem

Two hours before each scheduled departure, which Delta mainline flights leaving Atlanta (ATL) are at risk of a **15+ minute departure delay**? And if the operation has a few spare aircraft, where should they go? Does an optimization model do better than a simple "highest risk first" rule?

Every number below comes from a file in `outputs/`, written by the scripts in `src/`. The two-page summary is [outputs/brief.pdf](outputs/brief.pdf). Decisions made along the way are logged in [DECISIONS.md](DECISIONS.md).

## Data

- **Source:** BTS "Reporting Carrier On-Time Performance" monthly files, **Sep 2024 – Aug 2026** (24 months). August 2026 was the most recent month published when the data was pulled.
- **Scope:** `Reporting_Airline == "DL"` only (Delta mainline). **Delta Connection flights are flown and reported by other carriers (e.g. Endeavor, SkyWest) and are out of scope.**
- **Two tables:** DL departures from ATL (the flights we score) and DL arrivals into ATL (the inbound legs, matched by tail number).
- **Size:** 474,767 scheduled DL ATL departures, 20.4% delayed 15+ minutes (flown flights), 1.34% cancelled ([outputs/phase1_summary.csv](outputs/phase1_summary.csv)).
- **Cancelled flights** (6,374) are excluded from training and scoring. BTS does not record when a cancellation was announced, so some were likely already known at decision time.
- **Midnight handling:** actual departure = scheduled departure + `DepDelay`, so flights that slip past midnight stay on the right date. An inbound flight's arrival is moved to the next day when its arrival clock time is earlier than its departure clock time. This is safe because every domestic origin is in ATL's time zone or west of it. Checked: every inbound's implied time-zone gap came out to a whole number of hours.
- `data/` is not in git. Rerun the scripts to rebuild it.

## Approach

### 1. Features: only what is known 2 hours before departure

Decision time is 2 hours before scheduled departure (ATL local time). [src/config.py](src/config.py) holds the approved feature list and a `check_no_leakage()` function. It stops the program if any forbidden column (actual times, `DepDelay`, `TaxiOut`, `WheelsOff`, `ArrDelay`, all delay-cause columns) or any column not on the list reaches a model. Delay-cause columns are never even loaded. A second guard fails if any single feature predicts the label with AUC above 0.9.

| Feature | Why it is known at decision time |
|---|---|
| `dep_hour`, `day_of_week`, `days_to_holiday` | Schedule and calendar |
| `dest`, `distance`, `sched_block_min` | Published schedule |
| `same_hour_departures` | Count of DL ATL departures scheduled in the same hour (full schedule, including flights later cancelled) |
| `route_delay_rate` | Delay rate to that destination from training months only, computed **out-of-fold by month** for training flights |
| `inbound_dep_delay` | The same tail's earlier same-day inbound into ATL: its departure delay, used **only if** its estimated actual departure (scheduled arrival − scheduled block + departure delay, in ATL time) was at or before decision time; otherwise 0 |
| `recent_avg_delay`, `recent_count`, `recent_known` | Delay of DL ATL flights that actually left in the 2 hours before decision time |
| `backlog_count` | DL ATL flights scheduled in the 2 hours before decision time that had not left by decision time |

Three more inbound features were built, then **left out of the chosen model** because of aircraft swaps (see below): `turn_min` (scheduled turn time), `has_inbound`, `inbound_delay_known`. `month` was cut because training never saw the validation and test months under the original 12-month window.

**Aircraft swaps affect all inbound features.** BTS records the tail number that *actually flew*, not the tail planned 2 hours earlier. When the operation swaps aircraft, the recorded tail's "inbound" reflects that later decision. Evidence, from training flights ([outputs/phase2_turn_buckets.csv](outputs/phase2_turn_buckets.csv)):
- Scheduled turns under 40 minutes were delayed 41–91% of the time, even though their inbound left about 3 minutes early on average. From 40–45 minutes the rate drops to 20.4%, against 19.8% for all training flights. Turns under 40 minutes (12,925 flights) are treated as "no reliable inbound".
- Even after that cutoff, afternoon flights with no reliable inbound were delayed 45.1% of the time, against 21.2% with one. Spare or swapped aircraft still show up through the flag.

### 2. Model

- **Time-based split:** train Sep 2024 – May 2026 (405,579 flights), validation Jun – Jul 2026 (41,927), test Aug 2026 (20,887). No random splits. Validation is used for LightGBM early stopping and for choices; test was used only to report.
- Logistic regression baseline (numeric features, scaled), then LightGBM.
- Three LightGBM feature sets, reported as a range: **(a)** all features, **(b) swap-safe**: all except `turn_min`, `has_inbound`, `inbound_delay_known` (**chosen**), **(c)** no inbound features.

### 3. Spare aircraft decision

For each day of the test month:

- `risk_minutes = P(delay 15+) × average delay minutes of delayed flights in the same departure hour` (training data; [outputs/phase4_hourly_delay_minutes.csv](outputs/phase4_hourly_delay_minutes.csv), 49.1–77.4 minutes by hour).
- **Optimization (PuLP + HiGHS):** binary `x[s, f]`. Maximize risk minutes removed. Each flight gets at most one spare. For every spare and every 15-minute slot `t`, at most one assigned flight departs in `[t, t+H)`.
- **Greedy:** flights from highest to lowest risk; each gets the first spare that is free.
- **Random:** random order, first free spare, averaged over 20 seeds.
- Greedy and random use **exactly the same "free" rule** as the optimization, so the comparison is fair. A check confirms the optimization never scores below greedy on any day.

### 4. Monitoring

Month-by-month backtest: for each month from Mar 2025 to Aug 2026, train on all earlier months and test on that month. The route delay rate is recomputed inside every fold from earlier months only, and every fold uses 200 trees.

## Results

### Model (test month, Aug 2026; delay rate 22.3%) — [outputs/phase3_results.csv](outputs/phase3_results.csv)

| Model | ROC-AUC | PR-AUC | Calibration error | Mean predicted vs actual |
|---|---|---|---|---|
| Logistic regression, all features | 0.765 | 0.526 | 0.016 | 22.0% vs 22.3% |
| LightGBM (a), all features | 0.800 | 0.609 | 0.015 | 20.7% vs 22.3% |
| **LightGBM (b), swap-safe (chosen)** | **0.769** | **0.539** | **0.013** | **21.2% vs 22.3%** |
| LightGBM (c), no inbound features | 0.726 | 0.476 | 0.012 | 21.3% vs 22.3% |

Random guessing gives ROC-AUC 0.5 and PR-AUC 0.223. The true performance with honest features is somewhere between (c) and (a). (b) keeps the inbound delay, the most physical signal (a late inbound causes a late outbound), and drops the features most exposed to swaps.

- **Calibration:** [outputs/calibration.png](outputs/calibration.png). All three LightGBM models track the diagonal. Model (b) under-predicts the test month by 1.1 points. No recalibration was applied.
- **Feature importance, model (b)** ([outputs/feature_importance.png](outputs/feature_importance.png)): `inbound_dep_delay` 33.1%, `backlog_count` 29.7%, `recent_avg_delay` 14.0%, `dest` 13.2%, `dep_hour` 3.6%. The rest are each under 3%.

### Spare aircraft (test month, S = 3, R = 0.6, H = 4 h) — [outputs/phase4_summary.csv](outputs/phase4_summary.csv)

| Method | Expected minutes removed / day | % of optimal | Spare uses / day | Assigned flights really delayed 15+ | Minutes removed / day on actual outcomes* |
|---|---|---|---|---|---|
| Optimization | 239.7 | 100.0% | 13.6 | 46.7% | 209.7 |
| Greedy | 219.4 | 91.5% | 11.8 | 47.7% | 205.5 |
| Random | 86.5 | 36.1% | 11.1 | 21.9% | 90.0 |

\*R × actual delay minutes of assigned flights that really were delayed 15+, assuming R holds.

**Plainly: greedy is nearly as good.** It gets 91.5% of the optimal *expected* minutes, and on what actually happened the gap is about 2% (205.5 vs 209.7 minutes per day). The optimization wins by fitting more spare uses into the day (13.6 vs 11.8), not by finding riskier flights. Greedy's first pick can block a spare for 4 hours.

**When the optimization would matter** ([outputs/phase4_sensitivity.csv](outputs/phase4_sensitivity.csv), [outputs/spares_by_s.png](outputs/spares_by_s.png)):
- **Spare count:** greedy stays at 91.1–92.3% of optimal for 1 to 8 spares.
- **Turnaround time:** the gap grows when spares turn around faster (greedy 88.9% of optimal at H = 2 h) and shrinks when they are tied up longer (94.4% at 6 h, 96.1% at 8 h).
- **Bigger problems:** more interacting constraints (several airports or fleet types sharing spares, crew and gate limits) are where greedy usually falls further behind.
- **R** multiplies every flight's value equally, so it never changes which flights are chosen. It only scales the minutes (e.g. 119.9 / 239.7 / 359.6 expected minutes per day for the optimization at R = 0.3 / 0.6 / 0.9 with 3 spares).

### Monitoring — [outputs/phase5_backtest.csv](outputs/phase5_backtest.csv), [outputs/backtest.png](outputs/backtest.png)

- **Ranking:** ROC-AUC ranged from 0.730 (Apr 2026) to 0.802 (Jun 2025), mean 0.764, with no downward trend over time.
- **Calibration drifts with the month's delay level.** The model under-predicts bad months and over-predicts calm ones: the correlation between a month's delay rate and (predicted − actual) is −0.87. The worst month was Jun 2025, predicted 28.9% vs actual 34.3% (−5.4 points, calibration error 0.054). Training for that fold had no summer months. After the first summer is in training, calibration error stays at or below 0.028.
- **Where it drifts:** the peaks are summer months (Jun–Jul 2025, Jul 2026). Over-prediction shows in calm months (Sep 2025 +2.2 points).

## Assumptions

| Assumption | Value | Where |
|---|---|---|
| Decision time | 2 hours before scheduled departure, ATL local | `config.DECISION_HOURS_BEFORE` |
| Delay label | Departure delay ≥ 15 min; cancelled flights excluded | `02_clean.py` |
| Spare aircraft S | 3 | `05_spares.py` |
| Share of risk minutes a spare removes R | 0.6 (not measured; assumed) | `05_spares.py` |
| Hours a spare is busy after use H | 4 (not measured; assumed) | `05_spares.py` |
| Busy rule | Per spare, at most one assigned flight departing in any `[t, t+H)`, t on a 15-min grid. Two flights can share a spare when they are up to 15 min less than H apart. | `05_spares.py` |
| Risk minutes | P(delay) × average delay of delayed flights in the same departure hour, training months | `05_spares.py` |
| Hours with < 30 delayed training flights | Use the overall average, 58.9 min (hours 0–5) | `phase4_hourly_delay_minutes.csv` |
| Spares | At ATL, interchangeable, crewed, reusable after H | — |
| Random baseline | Average of 20 seeds | `05_spares.py` |
| Model used for decisions | LightGBM (b), swap-safe features, not recalibrated | `config.CHOSEN_PREDICTION` |
| "No reliable inbound" cutoff | Scheduled turn < 40 min | `config.TURN_CUTOFF_MIN` |

## What didn't work

> **[EDIT IN MY WORDS]** Rough draft. Numbers are from `outputs/phase3_experiments.csv`, `outputs/phase3_feature_importance.csv` and `DECISIONS.md`.

- `same_hour_departures` (congestion): dropping it changed validation AUC by only −0.0008.
- `distance`, `sched_block_min`, `recent_known`: at most 0.2% of model (b)'s importance.
- In-sample route delay rate: looked better than it was (0.565 AUC on its own vs 0.544 out-of-fold, same 12-month data). Replaced with out-of-fold.
- `month`: with 12 months of data, training never saw Jun–Aug. Cut, and the data was extended to 24 months.
- `turn_min`, `has_inbound`, `inbound_delay_known`: added 0.029 validation AUC, but that gain is at least partly swap leakage, so they were dropped from the decision model.
- Logistic regression: 0.765 test AUC vs 0.800 for LightGBM on the same features.
- Optimization vs greedy: the optimization model was expected to matter more. On actual outcomes it beat greedy by about 2%.

## Why I made these choices

> **[EDIT IN MY WORDS]** Rough draft.

- Dropped cancellations because the label should mean one thing, and BTS doesn't say when a cancellation was known.
- Picked the turn cutoff from the data (training months only), not a rule of thumb.
- Used the swap-safe model for spare decisions because a model that detects swaps would make spare allocation look better than it is. A swap *is* a spare being used.
- Kept calibration in view because Phase 4 multiplies probabilities by minutes.
- Made greedy and random use the same feasibility rule as the optimization so the comparison is fair.

## Limitations

- Model (b) under-predicts the test month's delay rate by 1.1 points, so risk minutes are slightly understated. Not recalibrated.
- Calibration drifts with the month's delay level (see Monitoring). Bad months are under-predicted.
- Tail numbers are what actually flew. Swaps affect all inbound features, including `inbound_dep_delay`.
- The inbound's actual departure time is estimated from scheduled times plus delay, without time-zone tables.
- R and H are assumptions. Spares are assumed to be at ATL, interchangeable, and crewed. Risk minutes use an hourly average, not a per-flight forecast of minutes.
- Cancellations are excluded. Delta Connection is out of scope. No weather, gate, crew, or passenger-connection data. One airport, one test month for the decision comparison.

## How to rerun

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/01_download.py   # 24 monthly zips (~700 MB) into data/raw/, can take 20+ min
.venv/bin/python src/02_clean.py      # data/clean/, outputs/phase1_summary.csv
.venv/bin/python src/03_features.py   # data/features/, outputs/phase2_*.csv
.venv/bin/python src/04_model.py      # data/model/, outputs/phase3_*, calibration.png, feature_importance.png
.venv/bin/python src/05_spares.py     # outputs/phase4_*, spares_by_s.png (about 4 min)
.venv/bin/python src/06_backtest.py   # outputs/phase5_backtest.csv, backtest.png
.venv/bin/python src/07_brief.py      # outputs/brief.pdf
```

Run every script from the project root. Each one reads the previous step's output, so any step can be rerun on its own. If a BTS download fails, `01_download.py` prints the exact URL and where to save the file. The most recent month is found automatically, so a later rerun will use a later 24-month window and produce different numbers.

PuLP's bundled CBC solver is not available on this machine (Python 3.14, Apple Silicon), so PuLP uses HiGHS (`highspy`).
