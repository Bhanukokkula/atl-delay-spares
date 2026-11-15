# Delay risk and spare aircraft placement: DL departures from ATL

> **Independent portfolio project on public data.** Not affiliated with or endorsed by Delta Air Lines. Uses no Delta internal data. Flight data comes from the U.S. DOT Bureau of Transportation Statistics (BTS); weather from the Iowa Environmental Mesonet (IEM).

**Interactive demo:** https://bhanukokkula.github.io/atl-delay-spares/ · **Two-page brief:** [outputs/brief.pdf](outputs/brief.pdf) · **Decisions log:** [DECISIONS.md](DECISIONS.md)

## Problem

Two hours before each scheduled departure, which Delta mainline flights leaving Atlanta (ATL) are at risk of a **15+ minute departure delay**? And if the operation has a few spare aircraft, where should they go? Does an optimization model do better than a simple "highest risk first" rule?

Every number below comes from a file in `outputs/`, written by the scripts in `src/`. Ranges in brackets are 95% intervals from resampling test-month days (see [Uncertainty](#uncertainty)).

## Data

- **Flights:** BTS "Reporting Carrier On-Time Performance" monthly files, **Sep 2024 – Aug 2026** (24 months). August 2026 was the most recent month published when the data was pulled.
- **Scope:** `Reporting_Airline == "DL"` only (Delta mainline). **Delta Connection flights are flown and reported by other carriers (e.g. Endeavor, SkyWest) and are out of scope.**
- **Two flight tables:** DL departures from ATL (the flights we score) and DL arrivals into ATL (the inbound legs, matched by tail number).
- **Size:** 474,767 scheduled DL ATL departures, 20.4% delayed 15+ minutes (flown flights), 1.34% cancelled ([outputs/phase1_summary.csv](outputs/phase1_summary.csv)).
- **Weather:** ATL METAR/ASOS observations (routine and special reports) from the IEM ASOS archive, same 24 months: 20,422 observations, longest gap 3.0 hours ([outputs/phase7b_weather_summary.csv](outputs/phase7b_weather_summary.csv)).
- **Cancelled flights** (6,374) are excluded from training and scoring. BTS does not record when a cancellation was announced, so some were likely already known at decision time.
- **Midnight handling:** actual departure = scheduled departure + `DepDelay`, so flights that slip past midnight stay on the right date. An inbound flight's arrival is moved to the next day when its arrival clock time is earlier than its departure clock time. This is safe because every domestic origin is in ATL's time zone or west of it. Checked: every inbound's implied time-zone gap came out to a whole number of hours.
- **Reproducible:** the full pipeline was rerun from an empty `data/` folder and a fresh environment built from `requirements.txt`. Every CSV and chart came out byte-identical. `data/` is not in git; rerun the scripts to rebuild it.

## Approach

### 1. Features: only what is known 2 hours before departure

Decision time is 2 hours before scheduled departure (ATL local time). [src/config.py](src/config.py) holds the approved feature lists and a `check_no_leakage()` function. It stops the program if any forbidden column (actual times, `DepDelay`, `TaxiOut`, `WheelsOff`, `ArrDelay`, all delay-cause columns) or any column not on the lists reaches a model. Delay-cause columns are never even loaded. A second guard fails if any single feature predicts the label with AUC above 0.9. A third fails if any weather observation used for a flight is later than its decision time.

| Feature | Why it is known at decision time |
|---|---|
| `dep_hour`, `day_of_week`, `days_to_holiday` | Schedule and calendar |
| `dest`, `distance`, `sched_block_min` | Published schedule |
| `same_hour_departures` | Count of DL ATL departures scheduled in the same hour (full schedule, including flights later cancelled) |
| `route_delay_rate` | Delay rate to that destination from training months only, computed **out-of-fold by month** for training flights |
| `inbound_dep_delay` | The same tail's earlier same-day inbound into ATL: its departure delay, used **only if** its estimated actual departure (scheduled arrival − scheduled block + departure delay, in ATL time) was at or before decision time; otherwise 0 |
| `recent_avg_delay`, `recent_count`, `recent_known` | Delay of DL ATL flights that actually left in the 2 hours before decision time |
| `backlog_count` | DL ATL flights scheduled in the 2 hours before decision time that had not left by decision time |
| `wx_thunder`, `wx_precip`, `wx_visibility_mi`, `wx_ceiling_ft`, `wx_gust_kt` | From the **latest ATL observation at or before decision time** (median age 26 minutes; none missing or older than 3 hours) |

Three more inbound features were built, then **left out of the decision model** because of aircraft swaps (see below): `turn_min` (scheduled turn time), `has_inbound`, `inbound_delay_known`. `month` was cut because training never saw the validation and test months under the original 12-month window.

**Aircraft swaps affect all inbound features.** BTS records the tail number that *actually flew*, not the tail planned 2 hours earlier. When the operation swaps aircraft, the recorded tail's "inbound" reflects that later decision. Evidence, from training flights ([outputs/phase2_turn_buckets.csv](outputs/phase2_turn_buckets.csv)):
- Scheduled turns under 40 minutes were delayed 41–91% of the time, even though their inbound left about 3 minutes early on average. From 40–45 minutes the rate drops to 20.4%, against 19.8% for all training flights. Turns under 40 minutes (12,925 flights) are treated as "no reliable inbound".
- Even after that cutoff, afternoon flights with no reliable inbound were delayed 45.1% of the time, against 21.2% with one. Spare or swapped aircraft still show up through the flag.

### 2. Model

- **Time-based split:** train Sep 2024 – May 2026 (405,579 flights), validation Jun – Jul 2026 (41,927), test Aug 2026 (20,887). No random splits. Validation is used for LightGBM early stopping and for every choice; test was used only to report.
- Logistic regression baseline (numeric features, scaled), then LightGBM with four feature sets: **(a)** all features, **(b) swap-safe**: all except `turn_min`, `has_inbound`, `inbound_delay_known`, **(c)** no inbound features, **(b) + weather**.
- **Chosen model:** (b) + weather, chosen on validation, plus the daily drift correction below.

### 3. Daily drift correction

The backtest showed the model under-predicts bad months and over-predicts calm ones. The fix shifts every prediction on day *d* by one log-odds offset, chosen so that the average adjusted prediction over the **7 completed days before *d*** equals the share of those flights that were really delayed ([src/drift.py](src/drift.py)). The current day is never used. Every flight's label is known 15 minutes after its scheduled departure, so all of day *d−1* is known by 00:15 on day *d*, before any decision. The offset shifts all of a day's flights together, so it can't reorder flights within a day.

### 4. Spare aircraft decision

For each day of the test month:

- `risk_minutes = P(delay 15+) × average delay minutes of delayed flights in the same departure hour` (training data; [outputs/phase4_hourly_delay_minutes.csv](outputs/phase4_hourly_delay_minutes.csv), 49.1–77.4 minutes by hour).
- **Optimization (PuLP + HiGHS):** binary `x[s, f]`. Maximize risk minutes removed. Each flight gets at most one spare. For every spare and every 15-minute slot `t`, at most one assigned flight departs in `[t, t+H)`.
- **Greedy:** flights from highest to lowest risk; each gets the first spare that is free.
- **Random:** random order, first free spare, averaged over 20 seeds.
- Greedy and random use **exactly the same "free" rule** as the optimization, so the comparison is fair. A check confirms the optimization never scores below greedy on any day.

### 5. Monitoring

Month-by-month backtest of the chosen model: for each month from Mar 2025 to Aug 2026, train on all earlier months and test on that month. The route delay rate is recomputed inside every fold from earlier months only, and every fold uses 200 trees. For the drift correction, each fold's out-of-sample predictions are joined into one series, so every day's offset uses the 7 days before it.

## Results

### Model (test month, Aug 2026; delay rate 22.3%) — [outputs/phase3_results.csv](outputs/phase3_results.csv)

| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test calibration error | Mean predicted vs actual (test) |
|---|---|---|---|---|---|
| Logistic regression, all features | 0.787 | 0.765 | 0.526 | 0.016 | 22.0% vs 22.3% |
| LightGBM (a), all features | 0.824 | 0.800 | 0.609 | 0.015 | 20.7% vs 22.3% |
| LightGBM (b), swap-safe | 0.795 | 0.769 | 0.539 | 0.013 | 21.2% vs 22.3% |
| LightGBM (c), no inbound features | 0.754 | 0.726 | 0.476 | 0.012 | 21.3% vs 22.3% |
| LightGBM (b) + weather | 0.800 | 0.772 | 0.548 | 0.014 | 21.3% vs 22.3% |
| **(b) + weather + daily correction (chosen)** | n/a | **0.772 [0.744–0.798]** | **0.547** | **0.014** | **22.2% vs 22.3%** |

Random guessing gives ROC-AUC 0.5 and PR-AUC 0.223. The true performance with honest features is somewhere between (c) and (a). (b) keeps the inbound delay, the most physical signal (a late inbound causes a late outbound), and drops the features most exposed to swaps.

- **Weather:** chosen because it was better on validation (ROC-AUC 0.795 → 0.800, PR-AUC 0.619 → 0.631). **On the test month the gain is small and may not be real:** +0.0031 ROC-AUC, with a 95% interval of −0.0005 to +0.0071, which includes zero. The five weather features together carry 4.0% of the chosen model's importance (thunderstorm 2.0%).
- **Calibration:** [outputs/calibration.png](outputs/calibration.png). The daily correction brought the test month's average prediction from 21.3% to 22.2% (actual 22.3%). Calibration error barely moved for the weather model (0.0144 → 0.0135), though it fell more for model (b) (0.0131 → 0.0094).
- **Feature importance, chosen model** ([outputs/feature_importance.png](outputs/feature_importance.png)): `inbound_dep_delay` 33.6%, `backlog_count` 25.3%, `recent_avg_delay` 15.0%, `dest` 12.7%, `dep_hour` 3.9%. The rest are each 2.0% or less.

### Spare aircraft (test month, S = 3, R = 0.6, H = 4 h, chosen model) — [outputs/phase4_summary.csv](outputs/phase4_summary.csv)

| Method | Expected minutes removed / day | % of optimal | Spare uses / day | Assigned flights really delayed 15+ | Minutes removed / day on actual outcomes* |
|---|---|---|---|---|---|
| Optimization | 247.2 | 100.0% | 13.6 | 46.2% [40.3–51.6] | 221.3 |
| Greedy | 226.5 | 91.6% [90.4–92.8] | 11.8 | 49.0% [43.2–55.6] | 217.5 |
| Random | 90.4 | 36.6% | 11.1 | 21.9% [18.4–26.0] | 90.0 |

\*R × actual delay minutes of assigned flights that really were delayed 15+, assuming R holds.

**Plainly: greedy is nearly as good.** It gets 91.6% [90.4–92.8] of the optimal *expected* minutes, and on what actually happened the gap is 1.7% (217.5 vs 221.3 minutes per day). The optimization wins by fitting more spare uses into the day (13.6 vs 11.8), not by finding riskier flights. Greedy's first pick can block a spare for 4 hours. **The model's picks are clearly better than random:** flights given a spare were really delayed 24.3 points [19.9–28.4] more often than random picks under the optimization, and 27.1 points [22.9–31.9] more often under greedy.

**When the optimization would matter** ([outputs/phase4_sensitivity.csv](outputs/phase4_sensitivity.csv), [outputs/spares_by_s.png](outputs/spares_by_s.png)):
- **Spare count:** greedy stays at 89.2–92.6% of optimal for 1 to 8 spares.
- **Turnaround time:** the gap grows when spares turn around faster (greedy 89.9% of optimal at H = 2 h) and shrinks when they are tied up longer (93.8% at 6 h, 96.4% at 8 h).
- **Bigger problems:** more interacting constraints (several airports or fleet types sharing spares, crew and gate limits) are where greedy usually falls further behind.
- **R** multiplies every flight's value equally, so it never changes which flights are chosen. It only scales the minutes (e.g. 123.6 / 247.2 / 370.8 expected minutes per day for the optimization at R = 0.3 / 0.6 / 0.9 with 3 spares).

**What the Phase 7 upgrades changed in the decisions** ([outputs/phase7b_phase4_models.csv](outputs/phase7b_phase4_models.csv)): minutes removed on actual outcomes per day is the fair comparison across model versions. Expected minutes depend on each version's own probability level.

| Prediction used | Optimization | Greedy | Greedy hit rate |
|---|---|---|---|
| (b) | 209.7 | 205.5 | 47.7% |
| (b) + daily correction | 213.4 | 205.3 | 47.7% |
| (b) + weather | 224.7 | 212.9 | 49.2% |
| (b) + weather + daily correction (chosen) | 221.3 | 217.5 | 49.0% |

### Monitoring and the drift fix — [outputs/phase5_backtest.csv](outputs/phase5_backtest.csv), [outputs/phase7a_drift_summary.csv](outputs/phase7a_drift_summary.csv), [outputs/drift_offset.png](outputs/drift_offset.png)

- **Ranking:** ROC-AUC ranged from 0.734 (Apr 2026) to 0.807 (Jul 2026), mean 0.767, with no downward trend over time.
- **Without the correction, calibration drifts with the month's delay level.** The model under-predicts bad months and over-predicts calm ones: the correlation between a month's delay rate and (predicted − actual) is −0.88. The worst month was Jun 2025, predicted 28.4% vs actual 34.3% (calibration error 0.058).
- **With the daily correction:**

| 18-month backtest | Before | After |
|---|---|---|
| Mean calibration error | 0.0184 | 0.0101 |
| Worst month's calibration error | 0.0582 | 0.0150 |
| Average monthly bias (points, absolute) | 1.64 | 0.38 |
| Month-severity correlation | −0.88 | −0.18 |
| Mean ROC-AUC | 0.7673 | 0.7668 |

It helps a lot overall, but not every month: 5 of 18 months got slightly worse calibration (Aug 2025, Nov 2025, Jan 2026, Mar 2026, May 2026).

### Uncertainty

Test-month days are resampled with replacement 1,000 times. Whole days are resampled, not flights, because flights on the same day share weather and knock-on delays ([src/07_bootstrap.py](src/07_bootstrap.py), [outputs/phase7c_bootstrap.csv](outputs/phase7c_bootstrap.csv)). With only 31 test days, the intervals are wide for hit rates and narrow for greedy-vs-optimal.

| Metric | Estimate | 95% interval |
|---|---|---|
| Test ROC-AUC, chosen model | 0.772 | 0.744 – 0.798 |
| Test ROC-AUC gain from weather | +0.0031 | −0.0005 – +0.0071 |
| Hit rate, optimization − random (points) | 24.3 | 19.9 – 28.4 |
| Hit rate, greedy − random (points) | 27.1 | 22.9 – 31.9 |
| Greedy as % of optimal | 91.6% | 90.4% – 92.8% |

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
| Model used for decisions | LightGBM (b) + weather, with the daily correction, not otherwise recalibrated | `config.CHOSEN_MODEL`, `config.USE_DRIFT_OFFSET` |
| Daily correction | One log-odds offset per day, fitted on the 7 previous completed days; no offset without 7 days of history | `src/drift.py` |
| Weather observation | Latest at or before decision time; older than 3 h = missing; thunderstorm includes "in the vicinity" (VCTS); no ceiling = 20,000 ft; no gust reported = 0 | `02b_weather.py`, `03_features.py` |
| "No reliable inbound" cutoff | Scheduled turn < 40 min | `config.TURN_CUTOFF_MIN` |
| Uncertainty | 1,000 day-level bootstrap resamples, percentile intervals | `07_bootstrap.py` |

## What didn't work

> **[EDIT IN MY WORDS]** Rough draft. Numbers are from `outputs/phase3_experiments.csv`, `outputs/phase3_feature_importance.csv`, `outputs/phase7c_bootstrap.csv`, `outputs/phase7a_drift_offset.csv` and `DECISIONS.md`.

- `same_hour_departures` (congestion): dropping it changed validation AUC by only −0.0008.
- `distance`, `sched_block_min`, `recent_known`: at most 0.1% of the chosen model's importance each.
- In-sample route delay rate: looked better than it was (0.565 AUC on its own vs 0.544 out-of-fold, same 12-month data). Replaced with out-of-fold.
- `month`: with 12 months of data, training never saw Jun–Aug. Cut, and the data was extended to 24 months.
- `turn_min`, `has_inbound`, `inbound_delay_known`: added 0.029 validation AUC, but that gain is at least partly swap leakage, so they were dropped from the decision model.
- Logistic regression: 0.765 test AUC vs 0.800 for LightGBM on the same features.
- Optimization vs greedy: the optimization model was expected to matter more. On actual outcomes it beat greedy by 1.7%.
- ATL weather: better on validation, but the test-month gain (+0.0031 AUC) has an interval that includes zero.
- Daily drift correction: fixed most monthly bias, but made 5 of 18 backtest months slightly worse, and on the test month it improved the weather model's calibration error only from 0.0144 to 0.0135.

## Why I made these choices

> **[EDIT IN MY WORDS]** Rough draft.

- Dropped cancellations because the label should mean one thing, and BTS doesn't say when a cancellation was known.
- Picked the turn cutoff from the data (training months only), not a rule of thumb.
- Used a swap-safe model for spare decisions because a model that detects swaps would make spare allocation look better than it is. A swap *is* a spare being used.
- Kept calibration in view because the spare decision multiplies probabilities by minutes. That's also why I added the daily correction: a model that under-predicts bad months under-values spares exactly when they matter most.
- Chose the weather model on validation, not test, and then reported the test interval honestly, even though it includes zero.
- Resampled whole days for the intervals because flights on the same day aren't independent.
- Made greedy and random use the same feasibility rule as the optimization so the comparison is fair.

## Limitations

- The daily correction only fixes the overall level of each day. It can't reorder flights within a day, and it made 5 of 18 backtest months slightly worse. On the test month the corrected model predicts 22.2% vs 22.3% actual.
- Tail numbers are what actually flew. Swaps affect all inbound features, including `inbound_dep_delay`.
- The inbound's actual departure time is estimated from scheduled times plus delay, without time-zone tables.
- Weather is ATL observations only: no forecasts and no weather at destinations or along routes. METAR reports are sent a few minutes after their observation time; that lag is ignored.
- R and H are assumptions. Spares are assumed to be at ATL, interchangeable, and crewed. Risk minutes use an hourly average, not a per-flight forecast of minutes.
- Cancellations are excluded. Delta Connection is out of scope. No gate, crew, or passenger-connection data. One airport, one 31-day test month for the decision comparison, so intervals are wide.

## How to rerun

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/01_download.py   # 24 monthly BTS zips (~700 MB) into data/raw/, can take 20+ min
.venv/bin/python src/02_clean.py      # data/clean/, outputs/phase1_summary.csv
.venv/bin/python src/02b_weather.py   # ATL weather from IEM into data/raw/weather/, data/clean/weather.parquet (~1 min)
.venv/bin/python src/03_features.py   # data/features/, outputs/phase2_*.csv
.venv/bin/python src/04_model.py      # data/model/, outputs/phase3_*, phase7a_test_offsets_*, calibration.png, feature_importance.png
.venv/bin/python src/05_spares.py     # outputs/phase4_*, phase7b_phase4_models.csv, spares_by_s.png (several minutes)
.venv/bin/python src/06_backtest.py   # outputs/phase5_backtest.csv, phase7a_drift_*, backtest.png, drift_offset.png
.venv/bin/python src/07_bootstrap.py  # outputs/phase7c_bootstrap.csv
.venv/bin/python src/08_brief.py      # outputs/brief.pdf
.venv/bin/python src/09_demo.py       # docs/demo_data.json, docs/preview.png, docs/brief.pdf (about 10 minutes: 1,085 settings)
```

Run every script from the project root. Each one reads earlier steps' outputs, so any step can be rerun on its own. If a BTS or IEM download fails, the script prints the exact URL and where to save the file. The most recent month is found automatically, so a later rerun will use a later 24-month window and produce different numbers.

PuLP's bundled CBC solver is not available on this machine (Python 3.14, Apple Silicon), so PuLP uses HiGHS (`highspy`).

## Demo on GitHub Pages

The demo in [docs/](docs/) is plain HTML and JavaScript with Chart.js loaded from cdnjs. It reads `docs/demo_data.json`, which holds precomputed results for every test-month day, every S from 1 to 5 and every H from 2 to 8 hours. There is no server to sleep or pay for. `src/09_demo.py` writes the JSON using the same functions as `05_spares.py`, and it fails if the default setting doesn't match `outputs/phase4_daily.csv` exactly.

To publish it:

1. Create a public repository named `atl-delay-spares` under the `Bhanukokkula` account on GitHub (no README, no .gitignore, no license, since this repo already has them).
2. Push this project:
   ```bash
   git remote add origin https://github.com/Bhanukokkula/atl-delay-spares.git
   git branch -M main
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder: `/docs` → Save.**
4. Wait a minute or two, then open https://bhanukokkula.github.io/atl-delay-spares/ on a laptop and a phone.
5. Check the link preview: paste the URL into LinkedIn's Post Inspector (https://www.linkedin.com/post-inspector/). It reads the Open Graph tags in `docs/index.html` and shows `docs/preview.png` (1200 × 627). If you change the preview later, run the Post Inspector again to refresh LinkedIn's cache.

To preview locally before pushing (the page loads its JSON with `fetch`, which doesn't work from a `file://` path):

```bash
.venv/bin/python -m http.server 8000 --directory docs
# then open http://localhost:8000
```

If the repository or user name changes, update the three URLs in `docs/index.html` (`og:image`, `og:url`, and the GitHub link) and `DEMO_URL` / `REPO_URL` in `src/08_brief.py`.
