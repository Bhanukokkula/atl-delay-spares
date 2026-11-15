# Decisions log

Each checkpoint records the question asked and my answer word for word, with the date.

## Checkpoint 1 — Treatment of cancelled flights (2026-11-14)

Question: Drop cancelled flights from training and scoring (Option A), or count them as disrupted with label = 1 (Option B)?

Answer (word for word):
> option A

Effect: cancelled departures are removed before modelling. They still count toward the scheduled-departures-per-hour congestion feature, because the schedule is known in advance.

## Checkpoint 2 — Feature approval (2026-11-14)

Questions: (1) how to handle very short scheduled turns caused by tail swaps; (2) cut `month`?; (3) approve the remaining features?

Answer (word for word):
> Show delay rate by turn bucket first, then pick the “no reliable inbound” cutoff from that. Keep the run without inbound features in Phase 3, and say in the README that swaps affect all inbound features. Cut month, and extend data to Sep 2024 – Aug 2026 so training sees a summer. Make route_delay_rate out-of-fold and add a backlog feature for flights scheduled in the last 2 hours that haven’t left. Approve the rest.

How this was applied (Claude's notes, not part of the answer):
- Split with 24 months: validation and test stay the same (Jun–Jul 2026 and Aug 2026); training is every earlier month, Sep 2024 – May 2026 (21 months).
- Turn cutoff = 40 min, chosen from outputs/phase2_turn_buckets.csv (training months only): every bucket under 40 min was 41–91% delayed while its inbound left ~3 min early on average; 40–45 min was 20.4% vs 19.8% for all training flights, and flight volume jumps ~5x at 40 min. 12,925 flights reset to "no reliable inbound".
- Residual swap signal found after the cutoff: training flights departing 12:00 or later with no reliable inbound were 45.1% delayed vs 21.2% with one (`has_inbound` alone AUC 0.505 → 0.540). The Phase 3 run without inbound features measures how much this matters; the feature-set choice goes to Checkpoint 3.
- route_delay_rate is now out-of-fold by month for training flights (alone AUC 0.565 → 0.544 on the 12-month data, showing the in-sample version was optimistic).
- backlog_count excludes cancelled flights, because BTS does not record when a cancellation was announced.

## Checkpoint 3 — Model and recalibration (2026-11-14)

Questions: which model and feature set to carry forward — (a) all features, (b) all except turn_min/has_inbound/inbound_delay_known, (c) no inbound features — and whether to recalibrate.

Answer (word for word):
> LightGBM with option (b). Score (b) on test once now and report (a), (b), (c) side by side as the range. Use (b) in Phase 4, because a model that detects swaps would make spare-aircraft allocation look better than it is. No recalibration; note the 1.6-point under-prediction as a limitation. Now finish Phases 4–6 without stopping, using the defaults I sent earlier.

How this was applied (Claude's notes, not part of the answer):
- Checkpoint 4 is not held as a stop. Phase 4 uses the original defaults S = 3 spares, R = 0.6, H = 4 hours; the S and R sensitivity is still produced and reported.
- Option (b) scored on test once: ROC-AUC 0.7691, PR-AUC 0.5386, calibration error 0.0131. Its mean prediction is 21.2% vs 22.3% actual, so the chosen model under-predicts by 1.1 points; the 1.6-point figure was for option (a). The limitation note uses the (b) figure.

## Reproduction check (2026-11-14)

Full pipeline rerun from an empty `data/` folder and a fresh virtual environment built only from `requirements.txt` (BTS latest month still 2026-08). All 11 CSVs and all 4 PNGs in `outputs/` were byte-identical to the committed versions. `brief.pdf` differed only in its embedded creation date and file ID; its text was identical.

## Phase 7 defaults (review)

Phase 7 was requested to run to the end without stopping. Change to 7d requested during the phase (word for word):
> Change 7d: instead of Streamlit, build a static demo in docs/ (index.html + a small JSON of precomputed results for every test-month day, every S in 1–5 and every H in 2–8h) so it can be hosted on GitHub Pages and never sleeps. Plain HTML + simple JavaScript + one chart library from a CDN. Must work well on a phone. Add Open Graph title, description and a preview image so the LinkedIn link preview looks clean. Keep the banner "Independent demo on public BTS data. Not affiliated with Delta." Link the GitHub repo and brief.pdf from the page. Add GitHub Pages setup steps to the README.

GitHub username given: Bhanukokkula; repo: atl-delay-spares.

Choices made by Claude (please review):

**7a. Drift fix**
- Offset = one log-odds shift per day, fitted so the average adjusted prediction over the previous 7 calendar days equals the share of those flights that were really delayed (halving search, `src/drift.py`). The current day is never used.
- "Completed" day: a flight's label is known 15 minutes after its scheduled departure, so all of day d-1's labels are known by 00:15 on day d, before any decision on day d.
- A day gets an offset only if all 7 previous days have out-of-sample predictions; otherwise no offset. Test month: 31 of 31 days had history (the 7 days come from validation and earlier test days). Backtest: 542 of 549 days (the first 7 days of Mar 2025 had none).
- Backtest: each fold's out-of-sample predictions are stitched into one series before fitting offsets, so a day's offset at a month boundary uses the previous fold's predictions (from a model trained on one month less).
- Validation is not scored with the offset: its first week would need in-sample training-day predictions.
- Result: it helps calibration and does not change ranking. Backtest mean calibration error 0.0191 → 0.0107, month-severity correlation −0.87 → −0.07, mean ROC-AUC 0.7640 → 0.7633. Four months got slightly worse calibration. Test month: calibration error 0.0131 → 0.0094, mean prediction 21.2% → 22.1% vs 22.3% actual.
- Phase 4 is not changed in this step; the model and offset used for Phase 4 are chosen in 7b.

**7b. ATL weather**
- Source: IEM ASOS archive, station ATL, routine (type 3) and special (type 4) reports, downloaded in UTC and converted to America/New_York (handles daylight saving), one file per month (`src/02b_weather.py`). 20,422 observations; longest gap 3.0 h.
- Feature definitions: `wx_thunder` = any weather code containing "TS" (includes VCTS, thunderstorm in the vicinity); `wx_precip` = any of RA, SN, DZ, PL, GR, GS, IC, UP; `wx_ceiling_ft` = lowest BKN/OVC/VV layer, 20,000 ft when none; `wx_gust_kt` = 0 when no gust reported; `wx_visibility_mi` as reported.
- Matching: latest observation at or before decision time (`merge_asof`, backward, exact matches allowed). The program fails if any matched observation is after decision time. Observations older than 3 h would be treated as missing; none were (median age at decision time 26 min). METAR reports are sent a few minutes after their observation time; this small lag is ignored.
- The model choice was made on validation only: (b) + weather had validation ROC-AUC 0.8000 vs 0.7950, PR-AUC 0.6313 vs 0.6190, Brier 0.1361 vs 0.1382, so it was chosen. Its validation calibration error was slightly worse (0.0085 vs 0.0078).
- Drift offset kept on for the chosen model (`config.USE_DRIFT_OFFSET = True`) because it improved calibration in the 18-month backtest for both models. On the test month it helped the weather model less (calibration error 0.0144 → 0.0135) than model (b) (0.0131 → 0.0094).
- The backtest (Phase 5 outputs and 7a before/after) now uses the chosen model, so `phase5_backtest.csv`, `backtest.png`, `phase7a_*` and `drift_offset.png` are for (b) + weather. Model (b)'s backtest numbers are in the 7a commit.
- Feature importance output is now for the chosen model.
- Phase 4 is rerun with `p_lgbm_swap_safe_weather_offset`. `phase7b_phase4_models.csv` compares all four versions at default settings. Across versions, minutes removed on actual outcomes is the fair comparison; expected minutes depend on each version's probability level.
