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
- route_delay_rate is now out-of-fold by month for training flights (alone AUC 0.565 → 0.545 on the 12-month data, showing the in-sample version was optimistic).
- backlog_count excludes cancelled flights, because BTS does not record when a cancellation was announced.

## Checkpoint 3 — Model and recalibration (2026-11-14)

Questions: which model and feature set to carry forward — (a) all features, (b) all except turn_min/has_inbound/inbound_delay_known, (c) no inbound features — and whether to recalibrate.

Answer (word for word):
> LightGBM with option (b). Score (b) on test once now and report (a), (b), (c) side by side as the range. Use (b) in Phase 4, because a model that detects swaps would make spare-aircraft allocation look better than it is. No recalibration; note the 1.6-point under-prediction as a limitation. Now finish Phases 4–6 without stopping, using the defaults I sent earlier.

How this was applied (Claude's notes, not part of the answer):
- Checkpoint 4 is not held as a stop. Phase 4 uses the original defaults S = 3 spares, R = 0.6, H = 4 hours; the S and R sensitivity is still produced and reported.
- Option (b) scored on test once: ROC-AUC 0.7691, PR-AUC 0.5386, calibration error 0.0131. Its mean prediction is 21.2% vs 22.3% actual, so the chosen model under-predicts by 1.1 points; the 1.6-point figure was for option (a). The limitation note uses the (b) figure.
