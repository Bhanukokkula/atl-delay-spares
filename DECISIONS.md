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
