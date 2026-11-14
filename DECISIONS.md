# Decisions log

Each checkpoint records the question asked and my answer word for word, with the date.

## Checkpoint 1 — Treatment of cancelled flights (2026-11-14)

Question: Drop cancelled flights from training and scoring (Option A), or count them as disrupted with label = 1 (Option B)?

Answer (word for word):
> option A

Effect: cancelled departures are removed before modelling. They still count toward the scheduled-departures-per-hour congestion feature, because the schedule is known in advance.
