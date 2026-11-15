"""Step 8: 95% intervals by resampling days of the test month.

Flights on the same day share weather and knock-on delays, so they are not
independent. We therefore resample whole days (with replacement), 1,000 times,
and recompute each number on every resample.

Reads:  data/model/predictions.parquet   (test-month predictions)
        outputs/phase4_daily.csv          (per-day spare results, default settings)
Writes: outputs/phase7c_bootstrap.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

N_RESAMPLES = 1000
SEED = 42


def interval(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def main():
    predictions = pd.read_parquet("data/model/predictions.parquet")
    test = predictions[predictions["split"] == "test"]
    daily = pd.read_csv("outputs/phase4_daily.csv")

    # Per-day arrays so each resample is just a lookup.
    days = sorted(daily["day"].unique())
    y_by_day = {}
    p_by_day = {}
    for day, group in test.groupby("flight_date"):
        key = day.date().isoformat()
        y_by_day[key] = group[config.LABEL].to_numpy()
        p_by_day[key] = group[config.CHOSEN_PREDICTION].to_numpy()

    spare = {}
    for method in ["optimization", "greedy", "random"]:
        d = daily[daily["method"] == method].set_index("day")
        spare[method] = d

    def metrics(sample_days):
        y = np.concatenate([y_by_day[d] for d in sample_days])
        p = np.concatenate([p_by_day[d] for d in sample_days])
        result = {"test_roc_auc": roc_auc_score(y, p)}
        hit = {}
        for method in ["optimization", "greedy", "random"]:
            rows = spare[method].loc[sample_days]
            hit[method] = 100 * rows["assigned_really_delayed"].sum() / rows["assigned"].sum()
        result["hit_rate_optimization_pct"] = hit["optimization"]
        result["hit_rate_greedy_pct"] = hit["greedy"]
        result["hit_rate_random_pct"] = hit["random"]
        result["hit_rate_optimization_minus_random_pts"] = hit["optimization"] - hit["random"]
        result["hit_rate_greedy_minus_random_pts"] = hit["greedy"] - hit["random"]
        opt = spare["optimization"].loc[sample_days, "expected_min_removed"].sum()
        greedy = spare["greedy"].loc[sample_days, "expected_min_removed"].sum()
        result["greedy_pct_of_optimal"] = 100 * greedy / opt
        return result

    point = metrics(days)

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_RESAMPLES):
        sample = list(rng.choice(days, size=len(days), replace=True))
        draws.append(metrics(sample))
    draws = pd.DataFrame(draws)

    rows = []
    for name in point:
        low, high = interval(draws[name])
        rows.append({"metric": name, "estimate": round(point[name], 4),
                     "ci95_low": round(low, 4), "ci95_high": round(high, 4)})
    table = pd.DataFrame(rows)
    table.to_csv("outputs/phase7c_bootstrap.csv", index=False)
    print(f"Test month: {len(days)} days, {N_RESAMPLES} day-level resamples, "
          f"model column {config.CHOSEN_PREDICTION}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
