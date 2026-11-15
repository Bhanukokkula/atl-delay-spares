"""Step 3: build model features that are known 2 hours before departure.

Reads:  data/clean/departures.parquet, data/clean/arrivals.parquet,
        data/clean/weather.parquet
Writes: data/features/features.parquet
        outputs/phase2_feature_summary.csv
        outputs/phase2_turn_buckets.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

CLEAN_DIR = "data/clean"
FEATURE_DIR = "data/features"
OUT_DIR = "outputs"

# Major US travel holidays covering the data window plus a margin on each side.
HOLIDAYS = [
    "2024-05-27", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-05-26", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25", "2026-01-01", "2026-05-25", "2026-07-04", "2026-09-07",
    "2026-11-26", "2026-12-25",
]


def add_split(dep):
    with open("data/raw/months.txt") as f:
        months = [line.strip() for line in f if line.strip()]
    n_holdout = config.N_VALID_MONTHS + config.N_TEST_MONTHS
    train = months[:-n_holdout]
    valid = months[-n_holdout:-config.N_TEST_MONTHS]
    dep["split"] = "test"
    dep.loc[dep["year_month"].isin(train), "split"] = "train"
    dep.loc[dep["year_month"].isin(valid), "split"] = "valid"
    print(f"Train months: {train[0]} to {train[-1]}  Valid: {valid}  Test: {months[-1]}")
    return dep


def add_calendar_features(dep):
    dep["dep_hour"] = dep["sched_dep"].dt.hour
    holidays = pd.to_datetime(pd.Series(HOLIDAYS))
    # Days to the nearest holiday, before or after.
    nearest = None
    for h in holidays:
        days = (dep["flight_date"] - h).dt.days.abs()
        if nearest is None:
            nearest = days
        else:
            nearest = np.minimum(nearest, days)
    dep["days_to_holiday"] = nearest
    return dep


def add_congestion(dep):
    # Count from the full schedule, including flights later cancelled,
    # because the schedule is what is known in advance.
    hour_key = dep["sched_dep"].dt.floor("h")
    dep["same_hour_departures"] = hour_key.map(hour_key.value_counts())
    return dep


def route_rates(flights):
    # Delay rate per destination. Small routes are pulled toward the overall
    # rate so a route with 5 flights does not get an extreme value.
    overall = flights[config.LABEL].mean()
    smoothing = 50
    rates = {}
    for dest, group in flights.groupby("dest"):
        n = len(group)
        delayed = group[config.LABEL].sum()
        rates[dest] = (delayed + smoothing * overall) / (n + smoothing)
    return rates, overall


def add_route_delay_rate(dep):
    # Validation and test flights get the rate from all training months, so
    # their own outcomes never feed into a feature.
    train = dep[dep["split"] == "train"]
    rates, overall = route_rates(train)
    dep["route_delay_rate"] = dep["dest"].map(rates).fillna(overall)

    # Training flights get an out-of-fold rate: computed from the other
    # training months only, so a flight never sees its own outcome. Folds are
    # whole months because flights on the same bad day are not independent.
    for ym in train["year_month"].unique():
        in_month = (dep["split"] == "train") & (dep["year_month"] == ym)
        others = train[train["year_month"] != ym]
        rates, overall = route_rates(others)
        dep.loc[in_month, "route_delay_rate"] = dep.loc[in_month, "dest"].map(rates).fillna(overall)
    return dep


def add_backlog(dep):
    # Number of DL ATL flights scheduled in the 2 hours before decision time
    # that had not left by decision time. This is the queue we can see.
    # Cancelled flights are left out: BTS does not say when a cancellation was
    # announced, so we cannot tell if it was known at decision time.
    flown = dep[dep["cancelled"] == 0]
    window = pd.Timedelta(hours=2)
    decision_all = (dep["sched_dep"] - pd.Timedelta(hours=config.DECISION_HOURS_BEFORE)).to_numpy()
    dates = dep["flight_date"].to_numpy()
    backlog = np.zeros(len(dep), dtype=int)

    # Go one day at a time so the comparison tables stay small.
    for day in dep["flight_date"].unique():
        targets = np.where(dates == day)[0]
        decision = decision_all[targets]

        # Candidate flights: anything scheduled from 2 hours before the
        # earliest decision time up to the latest decision time.
        lo = decision.min() - window
        hi = decision.max()
        near = flown[(flown["sched_dep"] >= lo) & (flown["sched_dep"] < hi)]
        sched = near["sched_dep"].to_numpy()
        actual = near["actual_dep"].to_numpy()

        # One row per target flight, one column per candidate flight.
        in_window = (sched[None, :] >= (decision[:, None] - window)) & (sched[None, :] < decision[:, None])
        not_left = actual[None, :] >= decision[:, None]
        backlog[targets] = (in_window & not_left).sum(axis=1)

    dep["backlog_count"] = backlog
    return dep


def turn_bucket_table(dep):
    # Delay rate by scheduled turn, on training flights only, so the cutoff is
    # not chosen by looking at validation or test outcomes.
    train = dep[(dep["split"] == "train") & (dep["has_inbound"] == 1)]
    edges = [0, 10, 20, 25, 30, 35, 40, 45, 50, 60, 90, 180, 1440]
    rows = []
    for i in range(len(edges) - 1):
        low = edges[i]
        high = edges[i + 1]
        group = train[(train["turn_min"] > low) & (train["turn_min"] <= high)]
        rows.append({
            "turn_min": f"{low}-{high}",
            "flights": len(group),
            "delay15_rate_pct": round(100 * group[config.LABEL].mean(), 1),
            "inbound_dep_delay_known_avg": round(group.loc[group["inbound_delay_known"] == 1, "inbound_dep_delay"].mean(), 1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "phase2_turn_buckets.csv"), index=False)
    print("\nDelay rate by scheduled turn (training flights with an inbound):")
    print(table.to_string(index=False))
    print(f"All training flights: {100 * dep.loc[dep['split'] == 'train', config.LABEL].mean():.1f}% delayed\n")


def apply_turn_cutoff(dep):
    if config.TURN_CUTOFF_MIN is None:
        print("TURN_CUTOFF_MIN not set yet; no cutoff applied.")
        return dep
    short = (dep["has_inbound"] == 1) & (dep["turn_min"] < config.TURN_CUTOFF_MIN)
    dep.loc[short, "has_inbound"] = 0
    dep.loc[short, "turn_min"] = -1
    dep.loc[short, "inbound_delay_known"] = 0
    dep.loc[short, "inbound_dep_delay"] = 0.0
    print(f"Turns under {config.TURN_CUTOFF_MIN} min treated as no reliable inbound: {int(short.sum())} flights")
    return dep


def add_inbound_features(dep, arr):
    # The inbound leg is the same tail's most recent DL arrival into ATL that is
    # scheduled before our departure on the same day.
    arr = arr[arr["tail"].notna()].copy()
    arr = arr.rename(columns={"dep_delay": "inbound_raw_delay",
                              "cancelled": "inbound_cancelled"})
    arr["inbound_sched_arr"] = arr["sched_arr"]
    arr = arr.sort_values("sched_arr")

    left = dep[dep["tail"].notna()].sort_values("sched_dep")
    matched = pd.merge_asof(
        left[["row_id", "tail", "sched_dep"]],
        arr[["tail", "sched_arr", "inbound_sched_arr", "sched_block_min",
             "inbound_raw_delay", "inbound_cancelled"]],
        left_on="sched_dep", right_on="sched_arr", by="tail",
        direction="backward", allow_exact_matches=False,
    )
    # Only keep inbound legs that land on the same calendar day.
    same_day = matched["inbound_sched_arr"].dt.normalize() == matched["sched_dep"].dt.normalize()
    matched.loc[~same_day, "inbound_sched_arr"] = pd.NaT

    matched["has_inbound"] = matched["inbound_sched_arr"].notna().astype(int)
    matched["turn_min"] = (matched["sched_dep"] - matched["inbound_sched_arr"]).dt.total_seconds() / 60

    # Estimate when the inbound actually departed, in ATL clock time, without
    # time zone tables: scheduled arrival - scheduled block + departure delay.
    est_dep = (matched["inbound_sched_arr"]
               - pd.to_timedelta(matched["sched_block_min"], unit="m")
               + pd.to_timedelta(matched["inbound_raw_delay"], unit="m"))
    decision_time = matched["sched_dep"] - pd.Timedelta(hours=config.DECISION_HOURS_BEFORE)
    # Known only if the inbound had actually left by decision time.
    # A cancelled inbound has no delay, so it stays unknown.
    known = (matched["has_inbound"] == 1) & est_dep.notna() & (est_dep <= decision_time)
    matched["inbound_delay_known"] = known.astype(int)
    matched["inbound_dep_delay"] = 0.0
    matched.loc[known, "inbound_dep_delay"] = matched.loc[known, "inbound_raw_delay"]

    keep = matched[["row_id", "has_inbound", "turn_min", "inbound_delay_known",
                    "inbound_dep_delay"]]
    dep = dep.merge(keep, on="row_id", how="left")
    dep["has_inbound"] = dep["has_inbound"].fillna(0).astype(int)
    dep["inbound_delay_known"] = dep["inbound_delay_known"].fillna(0).astype(int)
    dep["inbound_dep_delay"] = dep["inbound_dep_delay"].fillna(0.0)
    # No inbound means the aircraft overnighted in ATL. Use -1 so the value is
    # clearly separate from a real turn time.
    dep["turn_min"] = dep["turn_min"].fillna(-1)
    return dep


def add_recent_delay(dep):
    # Average departure delay of DL ATL flights that actually left in the
    # 2 hours before decision time. Their departure is already known.
    flown = dep[dep["cancelled"] == 0].sort_values("actual_dep")
    times = flown["actual_dep"].to_numpy()
    delays = flown["dep_delay"].to_numpy()
    # Running total so each window sum is one subtraction.
    running = np.concatenate([[0.0], np.cumsum(delays)])

    decision_time = dep["sched_dep"] - pd.Timedelta(hours=config.DECISION_HOURS_BEFORE)
    window_start = decision_time - pd.Timedelta(hours=2)
    start_idx = np.searchsorted(times, window_start.to_numpy(), side="left")
    end_idx = np.searchsorted(times, decision_time.to_numpy(), side="left")

    count = end_idx - start_idx
    total = running[end_idx] - running[start_idx]
    dep["recent_count"] = count
    dep["recent_known"] = (count > 0).astype(int)
    dep["recent_avg_delay"] = 0.0
    has_any = count > 0
    dep.loc[has_any, "recent_avg_delay"] = total[has_any] / count[has_any]
    return dep


def add_weather(dep):
    # Latest ATL observation at or before decision time (2 hours before
    # scheduled departure). Nothing observed after decision time can be used.
    weather = pd.read_parquet(os.path.join(CLEAN_DIR, "weather.parquet"))
    weather = weather.sort_values("obs_time")
    dep["decision_time"] = dep["sched_dep"] - pd.Timedelta(hours=config.DECISION_HOURS_BEFORE)
    dep = dep.sort_values("decision_time")
    dep = pd.merge_asof(dep, weather, left_on="decision_time", right_on="obs_time",
                        direction="backward", allow_exact_matches=True)

    # Leakage check: fail loudly if any observation is later than decision time.
    late = dep["obs_time"] > dep["decision_time"]
    if late.any():
        raise RuntimeError(f"LEAKAGE: {int(late.sum())} weather observations after decision time")

    # A very old observation means the station had a gap; treat it as missing.
    age_hours = (dep["decision_time"] - dep["obs_time"]).dt.total_seconds() / 3600
    stale = dep["obs_time"].isna() | (age_hours > config.WEATHER_MAX_AGE_HOURS)
    for col in config.WEATHER_FEATURES:
        dep.loc[stale, col] = np.nan
    median_age_min = 60 * age_hours.median()
    print(f"Weather: median observation age at decision time {median_age_min:.0f} min; "
          f"missing or older than {config.WEATHER_MAX_AGE_HOURS} h: {int(stale.sum())} flights")
    dep = dep.drop(columns=["decision_time", "obs_time"])
    # Put rows back in their original order so every other step is unchanged.
    return dep.sort_values("row_id").reset_index(drop=True)


def single_feature_auc_check(dep):
    # Second leakage guard: no single feature should predict the label almost
    # perfectly. If one does, it is probably measuring the outcome.
    train = dep[dep["split"] == "train"]
    rows = []
    for col in config.NUMERIC_FEATURES + config.WEATHER_FEATURES:
        values = train[col].fillna(train[col].median())
        auc = roc_auc_score(train[config.LABEL], values)
        strength = max(auc, 1 - auc)
        rows.append({"feature": col, "single_feature_auc": round(strength, 3)})
        if strength > 0.9:
            raise RuntimeError(f"LEAKAGE SUSPECTED: '{col}' alone has AUC {strength:.3f}")
    return pd.DataFrame(rows)


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    dep = pd.read_parquet(os.path.join(CLEAN_DIR, "departures.parquet"))
    arr = pd.read_parquet(os.path.join(CLEAN_DIR, "arrivals.parquet"))
    dep["row_id"] = np.arange(len(dep))

    dep = add_split(dep)
    dep = add_calendar_features(dep)
    dep = add_congestion(dep)          # uses the full schedule, before dropping
    dep = add_recent_delay(dep)        # uses all flights that departed
    dep = add_backlog(dep)
    dep = add_inbound_features(dep, arr)
    dep = add_weather(dep)

    # Checkpoint 1 decision (Option A): cancelled flights are not scored.
    n_before = len(dep)
    dep = dep[dep["cancelled"] == 0].copy()
    print(f"Dropped {n_before - len(dep)} cancelled departures, {len(dep)} remain")

    turn_bucket_table(dep)
    dep = apply_turn_cutoff(dep)
    dep = add_route_delay_rate(dep)

    config.check_no_leakage(config.FEATURES + config.WEATHER_FEATURES)
    auc_table = single_feature_auc_check(dep)

    # Keep ids, label, and dep_delay (needed in Phase 4 for delay minutes; it is
    # NOT a feature and check_no_leakage blocks it from the model).
    keep = (["row_id", "year_month", "split", "flight_date", "flight_number", "tail",
             "sched_dep", "dep_delay", config.LABEL] + config.FEATURES + config.WEATHER_FEATURES)
    dep = dep[keep]
    dep.to_parquet(os.path.join(FEATURE_DIR, "features.parquet"), index=False)

    # Summary for the checkpoint.
    summary = auc_table.copy()
    means = []
    for col in config.NUMERIC_FEATURES + config.WEATHER_FEATURES:
        means.append(round(dep[col].mean(), 3))
    summary["mean"] = means
    summary.to_csv(os.path.join(OUT_DIR, "phase2_feature_summary.csv"), index=False)
    print()
    print(summary.to_string(index=False))
    print()
    print("Rows per split:", dep["split"].value_counts().to_dict())
    print(f"Flights with a same-day inbound leg: {100 * dep['has_inbound'].mean():.1f}%")
    with_inbound = dep[dep["has_inbound"] == 1]
    print(f"  of those, inbound delay known at decision time: "
          f"{100 * with_inbound['inbound_delay_known'].mean():.1f}%")
    print(f"  median scheduled turn: {with_inbound['turn_min'].median():.0f} min, "
          f"turns under 30 min: {(with_inbound['turn_min'] < 30).sum()}")
    print(f"Flights with at least one departure in the recent 2h window: "
          f"{100 * dep['recent_known'].mean():.1f}%")
    print(f"Distinct destinations: {dep['dest'].nunique()}")
    print(f"Label rate by split: "
          f"{dep.groupby('split')[config.LABEL].mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
