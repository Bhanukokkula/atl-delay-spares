"""Step 5: decide where to place spare aircraft in the test month.

For each day, three ways to assign S spare aircraft to flights:
  - optimization (PuLP integer program)
  - greedy: highest risk first, first free spare
  - random: random order, first free spare (averaged over several seeds)

Reads:  data/features/features.parquet   (training flights, for delay minutes by hour)
        data/model/predictions.parquet   (test-month predictions from model b)
Writes: outputs/phase4_hourly_delay_minutes.csv
        outputs/phase4_daily.csv          per-day results at the default settings
        outputs/phase4_summary.csv        comparison at the default settings
        outputs/phase4_sensitivity.csv    results for other S, R, H
        outputs/spares_by_s.png
"""
import os
import random
import numpy as np
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT_DIR = "outputs"

# Defaults (kept at Checkpoint 3; see DECISIONS.md).
S_DEFAULT = 3      # spare aircraft
R_DEFAULT = 0.6    # share of a flight's risk minutes a spare removes
H_DEFAULT = 4      # hours a spare is busy after it is used
SLOT_MIN = 15      # the busy rule is checked on 15-minute slots
N_RANDOM_SEEDS = 20
MIN_FLIGHTS_PER_HOUR = 30   # below this, use the overall average delay minutes

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#2b2b2b"
GRID = "#e4e3df"


def delay_minutes_by_hour(train):
    # Average delay minutes of delayed flights, by scheduled departure hour,
    # from training months only.
    delayed = train[train[config.LABEL] == 1]
    overall = delayed["dep_delay"].mean()
    rows = []
    for hour in range(24):
        group = delayed[delayed["dep_hour"] == hour]
        if len(group) >= MIN_FLIGHTS_PER_HOUR:
            minutes = group["dep_delay"].mean()
            source = "hour"
        else:
            minutes = overall
            source = "overall (few flights)"
        rows.append({"dep_hour": hour, "delayed_flights": len(group),
                     "avg_delay_min": round(minutes, 1), "source": source})
    return pd.DataFrame(rows)


def conflict(t1, t2, hours):
    """True if two departures (minutes after midnight) cannot share one spare.

    The rule: for every 15-minute slot t, a spare covers at most one flight
    departing in [t, t + H). Two flights clash if some slot start t lies in
    (later - H, earlier]. The optimization and both baselines use this same rule.
    """
    earlier = min(t1, t2)
    later = max(t1, t2)
    last_slot = (earlier // SLOT_MIN) * SLOT_MIN   # latest slot start <= earlier
    return last_slot > later - hours * 60


def assign_in_order(flights, order, n_spares, hours):
    # Walk through flights in the given order; give each the first spare that
    # does not clash with flights that spare already covers.
    spares = []
    for _ in range(n_spares):
        spares.append([])
    chosen = []
    for i in order:
        t = flights[i]["minute"]
        for covered in spares:
            free = True
            for other in covered:
                if conflict(t, other, hours):
                    free = False
                    break
            if free:
                covered.append(t)
                chosen.append(i)
                break
    return chosen


def solve_optimal(flights, n_spares, hours):
    problem = pulp.LpProblem("spares", pulp.LpMaximize)
    x = {}
    for s in range(n_spares):
        for i in range(len(flights)):
            x[(s, i)] = problem.add_variable(f"x_{s}_{i}", cat="Binary")

    # Objective: total risk minutes covered. R multiplies every term equally,
    # so it does not change which flights are chosen; it is applied afterwards.
    objective = []
    for (s, i), var in x.items():
        objective.append(flights[i]["risk"] * var)
    problem += pulp.lpSum(objective)

    # Each flight gets at most one spare.
    for i in range(len(flights)):
        problem += pulp.lpSum([x[(s, i)] for s in range(n_spares)]) <= 1

    # Each spare covers at most one flight departing in any window [t, t + H).
    window = hours * 60
    for t in range(0, 24 * 60, SLOT_MIN):
        inside = []
        for i in range(len(flights)):
            if t <= flights[i]["minute"] < t + window:
                inside.append(i)
        if len(inside) < 2:
            continue
        for s in range(n_spares):
            problem += pulp.lpSum([x[(s, i)] for i in inside]) <= 1

    result = problem.solve(pulp.getSolver("HiGHS", msg=False))
    if result.status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError(f"Solver status: {result.status}")

    chosen = []
    for (s, i), var in x.items():
        if var.value() > 0.5:
            chosen.append(i)
    return chosen


def outcome(flights, chosen, share_removed):
    expected = 0.0
    really_delayed = 0
    actual_minutes = 0.0
    for i in chosen:
        expected += share_removed * flights[i]["risk"]
        if flights[i]["delayed"] == 1:
            really_delayed += 1
            actual_minutes += share_removed * flights[i]["dep_delay"]
    return expected, really_delayed, actual_minutes


def run_month(days, n_spares, share_removed, hours):
    """Run all three methods on every test day. Returns one row per day and method."""
    rows = []
    for day, flights in days.items():
        # Flight indexes from highest to lowest risk.
        pairs = []
        for i in range(len(flights)):
            pairs.append((flights[i]["risk"], i))
        pairs.sort(reverse=True)
        by_risk = []
        for risk, i in pairs:
            by_risk.append(i)
        picks = {
            "optimization": solve_optimal(flights, n_spares, hours),
            "greedy": assign_in_order(flights, by_risk, n_spares, hours),
        }
        for method, chosen in picks.items():
            expected, delayed, actual = outcome(flights, chosen, share_removed)
            rows.append({"day": day, "method": method, "assigned": len(chosen),
                         "expected_min_removed": expected,
                         "assigned_really_delayed": delayed,
                         "actual_min_removed_if_R_holds": actual})

        # Random baseline, averaged over seeds.
        totals = [0.0, 0.0, 0.0, 0.0]
        for seed in range(N_RANDOM_SEEDS):
            order = list(range(len(flights)))
            random.Random(seed).shuffle(order)
            chosen = assign_in_order(flights, order, n_spares, hours)
            expected, delayed, actual = outcome(flights, chosen, share_removed)
            totals[0] += len(chosen)
            totals[1] += expected
            totals[2] += delayed
            totals[3] += actual
        rows.append({"day": day, "method": "random",
                     "assigned": totals[0] / N_RANDOM_SEEDS,
                     "expected_min_removed": totals[1] / N_RANDOM_SEEDS,
                     "assigned_really_delayed": totals[2] / N_RANDOM_SEEDS,
                     "actual_min_removed_if_R_holds": totals[3] / N_RANDOM_SEEDS})

        # Sanity check: the optimum can never be worse than greedy under the same rule.
        opt = rows[-3]["expected_min_removed"]
        greedy = rows[-2]["expected_min_removed"]
        if opt < greedy - 1e-6:
            raise RuntimeError(f"{day}: optimization {opt:.2f} below greedy {greedy:.2f}")
    return pd.DataFrame(rows)


def summarize(daily, n_spares, share_removed, hours):
    rows = []
    for method in ["optimization", "greedy", "random"]:
        d = daily[daily["method"] == method]
        assigned = d["assigned"].sum()
        rows.append({
            "S": n_spares, "R": share_removed, "H": hours, "method": method,
            "days": len(d),
            "expected_min_removed_per_day": round(d["expected_min_removed"].mean(), 1),
            "assignments_per_day": round(d["assigned"].mean(), 1),
            "share_assigned_really_delayed_pct": round(100 * d["assigned_really_delayed"].sum() / assigned, 1),
            "actual_min_removed_per_day_if_R_holds": round(d["actual_min_removed_if_R_holds"].mean(), 1),
        })
    summary = pd.DataFrame(rows)
    opt = summary.loc[summary["method"] == "optimization", "expected_min_removed_per_day"].iloc[0]
    pct = []
    for value in summary["expected_min_removed_per_day"]:
        pct.append(round(100 * value / opt, 1))
    summary["pct_of_optimization"] = pct
    return summary


def plot_by_s(sensitivity, path):
    rows = sensitivity[(sensitivity["R"] == R_DEFAULT) & (sensitivity["H"] == H_DEFAULT)]
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = [("optimization", "Optimization (PuLP)", BLUE, "o"),
              ("greedy", "Greedy (highest risk first)", ORANGE, "s"),
              ("random", "Random (avg of 20 runs)", AQUA, "^")]
    for method, label, color, marker in styles:
        d = rows[rows["method"] == method].sort_values("S")
        ax.plot(d["S"], d["expected_min_removed_per_day"], color=color, linewidth=2,
                marker=marker, markersize=7, label=label)
    ax.set_xlabel("Number of spare aircraft (S)", color=INK)
    ax.set_ylabel("Expected delay minutes removed per day", color=INK)
    ax.set_title(f"Spare placement, test month (R = {R_DEFAULT}, H = {H_DEFAULT} h)", color=INK, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    features = pd.read_parquet("data/features/features.parquet")
    train = features[features["split"] == "train"]
    hourly = delay_minutes_by_hour(train)
    hourly.to_csv(os.path.join(OUT_DIR, "phase4_hourly_delay_minutes.csv"), index=False)
    minutes_by_hour = dict(zip(hourly["dep_hour"], hourly["avg_delay_min"]))

    predictions = pd.read_parquet("data/model/predictions.parquet")
    test = predictions[predictions["split"] == "test"].copy()
    test["risk"] = test[config.CHOSEN_PREDICTION] * test["dep_hour"].map(minutes_by_hour)
    test["minute"] = (test["sched_dep"] - test["flight_date"]).dt.total_seconds() / 60
    print(f"Test flights: {len(test)}, days: {test['flight_date'].nunique()}, "
          f"mean risk minutes per flight: {test['risk'].mean():.2f}")

    # One list of plain dicts per day keeps the loops easy to read.
    days = {}
    for day, group in test.groupby("flight_date"):
        flights = []
        for _, row in group.iterrows():
            flights.append({"minute": row["minute"], "risk": row["risk"],
                            "delayed": int(row[config.LABEL]), "dep_delay": row["dep_delay"]})
        days[day.date().isoformat()] = flights

    # Default run.
    daily = run_month(days, S_DEFAULT, R_DEFAULT, H_DEFAULT)
    daily.to_csv(os.path.join(OUT_DIR, "phase4_daily.csv"), index=False)
    summary = summarize(daily, S_DEFAULT, R_DEFAULT, H_DEFAULT)
    summary.to_csv(os.path.join(OUT_DIR, "phase4_summary.csv"), index=False)
    print("\nDefault settings:")
    print(summary.to_string(index=False))
    print(f"Test-month delay rate for all flights: {100 * test[config.LABEL].mean():.1f}%")

    pivot = daily.pivot(index="day", columns="method", values="expected_min_removed")
    gap = pivot["optimization"] - pivot["greedy"]
    print(f"Days where optimization beat greedy by more than 0.1 min: {(gap > 0.1).sum()} of {len(gap)}; "
          f"largest daily gap: {gap.max():.2f} min")

    # Sensitivity. R only rescales the objective, so the chosen flights are the
    # same for every R; we still rerun so the table comes straight from outputs.
    results = []
    for n_spares in [1, 2, 3, 5, 8]:
        for share in [0.3, 0.6, 0.9]:
            results.append(summarize(run_month(days, n_spares, share, H_DEFAULT), n_spares, share, H_DEFAULT))
    for hours in [2, 6, 8]:
        results.append(summarize(run_month(days, S_DEFAULT, R_DEFAULT, hours), S_DEFAULT, R_DEFAULT, hours))
    sensitivity = pd.concat(results, ignore_index=True)
    sensitivity.to_csv(os.path.join(OUT_DIR, "phase4_sensitivity.csv"), index=False)
    print("\nSensitivity (expected minutes removed per day; % of optimization):")
    cols = ["S", "R", "H", "method", "expected_min_removed_per_day", "pct_of_optimization",
            "share_assigned_really_delayed_pct"]
    print(sensitivity[cols].to_string(index=False))

    plot_by_s(sensitivity, os.path.join(OUT_DIR, "spares_by_s.png"))


if __name__ == "__main__":
    main()
