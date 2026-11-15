"""Step 9: precompute everything the static demo in docs/ needs.

For every test-month day, every S in 1-5 and every H in 2-8 hours, run the
greedy rule and the optimization from step 5 and save which flights get a
spare. The web page only looks results up, so it needs no server.

Reads:  data/model/predictions.parquet, data/features/features.parquet,
        outputs/phase4_hourly_delay_minutes.csv, outputs/phase4_summary.csv,
        outputs/phase4_sensitivity.csv, outputs/phase3_results.csv,
        outputs/phase7c_bootstrap.csv, outputs/brief.pdf
Writes: docs/demo_data.json, docs/preview.png, docs/brief.pdf
"""
import importlib
import json
import os
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Reuse the exact spare-assignment code from step 5. Its file name starts with
# a digit, so it is loaded with importlib instead of a normal import.
spares = importlib.import_module("05_spares")

S_VALUES = [1, 2, 3, 4, 5]
H_VALUES = [2, 3, 4, 5, 6, 7, 8]
DOCS = "docs"
MODEL_LABELS = {
    "lgbm_swap_safe": "LightGBM, swap-safe features",
    "lgbm_swap_safe_weather": "LightGBM, swap-safe features + ATL weather",
}


def load_test_flights():
    predictions = pd.read_parquet("data/model/predictions.parquet")
    test = predictions[predictions["split"] == "test"].copy()
    features = pd.read_parquet("data/features/features.parquet", columns=["row_id", "dest"])
    test = test.merge(features, on="row_id", how="left")
    # Same risk minutes as step 5: probability x hourly average delay minutes.
    hourly = pd.read_csv("outputs/phase4_hourly_delay_minutes.csv")
    minutes_by_hour = dict(zip(hourly["dep_hour"], hourly["avg_delay_min"]))
    test["p"] = test[config.CHOSEN_PREDICTION]
    test["risk"] = test["p"] * test["dep_hour"].map(minutes_by_hour)
    test["minute"] = (test["sched_dep"] - test["flight_date"]).dt.total_seconds() / 60
    return test


def number_spares(chosen, flights, n_spares, hours):
    """Give each chosen flight a spare number (1..S), for display only.

    Walk the chosen flights in time order and give each the first spare that
    is free. Each flight blocks a run of consecutive 15-minute slots, so this
    always succeeds for a plan that obeys the busy rule.
    """
    pairs = []
    for i in chosen:
        pairs.append((flights[i]["minute"], i))
    pairs.sort()
    used = []
    for _ in range(n_spares):
        used.append([])
    numbered = []
    for minute, i in pairs:
        number = 0
        for s in range(n_spares):
            free = True
            for other in used[s]:
                if spares.conflict(minute, other, hours):
                    free = False
                    break
            if free:
                used[s].append(minute)
                number = s + 1
                break
        if number == 0:
            raise RuntimeError("Could not number the spares: plan breaks the busy rule")
        numbered.append([int(i), number])
    return numbered


def run_day(group):
    # Flights in risk order, so the page can use the list position as the rank.
    group = group.sort_values("risk", ascending=False).reset_index(drop=True)
    flights = []
    rows = []
    for _, row in group.iterrows():
        flights.append({"minute": row["minute"], "risk": row["risk"],
                        "delayed": int(row[config.LABEL]), "dep_delay": row["dep_delay"]})
        rows.append([int(row["minute"]), int(row["flight_number"]), row["dest"],
                     round(float(row["p"]), 3), round(float(row["risk"]), 2), int(row[config.LABEL])])
    by_risk = list(range(len(flights)))   # already highest risk first

    runs = {}
    for n_spares in S_VALUES:
        for hours in H_VALUES:
            run = {}
            picks = {
                "greedy": spares.assign_in_order(flights, by_risk, n_spares, hours),
                "opt": spares.solve_optimal(flights, n_spares, hours),
            }
            for method, chosen in picks.items():
                expected, delayed, _ = spares.outcome(flights, chosen, spares.R_DEFAULT)
                run[method] = number_spares(chosen, flights, n_spares, hours)
                run[method + "_min"] = round(expected, 2)
                run[method + "_hits"] = int(delayed)
            runs[f"S{n_spares}_H{hours}"] = run
    return rows, runs


def model_label():
    label = MODEL_LABELS[config.CHOSEN_MODEL]
    if config.USE_DRIFT_OFFSET:
        label = label + ", with a daily correction fitted on the previous 7 days"
    return label


def month_results():
    # Short lines for the page footer, all read from step outputs.
    results = pd.read_csv("outputs/phase3_results.csv")
    model = config.CHOSEN_NAME
    row = results[(results["model"] == model) & (results["split"] == "test")].iloc[0]
    summary = pd.read_csv("outputs/phase4_summary.csv").set_index("method")
    boot = pd.read_csv("outputs/phase7c_bootstrap.csv").set_index("metric")

    def ci(metric, digits):
        low = boot.loc[metric, "ci95_low"]
        high = boot.loc[metric, "ci95_high"]
        return f"{low:.{digits}f} to {high:.{digits}f}"

    return [
        f"Whole test month: model ROC-AUC {row['roc_auc']:.3f} (95% interval {ci('test_roc_auc', 3)}).",
        f"With 3 spares and H = 4 h, greedy reached {summary.loc['greedy', 'pct_of_optimization']:.1f}% "
        f"of the optimization's expected minutes (95% interval {ci('greedy_pct_of_optimal', 1)}%).",
        f"Flights given a spare were really delayed {summary.loc['optimization', 'share_assigned_really_delayed_pct']:.1f}% "
        f"of the time (optimization) vs {summary.loc['random', 'share_assigned_really_delayed_pct']:.1f}% for random picks.",
    ]


def preview_image(path):
    # 1200 x 627 image for link previews (LinkedIn, Slack, etc.).
    sens = pd.read_csv("outputs/phase4_sensitivity.csv")
    summary = pd.read_csv("outputs/phase4_summary.csv").set_index("method")
    rows = sens[(sens["R"] == spares.R_DEFAULT) & (sens["H"] == spares.H_DEFAULT)]

    fig = plt.figure(figsize=(12, 6.27), dpi=100, facecolor="#fcfcfb")
    fig.text(0.05, 0.87, "Delay risk and spare aircraft placement at ATL", fontsize=27,
             fontweight="bold", color="#0b0b0b")
    fig.text(0.05, 0.80, "Predicting 15+ min departure delays 2 hours ahead, then placing spare aircraft",
             fontsize=16, color="#52514e")
    fig.text(0.05, 0.04, "Independent demo on public BTS data. Not affiliated with Delta.",
             fontsize=15, color="#5a4300", fontweight="bold")
    greedy_pct = summary.loc["greedy", "pct_of_optimization"]
    fig.text(0.62, 0.55, f"Greedy reaches\n{greedy_pct:.1f}% of optimal", fontsize=26,
             color="#0b0b0b", va="center")
    fig.text(0.62, 0.38, "Test month, 3 spares, H = 4 h", fontsize=15, color="#52514e")

    ax = fig.add_axes([0.08, 0.22, 0.47, 0.50])
    styles = [("optimization", "Optimization", "#2a78d6", "o"),
              ("greedy", "Greedy", "#eb6834", "s"),
              ("random", "Random", "#1baf7a", "^")]
    for method, label, color, marker in styles:
        d = rows[rows["method"] == method].sort_values("S")
        ax.plot(d["S"], d["expected_min_removed_per_day"], color=color, linewidth=3,
                marker=marker, markersize=9, label=label)
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel("Spare aircraft", fontsize=13, color="#52514e")
    ax.set_ylabel("Expected minutes saved / day", fontsize=13, color="#52514e")
    ax.set_ylim(bottom=0)
    ax.grid(color="#e4e3df", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    os.makedirs(DOCS, exist_ok=True)
    test = load_test_flights()

    days = []
    for day, group in test.groupby("flight_date"):
        rows, runs = run_day(group)
        days.append({
            "date": day.date().isoformat(),
            "label": day.strftime("%a %d %b %Y"),
            "n_flights": len(rows),
            "n_delayed": int(group[config.LABEL].sum()),
            "flights": rows,
            "runs": runs,
        })
        print(f"{day.date()}: {len(rows)} flights, {len(runs)} settings")

    # Check: the default setting must match step 5 exactly.
    daily = pd.read_csv("outputs/phase4_daily.csv")
    key = f"S{spares.S_DEFAULT}_H{spares.H_DEFAULT}"
    for d in days:
        for method, name in [("greedy", "greedy"), ("opt", "optimization")]:
            step5 = daily[(daily["day"] == d["date"]) & (daily["method"] == name)]
            expected = step5["expected_min_removed"].iloc[0]
            if abs(d["runs"][key][method + "_min"] - expected) > 0.01:
                raise RuntimeError(f"{d['date']} {name}: demo {d['runs'][key][method + '_min']} vs step 5 {expected}")
    print("Default setting matches outputs/phase4_daily.csv for every day.")

    meta = {
        "test_month": test["year_month"].iloc[0],
        "model_label": model_label(),
        "uses_weather": "weather" in config.CHOSEN_MODEL,
        "R": spares.R_DEFAULT,
        "s_values": S_VALUES,
        "h_values": H_VALUES,
        "month_results": month_results(),
    }
    with open(os.path.join(DOCS, "demo_data.json"), "w") as f:
        json.dump({"meta": meta, "days": days}, f, separators=(",", ":"))
    size_kb = os.path.getsize(os.path.join(DOCS, "demo_data.json")) / 1e3
    print(f"Wrote docs/demo_data.json ({size_kb:.0f} KB)")

    preview_image(os.path.join(DOCS, "preview.png"))
    shutil.copy("outputs/brief.pdf", os.path.join(DOCS, "brief.pdf"))
    print("Wrote docs/preview.png and copied outputs/brief.pdf to docs/brief.pdf")


if __name__ == "__main__":
    main()
