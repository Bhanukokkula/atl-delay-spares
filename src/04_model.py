"""Step 4: train and evaluate delay-risk models.

Reads:  data/features/features.parquet
Writes: data/model/predictions.parquet      predictions for validation + test flights
        outputs/phase3_results.csv           ROC-AUC, PR-AUC, calibration per model
        outputs/phase3_experiments.csv       feature-group experiments (what helped, what didn't)
        outputs/phase3_feature_importance.csv
        outputs/calibration.png
        outputs/feature_importance.png
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import config
import drift

FEATURE_DIR = "data/features"
MODEL_DIR = "data/model"
OUT_DIR = "outputs"

# Chart colors (validated categorical palette, fixed order).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#2b2b2b"
GRID = "#e4e3df"

# Models that also get the daily drift offset (Phase 7a).
OFFSET_MODELS = ["lgbm_swap_safe", "lgbm_swap_safe_weather"]

# Feature groups, used to test what each group adds.
GROUPS = {
    "calendar": ["dep_hour", "day_of_week", "days_to_holiday"],
    "route": ["dest", "distance", "sched_block_min", "route_delay_rate"],
    "congestion": ["same_hour_departures"],
    "inbound": config.INBOUND_FEATURES,
    "recent_departures": ["recent_known", "recent_avg_delay", "recent_count"],
    "backlog": ["backlog_count"],
}


def calibration_error(y, p, n_bins=10):
    # Expected calibration error: average gap between predicted and actual
    # delay rate, over 10 equal-size bins of predicted probability.
    order = np.argsort(p)
    y_sorted = np.asarray(y)[order]
    p_sorted = np.asarray(p)[order]
    bins_y = np.array_split(y_sorted, n_bins)
    bins_p = np.array_split(p_sorted, n_bins)
    total = 0.0
    for by, bp in zip(bins_y, bins_p):
        total += len(by) * abs(by.mean() - bp.mean())
    return total / len(y)


def score(name, split, y, p):
    return {
        "model": name,
        "split": split,
        "flights": len(y),
        "actual_delay_rate": round(float(np.mean(y)), 3),
        "mean_predicted": round(float(np.mean(p)), 3),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "calibration_error": round(calibration_error(y, p), 4),
    }


def fit_logistic(train, others, features):
    # Logistic regression uses numeric features only. The destination is
    # represented by route_delay_rate instead of 143 one-hot columns.
    numeric = []
    for col in features:
        if col != "dest":
            numeric.append(col)
    config.check_no_leakage(numeric)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[numeric])
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train[config.LABEL])
    predictions = []
    for df in others:
        predictions.append(model.predict_proba(scaler.transform(df[numeric]))[:, 1])
    return predictions


def prepare_lgbm(df, features, dest_categories):
    x = df[features].copy()
    if "dest" in features:
        # Same category list for every split; unseen destinations become missing.
        unseen = ~x["dest"].isin(dest_categories)
        x.loc[unseen, "dest"] = None
        x["dest"] = pd.Categorical(x["dest"], categories=dest_categories)
    return x


def fit_lgbm(train, valid, others, features):
    config.check_no_leakage(features)
    dest_categories = sorted(train["dest"].unique())
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,   # bigger leaves: fewer, steadier splits
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    # Validation is used for early stopping, so test stays untouched.
    model.fit(
        prepare_lgbm(train, features, dest_categories), train[config.LABEL],
        eval_X=prepare_lgbm(valid, features, dest_categories),
        eval_y=valid[config.LABEL],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    predictions = []
    for df in others:
        predictions.append(model.predict_proba(prepare_lgbm(df, features, dest_categories))[:, 1])
    return model, predictions


def calibration_points(y, p, n_bins=10):
    order = np.argsort(p)
    bins_y = np.array_split(np.asarray(y)[order], n_bins)
    bins_p = np.array_split(np.asarray(p)[order], n_bins)
    xs = []
    ys = []
    for by, bp in zip(bins_y, bins_p):
        xs.append(bp.mean())
        ys.append(by.mean())
    return xs, ys


def plot_calibration(test, columns, labels, colors, markers, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#9a9a96", linewidth=1, linestyle="--", label="Perfect calibration")
    for col, label, color, marker in zip(columns, labels, colors, markers):
        xs, ys = calibration_points(test[config.LABEL], test[col])
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7, label=label)
    ax.set_xlabel("Predicted probability of 15+ min delay (bin average)", color=INK)
    ax.set_ylabel("Actual share delayed", color=INK)
    ax.set_title("Calibration on test month (Aug 2026), 10 equal-size bins", color=INK, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance, path):
    top = importance.sort_values("gain_share", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["gain_share"] * 100, color=BLUE, height=0.6)
    for i, value in enumerate(top["gain_share"] * 100):
        ax.text(value + 0.5, i, f"{value:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Share of total split gain (%)", color=INK)
    ax.set_title(f"LightGBM feature importance ({config.CHOSEN_MODEL})", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(FEATURE_DIR, "features.parquet"))
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()
    print(f"Train {len(train)}  Valid {len(valid)}  Test {len(test)}")

    no_inbound = []
    for col in config.FEATURES:
        if col not in config.INBOUND_FEATURES:
            no_inbound.append(col)
    swap_safe = []
    for col in config.FEATURES:
        if col not in config.SWAP_EXPOSED:
            swap_safe.append(col)

    results = []

    # 1) Logistic regression baseline, both feature sets.
    for name, features in [("logistic_all", config.FEATURES), ("logistic_no_inbound", no_inbound)]:
        p_valid, p_test = fit_logistic(train, [valid, test], features)
        valid["p_" + name] = p_valid
        test["p_" + name] = p_test
        results.append(score(name, "valid", valid[config.LABEL], p_valid))
        results.append(score(name, "test", test[config.LABEL], p_test))

    # 2) LightGBM, three feature sets: (a) all, (b) swap-safe, (c) no inbound.
    main_model = None
    swap_safe_weather = swap_safe + config.WEATHER_FEATURES   # Phase 7b
    lgbm_runs = [("lgbm_all", config.FEATURES),
                 ("lgbm_swap_safe", swap_safe),
                 ("lgbm_no_inbound", no_inbound),
                 ("lgbm_swap_safe_weather", swap_safe_weather)]
    features_by_model = {}
    for name, features in lgbm_runs:
        model, (p_valid, p_test) = fit_lgbm(train, valid, [valid, test], features)
        print(f"{name}: {model.best_iteration_} trees")
        valid["p_" + name] = p_valid
        test["p_" + name] = p_test
        results.append(score(name, "valid", valid[config.LABEL], p_valid))
        results.append(score(name, "test", test[config.LABEL], p_test))
        features_by_model[name] = features
        if name == config.CHOSEN_MODEL:
            main_model = model

    # 2b) Daily drift offset (Phase 7a). Each day's predictions are shifted by
    # one log-odds offset fitted on the 7 completed days before it. For test
    # days those 7 days are validation or earlier test days, all out-of-sample.
    # Only the test month is scored: early validation days would need training
    # days, where predictions are in-sample.
    both = pd.concat([valid, test])
    for name in OFFSET_MODELS:
        adjusted, offsets = drift.apply_offsets(both, "p_" + name, config.LABEL)
        both["p_" + name + "_offset"] = adjusted
        valid["p_" + name + "_offset"] = both.loc[valid.index, "p_" + name + "_offset"]
        test["p_" + name + "_offset"] = both.loc[test.index, "p_" + name + "_offset"]
        results.append(score(name + "_offset", "test", test[config.LABEL], test["p_" + name + "_offset"]))
        test_days = offsets[offsets["flight_date"].isin(test["flight_date"].unique())].copy()
        test_days["model"] = name
        test_days["offset"] = test_days["offset"].round(4)
        test_days.to_csv(os.path.join(OUT_DIR, f"phase7a_test_offsets_{name}.csv"), index=False)
        print(f"{name}: test days with full 7-day history: {int(test_days['has_history'].sum())} of {len(test_days)}")

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(OUT_DIR, "phase3_results.csv"), index=False)
    print()
    print(results.to_string(index=False))

    # 3) Feature importance (gain) for the chosen model.
    gain = main_model.booster_.feature_importance(importance_type="gain")
    importance = pd.DataFrame({"feature": features_by_model[config.CHOSEN_MODEL], "gain": gain})
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()
    importance = importance.sort_values("gain_share", ascending=False)
    importance.to_csv(os.path.join(OUT_DIR, "phase3_feature_importance.csv"), index=False)
    print()
    print(importance[["feature", "gain_share"]].round(3).to_string(index=False))
    plot_importance(importance, os.path.join(OUT_DIR, "feature_importance.png"))

    # 4) Experiments: drop one feature group at a time from LightGBM (a) and
    # measure validation ROC-AUC. Validation only, so test is not used to choose.
    base_auc = roc_auc_score(valid[config.LABEL], valid["p_lgbm_all"])
    # One extra test: keep only the inbound's own delay and drop the turn time
    # and flags, which are the inbound features most exposed to aircraft swaps.
    to_drop = {}
    for group, cols in GROUPS.items():
        to_drop[f"drop {group}"] = cols
    to_drop["drop turn_min, has_inbound, inbound_delay_known"] = [
        "turn_min", "has_inbound", "inbound_delay_known"]

    experiments = []
    for experiment, cols in to_drop.items():
        features = []
        for col in config.FEATURES:
            if col not in cols:
                features.append(col)
        _, (p_valid,) = fit_lgbm(train, valid, [valid], features)
        auc = roc_auc_score(valid[config.LABEL], p_valid)
        experiments.append({
            "experiment": experiment,
            "valid_roc_auc": round(auc, 4),
            "change_vs_all_features": round(auc - base_auc, 4),
            "helped": "yes" if auc < base_auc - 0.001 else "no",
        })
    experiments = pd.DataFrame(experiments)
    experiments.to_csv(os.path.join(OUT_DIR, "phase3_experiments.csv"), index=False)
    print()
    print(f"Validation ROC-AUC with all features: {base_auc:.4f}")
    print(experiments.to_string(index=False))

    # 5) Calibration plot on the test month.
    plot_calibration(
        test,
        ["p_lgbm_all", "p_lgbm_swap_safe", "p_lgbm_no_inbound"],
        ["(a) LightGBM, all features", "(b) LightGBM, swap-safe (chosen)",
         "(c) LightGBM, no inbound features"],
        [BLUE, ORANGE, AQUA],
        ["o", "s", "^"],
        os.path.join(OUT_DIR, "calibration.png"),
    )

    keep = (["row_id", "split", "year_month", "flight_date", "flight_number", "tail",
             "sched_dep", "dep_hour", "dep_delay", config.LABEL,
             "p_logistic_all", "p_logistic_no_inbound", "p_lgbm_all", "p_lgbm_swap_safe", "p_lgbm_no_inbound", "p_lgbm_swap_safe_weather"])
    for name in OFFSET_MODELS:
        keep.append("p_" + name + "_offset")
    predictions = pd.concat([valid[keep], test[keep]], ignore_index=True)
    predictions.to_parquet(os.path.join(MODEL_DIR, "predictions.parquet"), index=False)
    print("\nSaved predictions and charts.")


if __name__ == "__main__":
    main()
