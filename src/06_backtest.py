"""Step 6: month-by-month backtest of the chosen model (b, swap-safe features).

For each month m: train on all months before m, test on month m.
The route delay rate is recomputed inside every fold from earlier months only,
because the version in features.parquet was built with later months too.

Reads:  data/features/features.parquet
Writes: outputs/phase5_backtest.csv
        outputs/backtest.png
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import config

OUT_DIR = "outputs"
MIN_TRAIN_MONTHS = 6     # first backtest month needs at least this much history
N_TREES = 200            # early stopping chose 198 for model (b) in step 4

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#2b2b2b"
GRID = "#e4e3df"


def calibration_error(y, p, n_bins=10):
    # Same definition as step 4: weighted gap over 10 equal-size bins.
    order = np.argsort(p)
    bins_y = np.array_split(np.asarray(y)[order], n_bins)
    bins_p = np.array_split(np.asarray(p)[order], n_bins)
    total = 0.0
    for by, bp in zip(bins_y, bins_p):
        total += len(by) * abs(by.mean() - bp.mean())
    return total / len(y)


def route_rates(flights):
    # Same smoothing as step 3.
    overall = flights[config.LABEL].mean()
    smoothing = 50
    rates = {}
    for dest, group in flights.groupby("dest"):
        rates[dest] = (group[config.LABEL].sum() + smoothing * overall) / (len(group) + smoothing)
    return rates, overall


def fold_route_rate(train, test):
    # Test month: rate from all training months.
    rates, overall = route_rates(train)
    test["route_delay_rate"] = test["dest"].map(rates).fillna(overall)
    # Training months: out-of-fold by month, as in step 3.
    for ym in train["year_month"].unique():
        in_month = train["year_month"] == ym
        rates, overall = route_rates(train[~in_month])
        train.loc[in_month, "route_delay_rate"] = train.loc[in_month, "dest"].map(rates).fillna(overall)
    return train, test


def prepare(df, features, dest_categories):
    x = df[features].copy()
    unseen = ~x["dest"].isin(dest_categories)
    x.loc[unseen, "dest"] = None
    x["dest"] = pd.Categorical(x["dest"], categories=dest_categories)
    return x


def plot_backtest(table, path):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    months = table["month"]

    axes[0].plot(months, table["roc_auc"], color=BLUE, linewidth=2, marker="o", markersize=6)
    axes[0].set_ylabel("ROC-AUC", color=INK)
    axes[0].set_title("Model (b) backtest: train on all earlier months, test on each month",
                      color=INK, fontsize=11)

    axes[1].plot(months, table["calibration_error"], color=BLUE, linewidth=2, marker="o", markersize=6)
    axes[1].set_ylabel("Calibration error", color=INK)
    axes[1].set_ylim(bottom=0)

    axes[2].plot(months, 100 * table["actual_delay_rate"], color=INK, linewidth=2,
                 marker="o", markersize=6, label="Actual")
    axes[2].plot(months, 100 * table["mean_predicted"], color=ORANGE, linewidth=2,
                 marker="s", markersize=6, label="Mean predicted")
    axes[2].set_ylabel("15+ min delay rate (%)", color=INK)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_parquet("data/features/features.parquet")
    features = []
    for col in config.FEATURES:
        if col not in config.SWAP_EXPOSED:
            features.append(col)
    config.check_no_leakage(features)

    months = sorted(df["year_month"].unique())
    rows = []
    for m_index in range(MIN_TRAIN_MONTHS, len(months)):
        month = months[m_index]
        train = df[df["year_month"] < month].copy()
        test = df[df["year_month"] == month].copy()
        train, test = fold_route_rate(train, test)

        dest_categories = sorted(train["dest"].unique())
        model = lgb.LGBMClassifier(
            n_estimators=N_TREES, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        model.fit(prepare(train, features, dest_categories), train[config.LABEL])
        p = model.predict_proba(prepare(test, features, dest_categories))[:, 1]
        y = test[config.LABEL]
        rows.append({
            "month": month,
            "train_months": m_index,
            "test_flights": len(test),
            "actual_delay_rate": round(float(y.mean()), 3),
            "mean_predicted": round(float(p.mean()), 3),
            "roc_auc": round(roc_auc_score(y, p), 4),
            "calibration_error": round(calibration_error(y, p), 4),
        })
        print(rows[-1])

    table = pd.DataFrame(rows)
    table["predicted_minus_actual_pts"] = round(100 * (table["mean_predicted"] - table["actual_delay_rate"]), 1)
    table.to_csv(os.path.join(OUT_DIR, "phase5_backtest.csv"), index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nROC-AUC: mean {table['roc_auc'].mean():.3f}, "
          f"min {table['roc_auc'].min():.3f} ({table.loc[table['roc_auc'].idxmin(), 'month']}), "
          f"max {table['roc_auc'].max():.3f} ({table.loc[table['roc_auc'].idxmax(), 'month']})")
    print(f"Calibration error: mean {table['calibration_error'].mean():.4f}, "
          f"max {table['calibration_error'].max():.4f} "
          f"({table.loc[table['calibration_error'].idxmax(), 'month']})")
    corr = np.corrcoef(table["actual_delay_rate"], table["predicted_minus_actual_pts"])[0, 1]
    print(f"Correlation between month delay rate and (predicted - actual): {corr:.2f}")
    plot_backtest(table, os.path.join(OUT_DIR, "backtest.png"))


if __name__ == "__main__":
    main()
