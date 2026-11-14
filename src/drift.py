"""Daily drift fix: shift every prediction on a day by one log-odds offset.

The offset for day d is fitted on the 7 completed days before d, so that the
average adjusted prediction on those days equals the share that was really
delayed. The current day's outcomes are never used.

Why the previous day is "completed" by decision time: a flight's label
(delayed 15+ or not) is known 15 minutes after its scheduled departure. The
last flight of day d-1 is scheduled before midnight, so every label from d-1
is known by 00:15 on day d, before the first decision on day d.
"""
import numpy as np
import pandas as pd

WINDOW_DAYS = 7


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_offset(p, y):
    """Find b so that mean(sigmoid(logit(p) + b)) equals mean(y).

    The mean prediction goes up as b goes up, so a simple halving search works.
    """
    target = np.mean(y)
    z = logit(np.asarray(p))
    low, high = -5.0, 5.0
    for _ in range(60):
        middle = (low + high) / 2
        if sigmoid(z + middle).mean() < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def daily_offsets(df, pred_col, label_col):
    """One offset per day in df["flight_date"].

    A day gets an offset only if all 7 previous days are present in df;
    otherwise it gets 0 (no adjustment) and has_history = 0.
    """
    dates = sorted(df["flight_date"].unique())
    available = set(dates)
    rows = []
    for d in dates:
        window = []
        for k in range(1, WINDOW_DAYS + 1):
            window.append(d - pd.Timedelta(days=k))
        complete = True
        for day in window:
            if day not in available:
                complete = False
        if complete:
            past = df[df["flight_date"].isin(window)]
            offset = fit_offset(past[pred_col], past[label_col])
        else:
            offset = 0.0
        rows.append({"flight_date": d, "offset": offset, "has_history": int(complete)})
    return pd.DataFrame(rows)


def apply_offsets(df, pred_col, label_col):
    """Return adjusted predictions (same order as df) and the offset table."""
    offsets = daily_offsets(df, pred_col, label_col)
    merged = df[["flight_date"]].merge(offsets, on="flight_date", how="left")
    adjusted = sigmoid(logit(df[pred_col].to_numpy()) + merged["offset"].to_numpy())
    return adjusted, offsets
