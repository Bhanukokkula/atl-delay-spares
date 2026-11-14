"""Step 2: clean the raw BTS files into two tables.

Outputs:
  data/clean/departures.parquet  DL mainline flights leaving ATL (the flights we score)
  data/clean/arrivals.parquet    DL mainline flights arriving ATL (inbound legs)
  outputs/phase1_summary.csv     row counts, delay rate, cancellation rate per month
"""
import os
import zipfile
import pandas as pd

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
OUT_DIR = "outputs"

# Only the columns we need. Delay cause columns and taxi/wheels times are
# never loaded, so they cannot leak into the model by accident.
COLUMNS = [
    "FlightDate", "Month", "DayOfWeek", "Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepDelay", "CRSArrTime", "ArrDelay",
    "Cancelled", "CancellationCode", "Diverted", "CRSElapsedTime", "Distance",
]
TEXT_COLUMNS = {"CRSDepTime": str, "CRSArrTime": str, "CancellationCode": str,
                "Tail_Number": str}


def hhmm_to_minutes(series):
    # BTS stores clock times as "hhmm" strings. "2400" means midnight at the
    # end of the day; treating it as 1440 minutes keeps it on the right day.
    number = series.astype(int)
    return (number // 100) * 60 + (number % 100)


def read_month(path):
    with zipfile.ZipFile(path) as z:
        csv_name = None
        for name in z.namelist():
            if name.endswith(".csv"):
                csv_name = name
        with z.open(csv_name) as f:
            df = pd.read_csv(f, usecols=COLUMNS, dtype=TEXT_COLUMNS, low_memory=False)
    # Mainline Delta only. Delta Connection flights are reported by other carriers.
    df = df[df["Reporting_Airline"] == "DL"]
    df = df[(df["Origin"] == "ATL") | (df["Dest"] == "ATL")]
    return df


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(RAW_DIR, "months.txt")) as f:
        months = [line.strip() for line in f if line.strip()]

    parts = []
    for ym in months:
        year, month = ym.split("-")
        path = os.path.join(RAW_DIR, f"bts_{year}_{month}.zip")
        df = read_month(path)
        df["year_month"] = ym
        print(f"{ym}: {len(df)} DL rows touching ATL")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    df["flight_date"] = pd.to_datetime(df["FlightDate"])
    df["crs_dep_min"] = hhmm_to_minutes(df["CRSDepTime"])
    df["crs_arr_min"] = hhmm_to_minutes(df["CRSArrTime"])
    df["cancelled"] = df["Cancelled"].astype(int)
    df["diverted"] = df["Diverted"].astype(int)

    # ---- Departures from ATL ----
    dep = df[df["Origin"] == "ATL"].copy()
    # CRSDepTime is ATL local time and FlightDate is the ATL local date.
    dep["sched_dep"] = dep["flight_date"] + pd.to_timedelta(dep["crs_dep_min"], unit="m")
    # Actual departure = scheduled + delay. Using the delay (not DepTime) handles
    # flights that slip past midnight without any date arithmetic.
    dep["actual_dep"] = dep["sched_dep"] + pd.to_timedelta(dep["DepDelay"], unit="m")
    # Label is only defined for flights that departed. Cancelled flights stay in
    # the table (label = NaN) until we decide how to treat them.
    dep["delayed_15"] = (dep["DepDelay"] >= 15).astype(float)
    dep.loc[dep["cancelled"] == 1, "delayed_15"] = float("nan")

    dep = dep.rename(columns={
        "Tail_Number": "tail", "Flight_Number_Reporting_Airline": "flight_number",
        "Dest": "dest", "Distance": "distance", "CRSElapsedTime": "sched_block_min",
        "DayOfWeek": "day_of_week", "Month": "month", "DepDelay": "dep_delay",
        "CancellationCode": "cancel_code",
    })
    dep = dep[["year_month", "flight_date", "month", "day_of_week", "flight_number",
               "tail", "dest", "distance", "sched_block_min", "sched_dep", "actual_dep",
               "dep_delay", "delayed_15", "cancelled", "cancel_code", "diverted"]]
    dep = dep.sort_values("sched_dep").reset_index(drop=True)

    # ---- Arrivals into ATL (inbound legs) ----
    arr = df[df["Dest"] == "ATL"].copy()
    # CRSArrTime is ATL local, but FlightDate is the departure date at the origin.
    # All domestic origins are in ATL's time zone or west of it, so if the
    # arrival clock time is earlier than the departure clock time, the flight
    # landed the next day.
    next_day = arr["crs_arr_min"] < arr["crs_dep_min"]
    arr["sched_arr"] = arr["flight_date"] + pd.to_timedelta(arr["crs_arr_min"], unit="m")
    arr.loc[next_day, "sched_arr"] = arr.loc[next_day, "sched_arr"] + pd.Timedelta(days=1)

    arr = arr.rename(columns={
        "Tail_Number": "tail", "Flight_Number_Reporting_Airline": "flight_number",
        "Origin": "origin", "CRSElapsedTime": "sched_block_min", "DepDelay": "dep_delay",
        "ArrDelay": "arr_delay",
    })
    arr = arr[["year_month", "flight_date", "flight_number", "tail", "origin",
               "sched_block_min", "sched_arr", "dep_delay", "arr_delay",
               "cancelled", "diverted"]]
    arr = arr.sort_values("sched_arr").reset_index(drop=True)

    dep.to_parquet(os.path.join(CLEAN_DIR, "departures.parquet"), index=False)
    arr.to_parquet(os.path.join(CLEAN_DIR, "arrivals.parquet"), index=False)

    # ---- Checkpoint 1 summary ----
    rows = []
    for ym in months:
        d = dep[dep["year_month"] == ym]
        flown = d[d["cancelled"] == 0]
        n_arr = (arr["year_month"] == ym).sum()
        rows.append({
            "month": ym,
            "departures": len(d),
            "arrivals": int(n_arr),
            "cancelled": int(d["cancelled"].sum()),
            "cancel_rate_pct": round(100 * d["cancelled"].mean(), 2),
            "diverted": int(d["diverted"].sum()),
            "delay15_rate_pct": round(100 * flown["delayed_15"].mean(), 1),
            "missing_tail": int(d["tail"].isna().sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase1_summary.csv"), index=False)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Total departures: {len(dep)}  arrivals: {len(arr)}")
    print(f"Overall 15+ min delay rate (flown flights): "
          f"{100 * dep.loc[dep['cancelled'] == 0, 'delayed_15'].mean():.1f}%")
    print(f"Overall cancellation rate: {100 * dep['cancelled'].mean():.2f}%")
    print("Cancellation codes (A=carrier, B=weather, C=NAS, D=security):")
    print(dep.loc[dep["cancelled"] == 1, "cancel_code"].value_counts().to_string())
    print(f"Arrivals landing the day after departure: {int((arr['sched_arr'].dt.normalize() > arr['flight_date']).sum())}")


if __name__ == "__main__":
    main()
