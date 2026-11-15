"""Step 2b: download and clean ATL weather observations (METAR/ASOS).

Source: Iowa Environmental Mesonet (IEM) ASOS archive, station ATL.
Both routine hourly reports and special reports are kept, because specials
are issued when the weather changes quickly (e.g. a thunderstorm starts).

Reads:  data/raw/months.txt (to match the flight data window)
Writes: data/raw/weather/atl_{year}_{month}.csv   one raw file per month
        data/clean/weather.parquet                one row per observation
        outputs/phase7b_weather_summary.csv
"""
import os
import sys
import time
import urllib.request
import pandas as pd

RAW_DIR = "data/raw/weather"
CLEAN_DIR = "data/clean"
OUT_DIR = "outputs"
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=ATL&data=all"
       "&year1={y1}&month1={m1}&day1=1&year2={y2}&month2={m2}&day2=1"
       "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no"
       "&report_type=3&report_type=4")

# Weather codes that mean something is falling (rain, snow, drizzle, ice, hail).
PRECIP_CODES = ["RA", "SN", "DZ", "PL", "GR", "GS", "IC", "UP"]
NO_CEILING_FT = 20000   # value used when no broken/overcast layer is reported


def next_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def download(months):
    os.makedirs(RAW_DIR, exist_ok=True)
    failed = []
    for ym in months:
        year, month = int(ym[:4]), int(ym[5:])
        path = os.path.join(RAW_DIR, f"atl_{year}_{month:02d}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            print(f"{path}: already downloaded")
            continue
        y2, m2 = next_month(year, month)
        url = URL.format(y1=year, m1=month, y2=y2, m2=m2)
        request = urllib.request.Request(url, headers={"User-Agent": "portfolio-weather-script"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"{path}: {len(data) / 1e3:.0f} KB")
        except Exception as error:
            print(f"{path}: FAILED ({error})")
            failed.append((url, path))
        time.sleep(2)   # be polite to a free public service
    if failed:
        print("\nSome downloads failed. Download these by hand and save them as shown:")
        for url, path in failed:
            print(f"  {url}\n    -> save as {path}")
        sys.exit(1)


def has_any_code(wxcodes, codes):
    # wxcodes looks like "TS", "-RA BR", "VCTS" or "M" (none).
    if pd.isna(wxcodes) or wxcodes == "M":
        return 0
    for code in codes:
        if code in wxcodes:
            return 1
    return 0


def ceiling(row):
    # Ceiling = lowest broken (BKN), overcast (OVC) or vertical-visibility (VV) layer.
    lowest = NO_CEILING_FT
    for k in range(1, 5):
        cover = row[f"skyc{k}"]
        height = row[f"skyl{k}"]
        if cover in ["BKN", "OVC", "VV"] and not pd.isna(height):
            if height < lowest:
                lowest = height
    return lowest


def clean(months):
    parts = []
    for ym in months:
        path = os.path.join(RAW_DIR, f"atl_{ym[:4]}_{ym[5:]}.csv")
        parts.append(pd.read_csv(path, na_values=["M"], keep_default_na=False,
                                 dtype={"wxcodes": str}, low_memory=False))
    raw = pd.concat(parts, ignore_index=True)

    # IEM times are UTC. Flights are in ATL local time without a time zone, so
    # convert to Eastern time (handles daylight saving) and drop the zone label.
    utc = pd.to_datetime(raw["valid"]).dt.tz_localize("UTC")
    raw["obs_time"] = utc.dt.tz_convert("America/New_York").dt.tz_localize(None)

    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "obs_time": r["obs_time"],
            "wx_thunder": has_any_code(r["wxcodes"], ["TS"]),       # includes VCTS (nearby)
            "wx_precip": has_any_code(r["wxcodes"], PRECIP_CODES),
            "wx_visibility_mi": r["vsby"],
            "wx_ceiling_ft": ceiling(r),
            "wx_gust_kt": 0.0 if pd.isna(r["gust"]) else r["gust"],  # no gust reported = 0
        })
    weather = pd.DataFrame(rows)
    weather = weather.sort_values("obs_time").drop_duplicates("obs_time", keep="last")
    return weather.reset_index(drop=True)


def main():
    with open("data/raw/months.txt") as f:
        months = [line.strip() for line in f if line.strip()]
    download(months)
    weather = clean(months)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    weather.to_parquet(os.path.join(CLEAN_DIR, "weather.parquet"), index=False)

    # Summary per month for checking the data.
    weather["month"] = weather["obs_time"].dt.strftime("%Y-%m")
    rows = []
    for ym in months:
        w = weather[weather["month"] == ym]
        gaps = w["obs_time"].diff().dt.total_seconds() / 3600
        rows.append({
            "month": ym,
            "observations": len(w),
            "thunder_obs": int(w["wx_thunder"].sum()),
            "precip_obs": int(w["wx_precip"].sum()),
            "missing_visibility": int(w["wx_visibility_mi"].isna().sum()),
            "longest_gap_hours": round(gaps.max(), 1),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase7b_weather_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Total observations: {len(weather)}")


if __name__ == "__main__":
    main()
