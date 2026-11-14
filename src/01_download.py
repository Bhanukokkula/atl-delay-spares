"""Step 1: download the last 24 months of BTS on-time data.

Finds the most recent month BTS has published, then downloads that month
and the 23 before it into data/raw/. Files already on disk are skipped.
"""
import os
import sys
import urllib.request
import urllib.error

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
RAW_DIR = "data/raw"
N_MONTHS = 24


def file_exists_online(year, month):
    request = urllib.request.Request(URL.format(year=year, month=month), method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except urllib.error.URLError:
        return False


def previous_month(year, month):
    if month == 1:
        return year - 1, 12
    return year, month - 1


def find_latest_month(start_year, start_month):
    # BTS publishes with a lag of a few months, so walk back until a file exists.
    year, month = start_year, start_month
    for _ in range(12):
        if file_exists_online(year, month):
            return year, month
        year, month = previous_month(year, month)
    return None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Start the search from the current month.
    import datetime
    today = datetime.date.today()
    latest = find_latest_month(today.year, today.month)
    if latest is None:
        print("Could not find any BTS file online in the last 12 months (searched back from today).")
        sys.exit(1)
    print(f"Most recent month available: {latest[0]}-{latest[1]:02d}")

    # Build the list of months, oldest first.
    months = []
    year, month = latest
    for _ in range(N_MONTHS):
        months.append((year, month))
        year, month = previous_month(year, month)
    months.reverse()

    failed = []
    for year, month in months:
        filename = f"bts_{year}_{month:02d}.zip"
        path = os.path.join(RAW_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print(f"{filename}: already downloaded")
            continue
        url = URL.format(year=year, month=month)
        print(f"{filename}: downloading ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"{filename}: {os.path.getsize(path) / 1e6:.1f} MB")
        except Exception as error:
            print(f"{filename}: FAILED ({error})")
            failed.append((year, month, url, path))

    if failed:
        print("\nSome downloads failed. Download these by hand and save them as shown:")
        for year, month, url, path in failed:
            print(f"  {url}\n    -> save as {path}")
        sys.exit(1)

    # Record which months we used so later steps agree on the split.
    with open(os.path.join(RAW_DIR, "months.txt"), "w") as f:
        for year, month in months:
            f.write(f"{year}-{month:02d}\n")
    print(f"All {N_MONTHS} months downloaded.")


if __name__ == "__main__":
    main()
