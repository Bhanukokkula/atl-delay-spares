"""Settings shared by every step: data split, feature list, leakage check."""

DECISION_HOURS_BEFORE = 2  # we predict 2 hours before scheduled departure

# Time-based split over the months in data/raw/months.txt (Sep 2024 - Aug 2026).
# The last month is test, the 2 before it are validation, everything earlier
# is training. With 24 months, training includes a full summer.
N_VALID_MONTHS = 2
N_TEST_MONTHS = 1

# Inbound legs with a scheduled turn shorter than this are treated as
# "no reliable inbound". Real mainline turns are not planned this short; these
# come from aircraft swaps, because BTS records the tail that actually flew.
# Chosen from the delay rate by turn bucket (outputs/phase2_turn_buckets.csv).
TURN_CUTOFF_MIN = 40

LABEL = "delayed_15"

# Features the model is allowed to use. Every one must be known at decision time.
NUMERIC_FEATURES = [
    "dep_hour",
    "day_of_week",
    "days_to_holiday",
    "distance",
    "sched_block_min",
    "same_hour_departures",
    "route_delay_rate",
    "has_inbound",
    "turn_min",
    "inbound_delay_known",
    "inbound_dep_delay",
    "recent_known",
    "recent_avg_delay",
    "recent_count",
    "backlog_count",
]
# Inbound features, removed together for the "no inbound" comparison run.
INBOUND_FEATURES = ["has_inbound", "turn_min", "inbound_delay_known", "inbound_dep_delay"]
CATEGORICAL_FEATURES = ["dest"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Anything measured after decision time for the flight itself, plus all delay
# cause columns. Both the raw BTS names and our cleaned names are listed.
FORBIDDEN = [
    "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "ArrTime", "ArrDelay",
    "ArrDelayMinutes", "ArrDel15", "ActualElapsedTime", "AirTime",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "Cancelled", "CancellationCode", "Diverted",
    "dep_delay", "actual_dep", "arr_delay", "delayed_15", "cancelled",
    "cancel_code", "diverted",
]


def check_no_leakage(columns):
    """Stop the program if a forbidden column is about to reach the model."""
    bad = []
    for col in columns:
        if col in FORBIDDEN:
            bad.append(col)
    if bad:
        raise RuntimeError(f"LEAKAGE: forbidden columns in model input: {bad}")
    for col in columns:
        if col not in FEATURES:
            raise RuntimeError(f"LEAKAGE CHECK: '{col}' is not on the approved feature list")
