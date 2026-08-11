"""
data_parser.py

Parses raw LANL-schema authentication logs (auth.txt / auth.txt.gz) into a
clean, structured pandas DataFrame that the rest of the pipeline builds on.

Works directly against the real LANL Unified Host and Network Dataset
(https://csr.lanl.gov/data/) -- just point `input_path` at the real
auth.txt.gz file. No other code needs to change.
"""

import gzip
import pandas as pd

COLUMNS = [
    "time",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "result",
]


def _open(path):
    """Transparently handle both .gz and plain text LANL log files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_auth_log(input_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load a LANL-schema auth log file into a DataFrame.

    Parameters
    ----------
    input_path : path to auth.txt, auth.txt.gz, or our synthetic sample file
    nrows : optionally limit rows read (useful for the full ~1B-row LANL file
            while prototyping -- read a chunk first, scale up once the
            pipeline works)

    Returns
    -------
    DataFrame with columns: time, source_user, destination_user,
    source_computer, destination_computer, auth_type, logon_type,
    auth_orientation, result

    NOTE: repetitive string columns (users, computers, auth/logon/result
    fields) are loaded as `category` dtype rather than plain strings. In
    real LANL data these columns have relatively few DISTINCT values
    (thousands of users/hosts) repeated across tens of millions of rows.
    category dtype stores each value once and lets pandas' vectorized
    string/groupby operations run per-unique-value instead of per-row --
    this matters a lot once you're extracting more than one attack day
    (prepare_real_data.py's NUM_ATTACK_DAYS), where row counts can climb
    into the tens of millions and plain-string processing (e.g.
    str.split in clean_events()) can fail with a MemoryError on an
    otherwise-fine machine.
    """
    category_cols = [
        "source_user", "destination_user", "source_computer",
        "destination_computer", "auth_type", "logon_type",
        "auth_orientation", "result",
    ]
    dtype_map = {col: "category" for col in category_cols}

    with _open(input_path) as f:
        df = pd.read_csv(f, header=None, names=COLUMNS, nrows=nrows, dtype=dtype_map)
    return df


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning / normalization:
      - drop rows with missing critical fields
      - strip domain suffix from user@domain -> user (keep domain separately
        only if you need multi-domain analysis later)
      - normalize result to a boolean `success` column
      - ensure time is a proper int (seconds since simulation/log start,
        matching LANL's convention)
    """
    df = df.dropna(subset=["time", "source_computer", "destination_computer"]).copy()

    # LANL's real auth.txt.gz marks missing fields with a literal '?'
    # rather than leaving them blank -- these pass straight through
    # dropna() since they're valid (non-null) strings. A '?' host would
    # otherwise become a fake node that silently merges many unrelated
    # events together, so drop any row where a critical field is '?'.
    before = len(df)
    df = df[
        (df["source_computer"] != "?") &
        (df["destination_computer"] != "?") &
        (df["source_user"] != "?")
    ].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"clean_events: dropped {dropped:,} rows with '?' (missing) "
              f"critical fields ({dropped / before * 100:.2f}% of input)")

    # source_computer and destination_computer were loaded as category
    # dtype independently, so pandas fit each column's own separate set
    # of categories -- even though they represent the same "hosts"
    # universe, comparing them directly (e.g. src_host != dst_host in
    # filter_self_loops()) raises "Categoricals can only be compared if
    # 'categories' are the same." Unifying them onto one shared
    # categorical dtype (built from the union of both columns' values)
    # fixes this while keeping the category-dtype memory savings.
    host_categories = pd.api.types.union_categoricals(
        [df["source_computer"], df["destination_computer"]]
    ).categories
    host_dtype = pd.CategoricalDtype(categories=host_categories)
    df["source_computer"] = df["source_computer"].astype(host_dtype)
    df["destination_computer"] = df["destination_computer"].astype(host_dtype)

    df["source_user_clean"] = df["source_user"].str.split("@").str[0]
    df["destination_user_clean"] = df["destination_user"].str.split("@").str[0]

    df["success"] = df["result"].astype(str).str.lower().eq("success")  # small derived column, one-time cost is fine here
    df["time"] = df["time"].astype(int)

    df = df.sort_values("time").reset_index(drop=True)
    return df


def to_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to the minimal event schema the graph builder needs:
    timestamp, source host, destination host, user, auth_type, success.
    """
    events = df[[
        "time", "source_computer", "destination_computer",
        "source_user_clean", "auth_type", "logon_type", "success",
    ]].rename(columns={
        "time": "timestamp",
        "source_computer": "src_host",
        "destination_computer": "dst_host",
        "source_user_clean": "user",
    })
    return events


def parse(input_path: str, nrows: int = None) -> pd.DataFrame:
    """Convenience wrapper: raw log file path -> clean events DataFrame."""
    raw = load_auth_log(input_path, nrows=nrows)
    cleaned = clean_events(raw)
    events = to_events(cleaned)
    return events


def filter_self_loops(events: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drops events where src_host == dst_host (a host authenticating to
    itself -- local logon, service startup). This is common, normal
    background noise, not "movement" between hosts, so it's excluded
    wherever graph structure, novelty tracking, or path-finding actually
    care about host-to-host connections. The raw parsed events (from
    parse()) still include these rows -- only downstream analysis steps
    that build on host-to-host structure call this filter explicitly.

    verbose: set False when this gets called many times in a loop (e.g.
        once per time window) to avoid flooding the console.
    """
    before = len(events)
    filtered = events[events["src_host"] != events["dst_host"]].copy()
    dropped = before - len(filtered)
    if dropped > 0 and verbose:
        print(f"filter_self_loops: removed {dropped:,} self-authentication "
              f"events ({dropped / before * 100:.1f}% of input)")
    return filtered


if __name__ == "__main__":
    # Quick smoke test against the synthetic sample data
    events = parse("D:/Projects/lateral_movement_project/Data/auth_real_subset.txt")
    print(events.shape)
    print(events.head())
    print("\nUnique hosts:", pd.unique(events[["src_host", "dst_host"]].values.ravel()).shape[0])
    print("Time span (seconds):", events["timestamp"].max() - events["timestamp"].min())
