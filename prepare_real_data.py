"""
prepare_real_data.py

The full LANL auth.txt.gz is ~7.2GB / 1.6B+ events. The 749 labeled
red-team events are spread across many different days of the 58-day
dataset -- so a single continuous window from the earliest to latest
attack can end up covering nearly the ENTIRE dataset (this is what
happened on the first attempt: a 34GB subset from a single min-to-max
window).

Instead, this version:
  1. Groups the labeled red-team events by day
  2. Picks a configurable number of those attack-days (NUM_ATTACK_DAYS)
  3. Extracts only a narrow window around EACH chosen day (not one giant
     continuous span), streaming through auth.txt.gz once and checking
     every row against all the small windows at once
  4. Concatenates just those slices into a much smaller, manageable file

This keeps real labeled attacks with real baseline context, at a small
fraction of the full dataset's size.

Run this once, after downloading both files from
https://csr.lanl.gov/data/cyber1/ into your data folder.
"""

import gzip
import pandas as pd

# ---------------------------------------------------------------------------
# Edit these paths to match your setup
# ---------------------------------------------------------------------------
REDTEAM_PATH = r"D:\Projects\lateral_movement_project\Data\redteam.txt.gz"
AUTH_FULL_PATH = r"D:\Projects\lateral_movement_project\Data\auth.txt.gz"
OUTPUT_DIR = r"D:\Projects\lateral_movement_project\Data"

# How many distinct attack-days to include. Start small (3) to keep the
# subset manageable -- you can always increase this and re-run once you've
# confirmed the pipeline works end-to-end.
NUM_ATTACK_DAYS = 2

# Narrow window around EACH chosen attack day, not the whole dataset.
LOOKBACK_SECONDS = 3 * 3600   # 1 day of baseline before the attack day
LOOKAHEAD_SECONDS = 1 * 3600       # 6 hours after, to catch the full chain

AUTH_COLUMNS = [
    "time", "source_user", "destination_user",
    "source_computer", "destination_computer",
    "auth_type", "logon_type", "auth_orientation", "result",
]
REDTEAM_COLUMNS = ["time", "user", "src_computer", "dst_computer"]

CHUNK_SIZE = 2_000_000
SECONDS_PER_DAY = 86400


def load_redteam_events(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        redteam = pd.read_csv(f, header=None, names=REDTEAM_COLUMNS)
    redteam["time"] = redteam["time"].astype(int)
    return redteam


def pick_attack_windows(redteam: pd.DataFrame, num_days: int):
    """
    Groups red-team events by which day they fall on, picks the
    `num_days` days with the MOST labeled events (the richest, most
    useful windows to learn from), and returns a list of
    (window_start, window_end) tuples -- one narrow window per chosen day.
    """
    redteam = redteam.copy()
    redteam["day"] = redteam["time"] // SECONDS_PER_DAY

    counts = redteam.groupby("day").size().sort_values(ascending=False)
    chosen_days = counts.head(num_days).index.tolist()

    print(f"Red-team events span {redteam['day'].nunique()} distinct days total.")
    print(f"Choosing the {num_days} day(s) with the most labeled events:")
    for d in chosen_days:
        print(f"  day {d}: {counts[d]} labeled events")

    windows = []
    for d in chosen_days:
        day_start_t = d * SECONDS_PER_DAY
        day_end_t = day_start_t + SECONDS_PER_DAY
        window_start = max(0, day_start_t - LOOKBACK_SECONDS)
        window_end = day_end_t + LOOKAHEAD_SECONDS
        windows.append((window_start, window_end))

    return windows, chosen_days


def in_any_window(t, windows):
    for start, end in windows:
        if start <= t <= end:
            return True
    return False


def stream_filter_auth_log(auth_path: str, windows: list,
                            output_path: str, chunk_size: int = CHUNK_SIZE) -> int:
    """
    Streams auth.txt.gz once, keeping rows that fall inside ANY of the
    chosen narrow windows. Stops early once past the last window's end
    (file is assumed roughly chronological).
    """
    overall_end = max(end for _, end in windows)

    total_kept = 0
    total_seen = 0

    with gzip.open(auth_path, "rt") as f_in, open(output_path, "w") as f_out:
        reader = pd.read_csv(
            f_in, header=None, names=AUTH_COLUMNS,
            chunksize=chunk_size, low_memory=False,
        )
        for i, chunk in enumerate(reader):
            total_seen += len(chunk)

            chunk_min_time = chunk["time"].min()
            if chunk_min_time > overall_end:
                print(f"  reached t={chunk_min_time}, past last window -- stopping early")
                break

            mask = pd.Series(False, index=chunk.index)
            for start, end in windows:
                mask |= (chunk["time"] >= start) & (chunk["time"] <= end)

            kept = chunk[mask]
            if len(kept) > 0:
                kept.to_csv(f_out, header=False, index=False)
                total_kept += len(kept)

            if (i + 1) % 25 == 0:
                print(f"  ...scanned {total_seen:,} rows so far, "
                      f"kept {total_kept:,}, currently at t={chunk_min_time}")

    return total_kept


def write_real_ground_truth(redteam: pd.DataFrame, chosen_days: list, output_path: str) -> None:
    """Only writes ground truth for the attack-days actually included in
    the extracted subset, so labels line up with the data you have."""
    redteam = redteam.copy()
    redteam["day"] = redteam["time"] // SECONDS_PER_DAY
    subset = redteam[redteam["day"].isin(chosen_days)]

    out = subset[["time", "user", "src_computer", "dst_computer"]].copy()
    out = out.sort_values("time").reset_index(drop=True)
    out.to_csv(output_path, index=False)
    print(f"Included {len(out)} labeled events (out of {len(redteam)} total) "
          f"matching the chosen attack day(s).")


def main():
    print(f"Loading red-team labels from {REDTEAM_PATH} ...")
    redteam = load_redteam_events(REDTEAM_PATH)
    print(f"Loaded {len(redteam):,} labeled red-team events total\n")

    windows, chosen_days = pick_attack_windows(redteam, NUM_ATTACK_DAYS)
    total_span_hours = sum((end - start) / 3600 for start, end in windows)
    print(f"\nExtracting {len(windows)} narrow window(s), "
          f"~{total_span_hours:.1f} total hours of data (not one giant span)")

    auth_out_path = f"{OUTPUT_DIR}\\auth_real_subset.txt"
    print(f"\nStreaming {AUTH_FULL_PATH} and filtering to these windows...")
    kept = stream_filter_auth_log(AUTH_FULL_PATH, windows, auth_out_path)
    print(f"Kept {kept:,} auth events -> {auth_out_path}")

    gt_out_path = f"{OUTPUT_DIR}\\real_ground_truth.csv"
    write_real_ground_truth(redteam, chosen_days, gt_out_path)
    print(f"Saved reformatted ground truth -> {gt_out_path}")

    print("\nDone. If auth_real_subset.txt is still too large, lower "
          "NUM_ATTACK_DAYS (e.g. to 1 or 2) and re-run.")


if __name__ == "__main__":
    main()
