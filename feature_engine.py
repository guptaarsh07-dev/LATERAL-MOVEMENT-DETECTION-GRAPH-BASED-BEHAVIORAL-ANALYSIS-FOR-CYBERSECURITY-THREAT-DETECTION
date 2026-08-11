"""
feature_engine.py

Takes the per-host, per-window graph features (from graph_builder.py) and
turns them into anomaly signals:

  1. Baseline modeling: for each host, learn what "normal" looks like
     (average degree, betweenness, PageRank) from a clean/early period.
  2. Deviation scoring: for every later window, compare each host's actual
     metrics to its own baseline using z-scores.
  3. Novelty tracking: flag when a host contacts a destination host it has
     NEVER contacted before -- this is often the strongest lateral-movement
     signal, stronger than centrality alone, since an attacker moving to a
     new host produces a brand new edge that didn't exist in the baseline.

Output is a single table: one row per host per window, with deviation
z-scores, a novel-destination count, and a combined `deviation_score` you
can threshold on. This feeds directly into path_detector.py next, and later
doubles as engineered input features for the GNN.
"""

import numpy as np
import pandas as pd

from data_parser import parse, filter_self_loops
from graph_builder import build_window_graph, slice_into_windows, compute_host_features


def build_baseline(feature_df: pd.DataFrame, baseline_window_ids) -> pd.DataFrame:
    """
    Compute each host's baseline mean/std for degree, betweenness, and
    PageRank, using only the windows in `baseline_window_ids` (a "clean"
    period assumed to be free of attacks).

    Hosts that don't appear at all in the baseline period get a baseline
    of (mean=0, std=small epsilon) so any activity from them later reads
    as a strong deviation -- appropriate, since a host suddenly appearing
    that was silent before is itself notable.
    """
    baseline_slice = feature_df[feature_df["window_id"].isin(baseline_window_ids)]

    stats = baseline_slice.groupby("host").agg(
        degree_mean=("total_degree", "mean"),
        degree_std=("total_degree", "std"),
        betweenness_mean=("betweenness", "mean"),
        betweenness_std=("betweenness", "std"),
        pagerank_mean=("pagerank", "mean"),
        pagerank_std=("pagerank", "std"),
    ).reset_index()

    # Avoid divide-by-zero: hosts with a single baseline observation (std=NaN)
    # or genuinely zero variance get a small floor std.
    for col in ["degree_std", "betweenness_std", "pagerank_std"]:
        stats[col] = stats[col].fillna(0.0)
        stats[col] = stats[col].replace(0.0, 1e-3)

    return stats


def score_deviations(feature_df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Join each window's host features against that host's baseline and
    compute z-scores. Hosts never seen in the baseline period get a
    default baseline (mean=0, std=epsilon) so their first appearance is
    flagged as a deviation rather than silently ignored.
    """
    merged = feature_df.merge(baseline, on="host", how="left")

    defaults = {
        "degree_mean": 0.0, "degree_std": 1e-3,
        "betweenness_mean": 0.0, "betweenness_std": 1e-3,
        "pagerank_mean": 0.0, "pagerank_std": 1e-3,
    }
    for col, val in defaults.items():
        merged[col] = merged[col].fillna(val)

    merged["degree_z"] = (merged["total_degree"] - merged["degree_mean"]) / merged["degree_std"]
    merged["betweenness_z"] = (merged["betweenness"] - merged["betweenness_mean"]) / merged["betweenness_std"]
    merged["pagerank_z"] = (merged["pagerank"] - merged["pagerank_mean"]) / merged["pagerank_std"]

    return merged


def track_novel_destinations(events: pd.DataFrame, window_seconds: int = 3600) -> pd.DataFrame:
    """
    For each host, in each window, count how many *new* destination hosts
    it contacts that it has never contacted in any prior window. This
    requires walking windows in chronological order and maintaining a
    running "known destinations" set per host.

    Self-authentication events (host talking to itself) are excluded --
    a host's own local logons don't represent movement to a new host and
    would otherwise never register as "novel" in a meaningful way.

    Returns a DataFrame: host, window_id, window_start, novel_destinations
    (list of newly-seen destination hosts), novel_destination_count.
    """
    events = filter_self_loops(events)
    known_destinations = {}  # host -> set of previously-seen destination hosts
    rows = []

    for window_id, start, window_events in slice_into_windows(events, window_seconds):
        # group by source host to see who it talked to this window
        for host, group in window_events.groupby("src_host"):
            dests_this_window = set(group["dst_host"].unique())
            prior_known = known_destinations.get(host, set())
            novel = dests_this_window - prior_known

            rows.append({
                "host": host,
                "window_id": window_id,
                "window_start": start,
                "novel_destinations": sorted(novel),
                "novel_destination_count": len(novel),
            })

            known_destinations[host] = prior_known | dests_this_window

    return pd.DataFrame(rows)


def combine_scores(deviation_df: pd.DataFrame, novelty_df: pd.DataFrame,
                    weights=None) -> pd.DataFrame:
    """
    Merge z-score deviations with novel-destination counts into a single
    `deviation_score` per host per window, and flag hosts above a threshold.

    Default weighting favors novel destinations heavily, since a brand new
    connection to a host you've never touched is a stronger, more specific
    signal than a moderate centrality shift.
    """
    if weights is None:
        weights = {"degree_z": 0.15, "betweenness_z": 0.25, "pagerank_z": 0.15, "novelty": 0.45}

    merged = deviation_df.merge(
        novelty_df[["host", "window_id", "novel_destination_count"]],
        on=["host", "window_id"], how="left",
    )
    merged["novel_destination_count"] = merged["novel_destination_count"].fillna(0)

    # Clip extreme z-scores so a single wild outlier doesn't dominate,
    # then take absolute value since deviation in either direction matters.
    for col in ["degree_z", "betweenness_z", "pagerank_z"]:
        merged[col + "_abs_clipped"] = merged[col].abs().clip(upper=5)

    merged["deviation_score"] = (
        weights["degree_z"] * merged["degree_z_abs_clipped"]
        + weights["betweenness_z"] * merged["betweenness_z_abs_clipped"]
        + weights["pagerank_z"] * merged["pagerank_z_abs_clipped"]
        + weights["novelty"] * merged["novel_destination_count"].clip(upper=5)
    )

    return merged


def flag_anomalies(scored_df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Mark rows above `threshold` as flagged. Tune threshold against your
    ground truth (see run_baseline_eval below) rather than trusting the
    default blindly."""
    scored_df = scored_df.copy()
    scored_df["flagged"] = scored_df["deviation_score"] >= threshold
    return scored_df


def run_pipeline(auth_log_path: str, window_seconds: int = 3600,
                  baseline_fraction: float = 0.3, threshold: float = 1.5):
    """
    Full feature_engine pipeline, from raw log path to flagged hosts:
      parse -> window graphs -> baseline -> deviation scores -> novelty
      -> combined deviation_score -> flagged anomalies.

    baseline_fraction: the first N% of windows (chronologically) are
    treated as the "clean" learning period. In a real deployment this
    should be a period you've manually verified had no incidents.
    """
    from graph_builder import build_feature_timeline

    events = parse(auth_log_path)
    feature_df, _graphs = build_feature_timeline(events, window_seconds=window_seconds)

    all_window_ids = sorted(feature_df["window_id"].unique())
    cutoff = int(len(all_window_ids) * baseline_fraction)
    baseline_window_ids = all_window_ids[:cutoff]

    baseline = build_baseline(feature_df, baseline_window_ids)
    deviation_df = score_deviations(feature_df, baseline)

    novelty_df = track_novel_destinations(events, window_seconds=window_seconds)
    combined = combine_scores(deviation_df, novelty_df)
    flagged = flag_anomalies(combined, threshold=threshold)

    return flagged


def resolve_data_paths(data_dir: str = "D:/Projects/lateral_movement_project/Data"):
    """Auto-detects real vs. synthetic data -- same convention used across
    gnn_model.py / path_detector.py."""
    import os
    real_auth = f"{data_dir}/auth_real_subset.txt"
    real_gt = f"{data_dir}/real_ground_truth.csv"
    if os.path.exists(real_auth) and os.path.exists(real_gt):
        print("Using REAL LANL data (auth_real_subset.txt / real_ground_truth.csv)")
        return real_auth, real_gt
    print("Real data not found -- falling back to synthetic sample data. "
          "Run prepare_real_data.py first to switch to real LANL data.")
    return f"{data_dir}/auth_sample.txt", f"{data_dir}/ground_truth.csv"


if __name__ == "__main__":
    auth_path, gt_path = resolve_data_paths()
    result = run_pipeline(auth_path,
                          window_seconds=3600, baseline_fraction=0.3, threshold=1.5)

    print(f"Total (host, window) rows scored: {len(result)}")
    print(f"Flagged as anomalous: {result['flagged'].sum()}")

    top = result.sort_values("deviation_score", ascending=False).head(15)
    print("\nTop 15 most anomalous (host, window) pairs:")
    print(top[["host", "window_id", "window_start", "deviation_score",
               "novel_destination_count", "flagged"]].to_string(index=False))

    result.to_csv("D:/Projects/lateral_movement_project/Data/host_anomaly_scores.csv", index=False)
    print("\nSaved full scored table to data/host_anomaly_scores.csv")

    # Sanity check against ground truth (works for either format --
    # just displays whichever columns are present)
    gt = pd.read_csv(gt_path)
    print("\nGround truth attacks (for comparison):")
    print(gt.head(20).to_string(index=False))
