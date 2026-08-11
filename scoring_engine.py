"""
scoring_engine.py

Each earlier module produces its own signal:
  - feature_engine.py    -> per-host, per-window baseline deviation z-scores
  - gnn_model.py          -> per-host, per-window GNN compromise probability
  - path_detector.py      -> actual fast/novel traversal chains (the incidents)

A detected *path* from path_detector.py is the natural unit an analyst acts
on ("this host chain looks like an attack"), so this module treats each
path as one incident, and enriches it by pulling in the deviation score and
GNN probability of every host along that path at the relevant time window.
These get combined into one severity_score, then bucketed into
High / Medium / Low so the dashboard and incident reports have a single,
rankable number to work from.
"""

import pandas as pd 


def window_id_for_time(timestamp: int, window_seconds: int = 3600) -> int:
    """Same windowing convention used in graph_builder.py / feature_engine.py
    / gnn_model.py -- must match so lookups line up correctly."""
    return int(timestamp // window_seconds)


def enrich_path_with_host_signals(paths: pd.DataFrame, anomaly_scores: pd.DataFrame,
                                   gnn_flagged: pd.DataFrame, window_seconds: int = 3600) -> pd.DataFrame:
    """
    For every detected path, look up the baseline deviation_score and GNN
    probability of each host in the path, at the window_id corresponding
    to the path's start_time. Attaches the MEAN across hosts in the path
    for each signal.

    NOTE: this used to take the MAX across hosts in the path, on the
    reasoning that one clearly-anomalous host is concerning even if the
    others look ordinary. In practice this backfired badly: if even ONE
    host anywhere in the dataset happens to be the single most-confident
    GNN prediction (e.g. probability 0.337, matching the model's actual
    max output), then EVERY path that happens to include that one host
    -- regardless of how unrelated or how many other different hosts are
    in the chain -- inherits that same top score. This produced a top-10
    "most severe" list that was 10 completely different host chains all
    tied at the exact same severity_score, because they all shared one
    common hub host. Real, more spread-out attack evidence across a path
    doesn't get to compete with that on a max-aggregation basis.

    Mean aggregation fixes this: a path only scores highly if its hosts
    are, on average, more suspicious -- one inflated host among several
    ordinary ones gets diluted rather than dominating the whole path's
    score. The column names (max_deviation_score, max_gnn_probability)
    are kept as-is so nothing downstream (report_generator.py, app.py,
    index.html) needs to change, even though the values are now means.
    """
    anomaly_lookup = anomaly_scores.set_index(["host", "window_id"])["deviation_score"].to_dict()
    gnn_lookup = gnn_flagged.set_index(["host", "window_id"])["gnn_probability"].to_dict() \
        if len(gnn_flagged) > 0 else {}

    max_deviation = []
    max_gnn_prob = []

    for _, row in paths.iterrows():
        hosts = row["path"].split(" -> ")
        window_id = window_id_for_time(row["start_time"], window_seconds)

        deviations = [anomaly_lookup.get((h, window_id), 0.0) for h in hosts]
        gnn_probs = [gnn_lookup.get((h, window_id), 0.0) for h in hosts]

        max_deviation.append(sum(deviations) / len(deviations) if deviations else 0.0)
        max_gnn_prob.append(sum(gnn_probs) / len(gnn_probs) if gnn_probs else 0.0)

    enriched = paths.copy()
    enriched["max_deviation_score"] = max_deviation
    enriched["max_gnn_probability"] = max_gnn_prob
    return enriched


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a signal to [0, 1] so different-scale signals
    (z-scores, probabilities, hop counts) can be combined fairly."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def compute_severity(enriched: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Combines four normalized signals into one severity_score in [0, 1]:
      - max_deviation_score   : how far the worst host on the path deviates
                                 from its own normal baseline
      - max_gnn_probability   : the GNN's confidence any host on the path
                                 is compromised
      - num_novel_hops        : how many brand-new connections the path used
      - speed_score           : inverse of duration -- faster chains score
                                 higher (human admin work rarely moves this
                                 fast across multiple hosts)

    Default weights favor the path-detector's own signals (novelty, speed)
    since those were shown to reliably catch the real attacks; the GNN and
    baseline deviation act as corroborating evidence.
    """
    if weights is None:
        weights = {"deviation": 0.20, "gnn": 0.25, "novelty": 0.30, "speed": 0.25}

    df = enriched.copy()

    df["deviation_norm"] = normalize(df["max_deviation_score"])
    df["gnn_norm"] = df["max_gnn_probability"]  # already in [0, 1]
    df["novelty_norm"] = normalize(df["num_novel_hops"])

    # Speed score: shorter duration = more suspicious. Add 1 to avoid
    # divide-by-zero on instantaneous (0-second) chains.
    df["speed_score"] = normalize(1.0 / (df["duration_seconds"] + 1.0))

    df["severity_score"] = (
        weights["deviation"] * df["deviation_norm"]
        + weights["gnn"] * df["gnn_norm"]
        + weights["novelty"] * df["novelty_norm"]
        + weights["speed"] * df["speed_score"]
    )

    return df


def assign_severity_tier(df: pd.DataFrame, high_percentile: float = 0.99,
                          medium_percentile: float = 0.90) -> pd.DataFrame:
    """
    Buckets the continuous severity_score into High / Medium / Low, since
    that's what an analyst actually scans first on a dashboard.

    Cutoffs are now PERCENTILE-based rather than fixed absolute values
    (previously high_cutoff=0.65, medium_cutoff=0.4 hardcoded). Those
    fixed values were implicitly calibrated against an earlier, inflated
    GNN signal (a training-window score of 0.87 leaking into
    max_gnn_probability). Once that leak was fixed and GNN probabilities
    correctly topped out around 0.35, every single incident's combined
    severity_score fell under 0.4 -- so ALL 214,640 incidents landed in
    "Low", with nothing ever reaching Medium or High again. Fixed
    absolute cutoffs silently stop matching reality whenever the
    upstream signals shift (a different pos_weight, more training data,
    a different scoring formula, etc.).

    Percentile-based cutoffs instead always mean the same thing
    regardless of the score distribution: "High" = the top 1% most
    severe incidents in THIS run, "Medium" = the next 9% (90th-99th
    percentile), "Low" = everything else. This self-calibrates
    automatically any time you retrain the GNN, change the weighting in
    compute_severity(), or run against different data.
    """
    df = df.copy()

    high_cutoff = df["severity_score"].quantile(high_percentile)
    medium_cutoff = df["severity_score"].quantile(medium_percentile)

    def tier(score):
        if score >= high_cutoff:
            return "High"
        elif score >= medium_cutoff:
            return "Medium"
        return "Low"

    df["severity_tier"] = df["severity_score"].apply(tier)

    print(f"Severity tier cutoffs for this run: "
          f"High >= {high_cutoff:.4f} (top {(1-high_percentile)*100:.0f}%), "
          f"Medium >= {medium_cutoff:.4f} (top {(1-medium_percentile)*100:.0f}%)")

    return df


def run_scoring_pipeline(paths_path: str, anomaly_scores_path: str,
                          gnn_flagged_path: str, window_seconds: int = 3600) -> pd.DataFrame:
    paths = pd.read_csv(paths_path)
    anomaly_scores = pd.read_csv(anomaly_scores_path)
    gnn_flagged = pd.read_csv(gnn_flagged_path)

    enriched = enrich_path_with_host_signals(paths, anomaly_scores, gnn_flagged, window_seconds)
    scored = compute_severity(enriched)
    ranked = assign_severity_tier(scored)

    ranked = ranked.sort_values("severity_score", ascending=False).reset_index(drop=True)
    return ranked


if __name__ == "__main__":
    ranked = run_scoring_pipeline(
        paths_path="D:/Projects/lateral_movement_project/Data/suspicious_paths.csv",
        anomaly_scores_path="D:/Projects/lateral_movement_project/Data/host_anomaly_scores.csv",
        gnn_flagged_path="D:/Projects/lateral_movement_project/Data/gnn_flagged_hosts.csv",
        window_seconds=3600,
    )

    print(f"Scored {len(ranked)} incidents\n")
    print("Severity tier counts:")
    print(ranked["severity_tier"].value_counts().to_string())

    display_cols = ["path", "severity_score", "severity_tier", "num_novel_hops",
                     "duration_seconds", "max_deviation_score", "max_gnn_probability"]
    print("\nTop 10 ranked incidents:")
    print(ranked[display_cols].head(10).to_string(index=False))

    ranked.to_csv("D:/Projects/lateral_movement_project/Data/ranked_incidents.csv", index=False)
    print("\nSaved to data/ranked_incidents.csv")

    # Sanity check: where do the real (ground-truth) attacks rank?
    # Auto-detects synthetic (host_chain) vs real LANL (src_computer/
    # dst_computer/time) format -- same convention used by
    # path_detector.py's evaluate_against_ground_truth(), since this
    # project runs against real_ground_truth.csv, not the synthetic
    # ground_truth.csv, and the two have completely different columns.
    import os

    real_gt_path = "D:/Projects/lateral_movement_project/Data/real_ground_truth.csv"
    synthetic_gt_path = "D:/Projects/lateral_movement_project/Data/ground_truth.csv"

    if os.path.exists(real_gt_path):
        ground_truth = pd.read_csv(real_gt_path)
    else:
        ground_truth = pd.read_csv(synthetic_gt_path)

    print("\n--- Where do the real planted attacks rank? ---")

    if "host_chain" in ground_truth.columns:
        # Synthetic format: one row per full pre-assembled attack chain,
        # so we can check for an exact path match.
        for _, attack in ground_truth.iterrows():
            match = ranked[ranked["path"] == attack["host_chain"]]
            if len(match) > 0:
                row = match.iloc[0]
                rank_position = match.index[0] + 1
                print(f"Attack {attack['attack_id']} ({attack['host_chain']}): "
                      f"rank #{rank_position} of {len(ranked)}, "
                      f"severity={row['severity_score']:.3f} ({row['severity_tier']})")
            else:
                print(f"Attack {attack['attack_id']} ({attack['host_chain']}): not found in ranked incidents")

    elif {"src_computer", "dst_computer", "time"}.issubset(ground_truth.columns):
        # Real LANL redteam format: one row per individual malicious hop,
        # not a full chain -- so check whether that specific hop appears
        # ANYWHERE inside any ranked incident's path, same matching logic
        # path_detector.py already uses for this format.
        found = 0
        for _, event in ground_truth.iterrows():
            hop_str = f"{event['src_computer']} -> {event['dst_computer']}"
            match = ranked[ranked["path"].str.contains(hop_str, regex=False)]
            if len(match) > 0:
                row = match.iloc[0]
                rank_position = match.index[0] + 1
                found += 1
                print(f"Redteam hop {hop_str} (user={event['user']}): "
                      f"appears in rank #{rank_position} of {len(ranked)}, "
                      f"severity={row['severity_score']:.3f} ({row['severity_tier']})")
            else:
                print(f"Redteam hop {hop_str} (user={event['user']}): not found in ranked incidents")
        print(f"\nSummary: {found} / {len(ground_truth)} labeled red-team hops "
              f"({found / len(ground_truth) * 100:.1f}%) appear inside a ranked incident")

    else:
        raise ValueError(
            "Unrecognized ground truth format -- expected either a "
            "'host_chain' column (synthetic) or 'src_computer'/'dst_computer'/"
            "'time' columns (real LANL redteam.txt format)."
        )
