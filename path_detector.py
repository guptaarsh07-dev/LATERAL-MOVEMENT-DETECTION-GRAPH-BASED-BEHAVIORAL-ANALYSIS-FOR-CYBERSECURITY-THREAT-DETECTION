"""
path_detector.py

The centrality/GNN modules (graph_builder.py, feature_engine.py, gnn_model.py)
work on hourly windows -- good for learning what "normal" looks like, but
too coarse to see the actual shape of an attack, since a real lateral
movement chain (Host A -> Host B -> Host C) unfolds in seconds to minutes,
not across a full hour.

This module works directly on the raw, timestamp-ordered event stream and
searches for *time-respecting paths*: sequences of hops where each next hop
leaves soon after arriving at the previous host (fast enough that it's very
unlikely to be an unrelated human session), and where at least one hop is a
brand-new connection never seen before in the log's history.

This is the piece that answers "how did the threat move", not just "which
host looks off" -- it produces the actual attack path, timestamps, and
evidence that feeds the dashboard's path visualization and the incident
report generator.
"""

import pandas as pd

from data_parser import parse, filter_self_loops


def mark_first_time_edges(events: pd.DataFrame) -> pd.DataFrame:
    """
    Flags, for each event, whether this is the FIRST time `src_host` has
    ever contacted `dst_host` anywhere in the log up to this point. A
    first-time edge appearing inside a fast hop chain is one of the
    strongest lateral-movement signals: the attacker is reaching
    somewhere the source has never been.

    Vectorized with pandas groupby + cumcount instead of a Python
    row-by-row loop with a manual set -- the original approach was fine
    for thousands of rows but becomes a real bottleneck at tens of
    millions of rows (real LANL data scale). This does the same thing in
    one pass without iterating in Python at all.
    """
    events = events.sort_values("timestamp").reset_index(drop=True)
    # cumcount() == 0 marks the first occurrence of each (src_host, dst_host)
    # pair in timestamp order -- exactly "first time this edge appears".
    events = events.copy()
    events["is_novel_edge"] = (
        events.groupby(["src_host", "dst_host"]).cumcount() == 0
    )
    return events


def build_adjacency(events: pd.DataFrame) -> dict:
    """
    host -> sorted arrays of its outgoing events (timestamps as a numpy
    array for fast binary search, plus parallel lists for dst_host, user,
    is_novel_edge), used to walk forward in time from any given host.

    Storing timestamps as a separate sorted numpy array (rather than a
    list of dicts) lets _find_next_hops use binary search (bisect)
    instead of scanning every outgoing event of a host linearly -- this
    matters a lot for hub-like hosts (e.g. domain controllers) that can
    have huge numbers of outgoing events in real data.
    """
    import numpy as np

    adjacency = {}
    for host, group in events.groupby("src_host"):
        group = group.sort_values("timestamp")
        adjacency[host] = {
            "timestamps": group["timestamp"].to_numpy(),
            "dst_host": group["dst_host"].tolist(),
            "user": group["user"].tolist(),
            "is_novel_edge": group["is_novel_edge"].tolist(),
        }
    return adjacency


def _find_next_hops(adjacency: dict, host: str, after_time: int, max_gap: int):
    """Outgoing events from `host` that occur within `max_gap` seconds
    after `after_time` -- candidate next hops in a fast traversal chain.
    Uses binary search (bisect) on the pre-sorted timestamp array instead
    of a linear scan, since a host's outgoing-event list can be very
    large in real data (e.g. a domain controller)."""
    import bisect

    entry = adjacency.get(host)
    if entry is None:
        return []

    timestamps = entry["timestamps"]
    lo = bisect.bisect_right(timestamps, after_time)
    hi = bisect.bisect_right(timestamps, after_time + max_gap)
    if lo >= hi:
        return []

    return [
        {
            "timestamp": timestamps[i],
            "dst_host": entry["dst_host"][i],
            "user": entry["user"][i],
            "is_novel_edge": entry["is_novel_edge"][i],
        }
        for i in range(lo, hi)
    ]


def find_suspicious_paths(events: pd.DataFrame, max_gap_seconds: int = 120,
                           min_hops: int = 2, max_hops: int = 6,
                           require_novel_hop: bool = True) -> pd.DataFrame:
    """
    Searches for fast, multi-hop traversal chains starting from every
    (src, dst, time) event in the log.

    Parameters
    ----------
    max_gap_seconds : maximum time between arriving at a host and leaving
        it again for the chain to still count as "fast" / suspicious.
        120s is deliberately tight -- ordinary human-driven admin work
        rarely re-authenticates to a new host within 2 minutes of the last.
    min_hops : minimum chain length (number of hops/edges) to report --
        a single hop isn't "lateral movement", it's just a login.
    require_novel_hop : if True, only report chains containing at least
        one never-seen-before edge (the strongest signal). Set False to
        see all fast chains regardless of novelty, useful for tuning.

    Returns
    -------
    DataFrame, one row per detected suspicious path, with the host
    sequence, timestamps, duration, user(s) involved, and novel-hop count.
    """
    events = filter_self_loops(events)
    events = mark_first_time_edges(events)
    adjacency = build_adjacency(events)

    detected = []
    seen_path_signatures = set()  # avoid reporting the same chain many times

    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    total_events = len(events_sorted)
    print(f"path_detector: searching from {total_events:,} starting events...")

    # itertuples is much faster and far more memory-efficient than
    # to_dict("records") at real-data scale (tens of millions of rows) --
    # it avoids materializing a full list of Python dicts up front.
    for i, start_event in enumerate(events_sorted.itertuples(index=False)):
        if i > 0 and i % 1_000_000 == 0:
            print(f"  ...processed {i:,} / {total_events:,} starting events, "
                  f"{len(detected):,} suspicious paths found so far")

        path = [start_event.src_host, start_event.dst_host]
        timestamps = [start_event.timestamp]
        novel_flags = [start_event.is_novel_edge]
        users = {start_event.user}

        current_host = start_event.dst_host
        current_time = start_event.timestamp

        for _ in range(max_hops - 1):
            candidates = _find_next_hops(adjacency, current_host, current_time, max_gap_seconds)
            # Avoid immediately bouncing back to a host already in the path
            candidates = [c for c in candidates if c["dst_host"] not in path]
            if not candidates:
                break
            # Take the earliest next hop (the most literal "fast forward" continuation)
            next_hop = min(candidates, key=lambda c: c["timestamp"])

            path.append(next_hop["dst_host"])
            timestamps.append(next_hop["timestamp"])
            novel_flags.append(next_hop["is_novel_edge"])
            users.add(next_hop["user"])

            current_host = next_hop["dst_host"]
            current_time = next_hop["timestamp"]

        num_hops = len(path) - 1
        if num_hops < min_hops:
            continue
        if require_novel_hop and not any(novel_flags):
            continue

        signature = (tuple(path), timestamps[0])
        if signature in seen_path_signatures:
            continue
        seen_path_signatures.add(signature)

        detected.append({
            "path": " -> ".join(path),
            "num_hops": num_hops,
            "start_time": timestamps[0],
            "end_time": timestamps[-1],
            "duration_seconds": timestamps[-1] - timestamps[0],
            "num_novel_hops": sum(novel_flags),
            "final_host": path[-1],
            "users_involved": ", ".join(sorted(users)),
        })

    result = pd.DataFrame(detected)
    if len(result) > 0:
        result = result.sort_values(
            ["num_novel_hops", "num_hops"], ascending=[False, False]
        ).reset_index(drop=True)
    return result


def evaluate_against_synthetic_ground_truth(detected: pd.DataFrame, ground_truth: pd.DataFrame,
                                             time_tolerance: int = 60) -> None:
    """
    For the synthetic ground_truth.csv format (attack_id, start_time,
    host_chain, ...): checks whether any detected path matches the full
    planted chain, within `time_tolerance` seconds of its recorded start.
    """
    print("\n--- Ground truth match check (synthetic format) ---")
    for _, attack in ground_truth.iterrows():
        gt_chain = attack["host_chain"]
        gt_start = attack["start_time"]

        match = detected[
            (detected["path"].str.contains(gt_chain.split(" -> ")[0])) &
            (detected["start_time"].between(gt_start - time_tolerance, gt_start + time_tolerance + 300))
        ]

        exact = detected[detected["path"] == gt_chain]
        status = "EXACT MATCH" if len(exact) > 0 else (
            "partial/time-window match" if len(match) > 0 else "NOT FOUND"
        )
        print(f"Attack {attack['attack_id']} ({gt_chain}) @ t={gt_start}: {status}")


def evaluate_against_real_ground_truth(detected: pd.DataFrame, redteam: pd.DataFrame,
                                        time_tolerance: int = 300) -> None:
    """
    For the real LANL redteam.txt.gz format (time, user, src_computer,
    dst_computer): each row is a single malicious authentication event,
    not a full pre-assembled chain like the synthetic data. So instead of
    checking for an exact chain match, this checks whether the specific
    (src_computer -> dst_computer) hop appears ANYWHERE inside any detected
    path, with that hop's timestamp reasonably close to the labeled event
    time. This is the right check because a real attacker's full path may
    span more hosts than any single labeled hop records -- what matters is
    whether our detector caught that specific malicious connection at all.
    """
    print("\n--- Ground truth match check (real LANL redteam format) ---")
    total = len(redteam)
    found = 0

    for _, event in redteam.iterrows():
        src, dst, t = event["src_computer"], event["dst_computer"], event["time"]
        hop_str = f"{src} -> {dst}"

        match = detected[
            detected["path"].str.contains(hop_str, regex=False) &
            detected["start_time"].between(t - time_tolerance, t + time_tolerance)
        ]

        status = "FOUND" if len(match) > 0 else "NOT FOUND"
        if status == "FOUND":
            found += 1
        print(f"Redteam event user={event['user']} {hop_str} @ t={t}: {status}")

    print(f"\nSummary: {found} / {total} labeled red-team events "
          f"({found / total * 100:.1f}%) appear inside a detected path")


def evaluate_against_ground_truth(detected: pd.DataFrame, ground_truth: pd.DataFrame,
                                   time_tolerance: int = 60) -> None:
    """
    Dispatcher: auto-detects which ground-truth format was passed in
    (synthetic host_chain format vs. real LANL redteam format) and runs
    the matching evaluation. Use this from calling code so you don't need
    to remember which specific function to call.
    """
    if "host_chain" in ground_truth.columns:
        evaluate_against_synthetic_ground_truth(detected, ground_truth, time_tolerance)
    elif {"src_computer", "dst_computer", "time"}.issubset(ground_truth.columns):
        # Real format benefits from a wider default tolerance since labeled
        # events aren't tied to our fixed hourly windows the way synthetic
        # attacks are -- 300s (5 min) is a more realistic match window.
        evaluate_against_real_ground_truth(detected, ground_truth,
                                            time_tolerance=max(time_tolerance, 300))
    else:
        raise ValueError(
            "Unrecognized ground truth format -- expected either a "
            "'host_chain' column (synthetic) or 'src_computer'/'dst_computer'/"
            "'time' columns (real LANL redteam.txt format)."
        )


def resolve_data_paths(data_dir: str = "D:/Projects/lateral_movement_project/Data"):
    """Auto-detects real vs. synthetic data, same convention as
    gnn_model.py's resolve_data_paths() -- keeps both scripts consistent
    so switching to real data never requires editing file paths by hand."""
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
    events = parse(auth_path)
    ground_truth = pd.read_csv(gt_path)

    detected = find_suspicious_paths(
        events, max_gap_seconds=120, min_hops=2, max_hops=6, require_novel_hop=True
    )

    print(f"Detected {len(detected)} suspicious fast/novel traversal chains\n")
    if len(detected) > 0:
        print(detected.head(15).to_string(index=False))

    detected.to_csv("D:/Projects/lateral_movement_project/Data/suspicious_paths.csv", index=False)
    print("\nSaved to data/suspicious_paths.csv")

    evaluate_against_ground_truth(detected, ground_truth)
