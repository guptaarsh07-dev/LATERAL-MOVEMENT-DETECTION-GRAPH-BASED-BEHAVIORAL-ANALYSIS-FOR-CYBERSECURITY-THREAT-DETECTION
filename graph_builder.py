"""
graph_builder.py

Builds time-windowed directed graphs from parsed authentication events, and
computes per-host structural features (degree, betweenness, PageRank) for
each window. These features feed both the rule-based baseline detector and,
later, the GNN model (gnn_model.py).

Lateral movement is a *temporal* pattern, so we don't build one static graph
-- we build a sequence of graph snapshots over time and track how each
host's position in the graph changes window to window.
"""

import networkx as nx
import pandas as pd

from data_parser import filter_self_loops


def build_window_graph(events: pd.DataFrame, exclude_self_loops: bool = True) -> nx.DiGraph:
    """
    Build a single directed graph from a slice of events.
    Nodes = hosts. Edge (src -> dst) weight = number of auth events
    between them in this window.

    exclude_self_loops: drops host-authenticating-to-itself events
        (local logon, service startup) before building the graph --
        this is normal background noise, not host-to-host movement,
        and including it would inflate degree/centrality for hosts
        with lots of local activity but little real lateral connectivity.
        Set False if you want the raw graph including self-loops.
    """
    if exclude_self_loops:
        events = filter_self_loops(events, verbose=False)

    G = nx.DiGraph()
    hosts = pd.unique(events[["src_host", "dst_host"]].values.ravel())
    G.add_nodes_from(hosts)

    grouped = events.groupby(["src_host", "dst_host"]).size().reset_index(name="weight")
    for _, row in grouped.iterrows():
        G.add_edge(row["src_host"], row["dst_host"], weight=int(row["weight"]))

    return G


def slice_into_windows(events: pd.DataFrame, window_seconds: int = 3600):
    """
    Split events into time windows and yield (window_id, window_start,
    events_in_window) tuples.

    IMPORTANT: window_id is computed from ABSOLUTE time (timestamp //
    window_seconds), NOT relative to where this particular events slice
    happens to start. This must match the windowing convention used by
    gnn_model.py's build_labels() and scoring_engine.py's
    window_id_for_time() -- both of those compute window_id as
    `time // window_seconds` directly from the raw LANL timestamp.

    If this function instead numbered windows starting from 0 at
    events["timestamp"].min(), window IDs here would only agree with
    build_labels()/window_id_for_time() when the slice happens to start
    at t=0. On a real LANL subset (which starts wherever
    prepare_real_data.py's extraction window begins, e.g. t=680400) the
    two schemes point at completely different windows, so every label
    and every anomaly-score/GNN-probability lookup silently misses --
    no error, just empty joins and an all-zero confusion matrix.

    Default window_seconds=3600 (hourly). Use a smaller window (e.g. 300s /
    5 minutes) if you want finer-grained detection of fast attack chains --
    trade-off is more windows to process.
    """
    min_t, max_t = events["timestamp"].min(), events["timestamp"].max()
    first_window_id = int(min_t // window_seconds)
    last_window_id = int(max_t // window_seconds)

    for window_id in range(first_window_id, last_window_id + 1):
        start = window_id * window_seconds
        end = start + window_seconds
        window_events = events[(events["timestamp"] >= start) & (events["timestamp"] < end)]
        if len(window_events) > 0:
            yield window_id, start, window_events


def compute_host_features(G: nx.DiGraph, betweenness_approx_threshold: int = 500,
                           betweenness_k: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Compute per-host structural features for a single window's graph.
    These are the raw features later used both by the baseline deviation
    scorer and as GNN node input features.

    Exact betweenness centrality costs roughly O(V * E) -- fine for the
    dozens-of-hosts synthetic data, but with thousands of real hosts per
    window this becomes a serious bottleneck (potentially hours per
    window). Above `betweenness_approx_threshold` nodes, this switches to
    NetworkX's k-sampled approximation (only `betweenness_k` random
    source nodes are used to estimate centrality instead of all of them)
    -- much faster, and a good enough estimate for anomaly-scoring
    purposes, where relative ranking matters more than exact values.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "host", "in_degree", "out_degree", "total_degree",
            "betweenness", "pagerank",
        ])

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    n_nodes = G.number_of_nodes()
    if n_nodes > betweenness_approx_threshold:
        k = min(betweenness_k, n_nodes)
        betweenness = nx.betweenness_centrality(G, k=k, weight="weight", seed=seed)
    else:
        betweenness = nx.betweenness_centrality(G, weight="weight")

    pagerank = nx.pagerank(G, weight="weight")

    rows = []
    for host in G.nodes():
        rows.append({
            "host": host,
            "in_degree": in_deg.get(host, 0),
            "out_degree": out_deg.get(host, 0),
            "total_degree": in_deg.get(host, 0) + out_deg.get(host, 0),
            "betweenness": betweenness.get(host, 0.0),
            "pagerank": pagerank.get(host, 0.0),
        })
    return pd.DataFrame(rows)


def build_feature_timeline(events: pd.DataFrame, window_seconds: int = 3600) -> pd.DataFrame:
    """
    Full pipeline: events -> windowed graphs -> per-host features per window.
    Returns one combined DataFrame with a `window_id` and `window_start`
    column, ready to feed into baseline modeling / the GNN.
    """
    import time

    all_features = []
    graphs = {}

    t0 = time.time()
    for window_id, start, window_events in slice_into_windows(events, window_seconds):
        G = build_window_graph(window_events)
        graphs[window_id] = G

        features = compute_host_features(G)
        features["window_id"] = window_id
        features["window_start"] = start
        all_features.append(features)

        if window_id > 0 and window_id % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ...processed window {window_id} "
                  f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges), "
                  f"{elapsed:.1f}s elapsed")

    feature_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    return feature_df, graphs


if __name__ == "__main__":
    from data_parser import parse

    events = parse("D:/Projects/lateral_movement_project/Data/auth_real_subset.txt")

    # Hourly windows over our 3-day synthetic simulation
    feature_df, graphs = build_feature_timeline(events, window_seconds=3600)

    print(f"Built {len(graphs)} time-windowed graphs")
    print(f"Feature table shape: {feature_df.shape}")
    print(feature_df.sort_values("betweenness", ascending=False).head(10))

    feature_df.to_csv("D:/Projects/lateral_movement_project/Data/host_features_timeline.csv", index=False)
    print("\nSaved feature timeline to data/host_features_timeline.csv")
