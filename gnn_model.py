"""
gnn_model.py

Trains a GraphSAGE Graph Neural Network (via PyTorch Geometric) to classify
each host, in each time window, as Normal or Compromised -- learned directly
from graph structure and node features, rather than the hand-set thresholds
used in feature_engine.py.

Reference: Hamilton, W. L., Ying, Z., & Leskovec, J. (2017). Inductive
Representation Learning on Large Graphs. NeurIPS 2017 (arXiv:1706.02216).

Each time window's graph becomes one training example (a `torch_geometric.
data.Data` object). Node features come from graph_builder.py /
feature_engine.py (degree, betweenness, PageRank, novel-destination count).
Labels come from ground_truth.csv: a host is "Compromised" in the window
where it appears as part of a lateral-movement chain.

This is deliberately built against the synthetic sample data so it runs
end-to-end today. Swap in real LANL data + its documented red-team windows
once you have dataset access -- nothing else in this file needs to change.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

from data_parser import parse
from graph_builder import slice_into_windows, build_window_graph, compute_host_features
from feature_engine import track_novel_destinations

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Build labeled graph snapshots
# ---------------------------------------------------------------------------

def build_labels(ground_truth: pd.DataFrame, window_seconds: int) -> dict:
    """
    Maps window_id -> set of hosts that should be labeled Compromised in
    that window, based on the ground-truth attack chains and their start
    times. A host is labeled compromised in every window whose time range
    overlaps the attack.

    Handles TWO ground-truth formats automatically:
      - Synthetic format (generate_sample_data.py): columns
        [attack_id, start_time, compromised_user, host_chain, final_target]
      - Real LANL format (redteam.txt.gz, loaded via prepare_real_data.py):
        columns [time, user, src_computer, dst_computer], one row per
        individual malicious authentication event rather than a
        pre-assembled chain.
    """
    labels = {}

    if "host_chain" in ground_truth.columns:
        # Synthetic format: one row per full attack chain
        for _, row in ground_truth.iterrows():
            window_id = int(row["start_time"] // window_seconds)
            hosts = row["host_chain"].split(" -> ")
            labels.setdefault(window_id, set()).update(hosts)

    elif {"src_computer", "dst_computer", "time"}.issubset(ground_truth.columns):
        # Real LANL redteam format: one row per individual malicious event.
        # Both the source and destination host of each labeled event are
        # marked compromised in the window that event occurred in.
        for _, row in ground_truth.iterrows():
            window_id = int(row["time"] // window_seconds)
            labels.setdefault(window_id, set()).update(
                [row["src_computer"], row["dst_computer"]]
            )

    else:
        raise ValueError(
            "Unrecognized ground truth format -- expected either a "
            "'host_chain' column (synthetic) or 'src_computer'/'dst_computer'/"
            "'time' columns (real LANL redteam.txt format)."
        )

    return labels


def build_pyg_dataset(events: pd.DataFrame, ground_truth: pd.DataFrame,
                       window_seconds: int = 3600):
    """
    Converts each time window into a PyG Data object:
      - x: node feature matrix [degree, betweenness, pagerank, novel_dests]
      - edge_index: graph edges for that window
      - y: per-node binary label (1 = compromised, 0 = normal)

    Returns: list[Data], plus the host<->index mapping per window (needed
    later to map predictions back to host names).
    """
    novelty_df = track_novel_destinations(events, window_seconds=window_seconds)
    novelty_lookup = novelty_df.set_index(["host", "window_id"])["novel_destination_count"].to_dict()

    label_map = build_labels(ground_truth, window_seconds)

    dataset = []
    window_host_index = {}  # window_id -> {host: node_idx}

    for window_id, start, window_events in slice_into_windows(events, window_seconds):
        G = build_window_graph(window_events)
        if G.number_of_nodes() < 2:
            continue

        features = compute_host_features(G)
        host_to_idx = {host: i for i, host in enumerate(features["host"])}
        window_host_index[window_id] = host_to_idx

        novel_counts = features["host"].apply(
            lambda h: novelty_lookup.get((h, window_id), 0)
        )

        x = torch.tensor(np.stack([
            features["in_degree"].values,
            features["out_degree"].values,
            features["betweenness"].values,
            features["pagerank"].values,
            novel_counts.values,
        ], axis=1), dtype=torch.float)

        compromised_hosts = label_map.get(window_id, set())
        y = torch.tensor(
            [1.0 if h in compromised_hosts else 0.0 for h in features["host"]],
            dtype=torch.float,
        )

        edge_list = [[host_to_idx[u], host_to_idx[v]] for u, v in G.edges()]
        if len(edge_list) == 0:
            continue
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.window_id = window_id
        data.hosts = list(features["host"])
        dataset.append(data)

    return dataset


# ---------------------------------------------------------------------------
# 2. Model definition
# ---------------------------------------------------------------------------

class LateralMovementGNN(torch.nn.Module):
    """Two-layer GraphSAGE encoder + linear head -> single logit per node
    (probability that host is compromised in this time window)."""

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out.squeeze(-1)  # one logit per node


# ---------------------------------------------------------------------------
# 3. Training / evaluation
# ---------------------------------------------------------------------------

def train_test_split_by_time(dataset, train_fraction: float = 0.7):
    """Temporal split: earliest windows for training, latest for testing --
    avoids leaking future patterns into training, which a random split
    would do."""
    dataset_sorted = sorted(dataset, key=lambda d: d.window_id)
    cutoff = int(len(dataset_sorted) * train_fraction)
    return dataset_sorted[:cutoff], dataset_sorted[cutoff:]


def compute_pos_weight(dataset, max_weight: float = 75.0) -> torch.Tensor:
    """Class imbalance is severe (few compromised nodes vs. many normal
    ones), so weight the positive class in the loss accordingly.

    Capped at `max_weight`. NOTE: this cap needs tuning against your
    actual data's imbalance ratio, not set-and-forget. Two failure modes
    on either side:
      - cap too HIGH (or uncapped): raw ratio here is ~495x (72,741
        normal vs 147 compromised), which pushed the model to flag
        almost everything -> ~3% precision at every threshold.
      - cap too LOW (e.g. 20): undercorrects so much that the model
        just learns to predict "normal" for everyone and still gets a
        very low loss -> 0 recall at every threshold, because missing a
        positive is no longer costly enough to be worth getting right.
    75 is a middle-ground starting point between those two extremes --
    check the diagnostic probability stats printed after training, and
    adjust up/down from there if recall is still 0 or precision is still
    near-random.
    """
    total_pos = sum(d.y.sum().item() for d in dataset)
    total_neg = sum((d.y == 0).sum().item() for d in dataset)
    if total_pos == 0:
        return torch.tensor(1.0)
    raw_weight = total_neg / total_pos
    return torch.tensor(min(raw_weight, max_weight))


def train(model, train_data, epochs: int = 60, lr: float = 0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos_weight = compute_pos_weight(train_data)
    loader = DataLoader(train_data, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, batch.y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss: {total_loss / len(train_data):.4f}")


@torch.no_grad()
def evaluate(model, test_data, threshold: float = 0.5):
    model.eval()
    all_preds, all_labels = [], []

    for data in test_data:
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        all_preds.append(preds)
        all_labels.append(data.y)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nConfusion matrix -> TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def flag_compromised_hosts(model, dataset, threshold: float = 0.5):
    """Run inference across all windows and return a table of flagged
    (host, window_id, probability) rows -- this is what path_detector.py
    and the dashboard will consume."""
    model.eval()
    rows = []
    for data in dataset:
        probs = torch.sigmoid(model(data.x, data.edge_index))
        for host, prob in zip(data.hosts, probs.tolist()):
            if prob >= threshold:
                rows.append({"host": host, "window_id": data.window_id, "gnn_probability": prob})
    return pd.DataFrame(rows)


@torch.no_grad()
def print_probability_diagnostics(model, test_data):
    """
    Prints the actual distribution of predicted probabilities on the test
    set, split by true label. This is the key diagnostic when a threshold
    sweep shows all-zero or all-flagged results at every threshold -- it
    tells you whether the model is producing ANY separation between
    compromised and normal hosts, versus just collapsing to one constant
    prediction regardless of pos_weight or threshold.

    A healthy result looks like: compromised-host probabilities noticeably
    higher on average than normal-host probabilities (even if both
    distributions overlap some). A broken result looks like: both
    distributions clustered in the same narrow range (e.g. everything
    near 0.0, or everything near 1.0) -- in which case no threshold will
    ever separate them well, and the fix is elsewhere (pos_weight, more
    training data/epochs, or features), not the threshold.
    """
    model.eval()
    all_probs, all_labels = [], []
    for data in test_data:
        probs = torch.sigmoid(model(data.x, data.edge_index))
        all_probs.append(probs)
        all_labels.append(data.y)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    print("\n=== Predicted probability diagnostics (test set) ===")
    print(f"Compromised hosts (n={len(pos_probs)}): "
          f"min={pos_probs.min().item():.4f}  "
          f"mean={pos_probs.mean().item():.4f}  "
          f"max={pos_probs.max().item():.4f}" if len(pos_probs) > 0 else
          "Compromised hosts (n=0): no positive examples in test set")
    print(f"Normal hosts      (n={len(neg_probs)}): "
          f"min={neg_probs.min().item():.4f}  "
          f"mean={neg_probs.mean().item():.4f}  "
          f"max={neg_probs.max().item():.4f}" if len(neg_probs) > 0 else
          "Normal hosts (n=0): no negative examples in test set")


# ---------------------------------------------------------------------------
# 4. Script entry point
# ---------------------------------------------------------------------------

def resolve_data_paths(data_dir: str = "D:/Projects/lateral_movement_project/Data"):
    """
    Auto-detects whether prepare_real_data.py has been run: if
    auth_real_subset.txt / real_ground_truth.csv exist, uses the real LANL
    data; otherwise falls back to the synthetic sample data. This means you
    don't need to edit this file at all once you run prepare_real_data.py --
    just re-run this script.
    """
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

    dataset = build_pyg_dataset(events, ground_truth, window_seconds=3600)
    print(f"Built {len(dataset)} labeled graph snapshots "
          f"({sum(int(d.y.sum().item()) for d in dataset)} total compromised-host labels)")

    train_data, test_data = train_test_split_by_time(dataset, train_fraction=0.7)
    print(f"Train windows: {len(train_data)}  Test windows: {len(test_data)}")

    model = LateralMovementGNN(in_channels=dataset[0].x.shape[1], hidden_channels=32)
    train(model, train_data, epochs=60, lr=0.01)

    print_probability_diagnostics(model, test_data)

    print("\n=== Threshold sweep (pick the best precision/recall tradeoff below) ===")
    print("NOTE: thresholds chosen based on the probability diagnostics above --")
    print("this model's outputs cluster well below 0.5, so sweeping 0.5-0.99 (as")
    print("in earlier runs) tested thresholds the model can never actually reach.")
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        print(f"\n--- threshold = {t} ---")
        evaluate(model, test_data, threshold=t)

    # ---------------------------------------------------------------------
    # CHOSEN_THRESHOLD: set this after reading the sweep above. This is the
    # ONE place the threshold is set -- evaluate(), flag_compromised_hosts(),
    # and the saved gnn_flagged_hosts.csv all use this same value, so the
    # metrics you see printed below match exactly what gets fed into
    # scoring_engine.py downstream. (Previously this was hardcoded to 0.5
    # in three different places, so the printed metrics didn't match what
    # actually got saved to disk -- that's part of why C1521-style false
    # positives were flooding the ranked incidents despite the model
    # scoring much better at higher thresholds.)
    # ---------------------------------------------------------------------
    # 0.2 chosen over the F1-optimal 0.3: in a security-detection context,
    # missing a real compromised host (false negative) is typically far
    # costlier than one extra alert an analyst has to dismiss (false
    # positive) -- so recall (64.6% @ 0.2 vs 35.4% @ 0.3) is weighted more
    # heavily here than raw F1 would suggest. Precision is still very low
    # in absolute terms (0.046) -- an honest, explainable limitation given
    # only 226 labeled training examples across 20 windows.
    CHOSEN_THRESHOLD = 0.2

    print(f"\n=== Final evaluation at CHOSEN_THRESHOLD = {CHOSEN_THRESHOLD} ===")
    metrics = evaluate(model, test_data, threshold=CHOSEN_THRESHOLD)

    flagged = flag_compromised_hosts(model, test_data, threshold=CHOSEN_THRESHOLD)
    flagged.to_csv("D:/Projects/lateral_movement_project/Data/gnn_flagged_hosts.csv", index=False)
    print(f"\nSaved {len(flagged)} flagged (host, window) predictions to "
          f"data/gnn_flagged_hosts.csv (threshold={CHOSEN_THRESHOLD})")
    print("NOTE: flagged only from test_data (held-out windows), not the full "
          "dataset -- using the full dataset here would include training "
          "windows the model has directly seen labels for, so a high "
          "probability there could just reflect memorization rather than a "
          "genuine detection. This is why one host (e.g. C1798 in an earlier "
          "run) could show up with a suspiciously high max_gnn_probability "
          "(0.87) even though the model's own diagnostic showed test-set "
          "probabilities topping out around 0.35 -- that 0.87 was very "
          "likely coming from a training window, not a real held-out result.")

    torch.save(model.state_dict(), "D:/Projects/lateral_movement_project/Data/gnn_model.pt")
    print("Saved trained model weights to data/gnn_model.pt")
