"""
app.py

Flask backend for the Lateral Movement Detection dashboard. Serves:
  - GET /                -> the dashboard page (D3.js force-directed graph
                             + ranked incident sidebar)
  - GET /api/graph       -> network topology built ONLY from the top 100
                             flagged (highest-severity) incident paths, with
                             each host's risk tier attached
  - GET /api/incidents   -> ranked incidents as JSON, for the sidebar list
  - GET /api/incidents/<n> -> a single incident's full detail

This reads the CSVs already produced by the rest of the pipeline
(data_parser -> graph_builder -> feature_engine -> gnn_model ->
path_detector -> scoring_engine) rather than recomputing anything -- the
dashboard is a pure presentation layer on top of what's already detected.

NOTE: /api/graph used to parse the ENTIRE auth log and build the whole
network's graph (potentially tens of millions of events, all hosts) just
to visualize it -- slow to build and, more importantly, not what an
analyst actually wants to look at: a wall of thousands of irrelevant
normal hosts drowning out the handful that matter. The graph is now built
directly from the top 100 highest-severity incidents' paths only, which
is both faster (no full-log parse needed at all) and a more useful,
focused view -- exactly the hosts and connections behind your most
severe flagged incidents.
"""

from flask import Flask, jsonify, render_template
import pandas as pd
import networkx as nx

app = Flask(__name__)

DATA_DIR = r"D:\Projects\lateral_movement_project\Data"
RANKED_INCIDENTS_PATH = f"{DATA_DIR}/ranked_incidents.csv"

TIER_RANK = {"High": 3, "Medium": 2, "Low": 1}
TOP_N_INCIDENTS = 100


def load_incidents() -> pd.DataFrame:
    return pd.read_csv(RANKED_INCIDENTS_PATH)


def load_top_incidents(n: int = TOP_N_INCIDENTS) -> pd.DataFrame:
    """The single shared definition of 'top N incidents' -- used by both
    /api/graph and /api/incidents so the graph and the sidebar list always
    show the same underlying set, never two different top-100s."""
    return (
        load_incidents()
        .sort_values("severity_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def compute_host_risk(incidents: pd.DataFrame) -> dict:
    """For every host that appears in any of these incidents' paths, keep
    the highest severity tier it was ever part of -- that's what colors
    the node in the graph view."""
    host_risk = {}
    for _, row in incidents.iterrows():
        hosts = row["path"].split(" -> ")
        tier = row["severity_tier"]
        for h in hosts:
            current = host_risk.get(h)
            if current is None or TIER_RANK[tier] > TIER_RANK[current]:
                host_risk[h] = tier
    return host_risk


def build_graph_payload() -> dict:
    """
    Builds the graph-view JSON payload directly from the top N ranked
    incidents' paths -- NOT from the full auth log. Each incident's path
    (e.g. "C1521 -> C625 -> C2162") contributes its hosts as nodes and its
    consecutive hops as edges; an edge's weight is how many of these top
    incidents traversed that exact hop (so a hop shared by several
    high-severity paths shows up thicker in the visualization).

    This means /api/graph no longer needs to parse the auth log or import
    data_parser/graph_builder at all -- it's now just derived from
    ranked_incidents.csv, same as /api/incidents, so there's nothing slow
    to cache and no separate startup step required.
    """
    top_incidents = load_top_incidents()
    host_risk = compute_host_risk(top_incidents)

    G = nx.DiGraph()
    for _, row in top_incidents.iterrows():
        hosts = row["path"].split(" -> ")
        for h in hosts:
            G.add_node(h)
        for u, v in zip(hosts[:-1], hosts[1:]):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    nodes = [
        {
            "id": host,
            "risk": host_risk.get(host, "Normal"),
            "degree": G.in_degree(host) + G.out_degree(host),
        }
        for host in G.nodes()
    ]
    links = [
        {"source": u, "target": v, "weight": data.get("weight", 1)}
        for u, v, data in G.edges(data=True)
    ]

    return {"nodes": nodes, "links": links}


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    return jsonify(build_graph_payload())


@app.route("/api/incidents")
def api_incidents():
    incidents = load_top_incidents()

    incidents["incident_id"] = incidents.index + 1

    cols = [
        "incident_id",
        "path",
        "severity_tier",
        "severity_score",
        "num_hops",
        "num_novel_hops",
        "duration_seconds",
        "start_time",
        "end_time",
        "final_host",
        "users_involved",
        "max_deviation_score",
        "max_gnn_probability"
    ]

    return jsonify(incidents[cols].to_dict(orient="records"))

@app.route("/api/incidents/<int:incident_id>")
def api_incident_detail(incident_id):
    incidents = load_incidents().sort_values("severity_score", ascending=False).reset_index(drop=True)
    if incident_id < 1 or incident_id > len(incidents):
        return jsonify({"error": "incident not found"}), 404
    row = incidents.iloc[incident_id - 1]
    return jsonify(row.to_dict())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
