"""
report_generator.py

Turns the ranked_incidents.csv output from scoring_engine.py into
human-readable incident write-ups -- the kind of text a SOC analyst would
actually read, or that goes into a compliance/audit report.

Deliberately template-based, NOT LLM-generated: every sentence is built
from a fixed template with the actual detected values filled in. This
keeps it fully explainable and auditable -- an analyst (or an evaluator)
can trace every claim in the report straight back to a specific number in
the pipeline, with no black-box reasoning involved.
"""

import pandas as pd
from datetime import timedelta 


def format_elapsed(seconds: int) -> str:
    """Formats a synthetic simulation-time offset (seconds since log start)
    as a readable elapsed-time string, e.g. '1d 05:42:18'. Swap this out
    for real datetime formatting once using real, timestamped LANL data."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def severity_explanation(row: pd.Series) -> str:
    """Builds the 'why this was flagged' sentence -- the evidence behind
    the severity score, in plain language."""
    reasons = []

    if row["num_novel_hops"] > 0:
        reasons.append(
            f"{int(row['num_novel_hops'])} of the {int(row['num_hops'])} hop(s) in this path "
            f"connect hosts that have never communicated with each other before in the log"
        )
    if row["duration_seconds"] <= 120:
        reasons.append(
            f"the entire chain completed in {int(row['duration_seconds'])} seconds, "
            f"far faster than typical human-driven administrative activity"
        )
    if row["max_deviation_score"] >= 1.5:
        reasons.append(
            f"at least one host on this path deviates significantly "
            f"(score {row['max_deviation_score']:.2f}) from its own historical baseline behavior"
        )
    if row["max_gnn_probability"] >= 0.5:
        reasons.append(
            f"the graph neural network model assigned a "
            f"{row['max_gnn_probability']*100:.0f}% compromise probability "
            f"to at least one host on this path"
        )

    if not reasons:
        return "This path was flagged as a fast, structurally unusual traversal chain."

    return "This path was flagged because " + "; and ".join(reasons) + "."


def generate_incident_narrative(row: pd.Series, incident_number: int) -> str:
    """Builds the full templated write-up for a single incident."""
    hosts = row["path"].split(" -> ")
    start_str = format_elapsed(row["start_time"])
    end_str = format_elapsed(row["end_time"])

    lines = [
        f"### Incident #{incident_number} — Severity: {row['severity_tier']} "
        f"(score {row['severity_score']:.2f})",
        "",
        f"**Traversal path:** `{row['path']}`  ",
        f"**Hosts involved:** {len(hosts)} ({row['num_hops']} hop(s))  ",
        f"**Time window:** {start_str} to {end_str} "
        f"(duration: {int(row['duration_seconds'])}s)  ",
        f"**User(s) involved:** {row['users_involved']}  ",
        f"**Final target host:** `{row['final_host']}`  ",
        "",
        severity_explanation(row),
        "",
        "**Recommended next step:** " + recommend_action(row),
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def recommend_action(row: pd.Series) -> str:
    """Simple rule-based action recommendation per severity tier -- the
    system informs the analyst's decision, it does not act autonomously."""
    if row["severity_tier"] == "High":
        return (
            "Escalate immediately. Investigate the final target host "
            f"(`{row['final_host']}`) for signs of compromise and consider "
            "isolating the involved hosts pending review."
        )
    elif row["severity_tier"] == "Medium":
        return (
            "Review within the current shift. Confirm whether this traversal "
            "matches known, legitimate administrative activity (e.g. scheduled "
            "maintenance or backup jobs) before dismissing."
        )
    return "Log for reference. No immediate action required unless corroborated by other alerts."


def generate_report(ranked_incidents: pd.DataFrame, min_tier: str = "Medium",
                     max_incidents: int = 25) -> str:
    """
    Builds the full Markdown incident report, highest severity first.

    min_tier: only include incidents at or above this severity
              ("High", "Medium", or "Low")
    max_incidents: cap the report length -- in a real deployment you'd
              paginate or filter by date range instead
    """
    tier_order = {"High": 0, "Medium": 1, "Low": 2}
    min_rank = tier_order[min_tier]

    filtered = ranked_incidents[
        ranked_incidents["severity_tier"].map(tier_order) <= min_rank
    ].sort_values("severity_score", ascending=False).head(max_incidents)

    tier_counts = ranked_incidents["severity_tier"].value_counts().to_dict()

    header = [
        "# Lateral Movement Detection — Incident Report",
        "",
        f"**Total incidents scored:** {len(ranked_incidents)}  ",
        f"**High severity:** {tier_counts.get('High', 0)}  "
        f"**Medium severity:** {tier_counts.get('Medium', 0)}  "
        f"**Low severity:** {tier_counts.get('Low', 0)}  ",
        f"**Showing:** top {len(filtered)} incident(s) at or above '{min_tier}' severity",
        "",
        "---",
        "",
    ]

    body = []
    for i, (_, row) in enumerate(filtered.iterrows(), start=1):
        body.append(generate_incident_narrative(row, i))

    return "\n".join(header + body)


if __name__ == "__main__":
    ranked = pd.read_csv("D:/Projects/lateral_movement_project/Data/ranked_incidents.csv")

    report_text = generate_report(ranked, min_tier="Medium", max_incidents=25)

    output_path = "D:/Projects/lateral_movement_project/Data/incident_report.md"
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"Generated report with incidents at Medium severity or above")
    print(f"Saved to {output_path}\n")
    print("--- Preview (first incident) ---\n")
    print(report_text[:report_text.find("---", 200) + 3])
