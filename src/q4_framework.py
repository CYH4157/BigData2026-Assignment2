"""Question 4: framework figure of the full pipeline (Graphviz).

Numbers come from the run outputs (outputs/figures/q1_stats.json, outputs/analysis.json,
outputs/ensemble/valid_report.json), so run q1_schema.py, analysis.py and ensemble.py first.
Run from the repository root:  python src/q4_framework.py
Writes outputs/figures/q4_framework.gv, and .png/.pdf if Graphviz (dot) is installed.
"""
import json
import shutil
from pathlib import Path

import graphviz

from data import MIN_COV

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"


def box(g, name, title, lines, color):
    body = "".join(f'<tr><td align="left"><font point-size="10">{l}</font></td></tr>' for l in lines)
    g.node(name, f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
                 f'<tr><td align="left"><b>{title}</b></td></tr>{body}</table>>', fillcolor=color)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    period = json.loads((OUT / "q1_stats.json").read_text())["period"]
    graph = json.loads((ROOT / "outputs" / "analysis.json").read_text())["graph"]
    ens = json.loads((ROOT / "outputs" / "ensemble" / "valid_report.json").read_text())
    w = ens["weight_gbdt"]

    g = graphviz.Digraph("framework", graph_attr={"rankdir": "TB", "nodesep": "0.3", "ranksep": "0.35", "newrank": "true",
                                                  "fontname": "Helvetica", "dpi": "200"},
                         node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica",
                                    "color": "#888888"},
                         edge_attr={"color": "#555555", "arrowsize": "0.8", "fontname": "Helvetica"})

    box(g, "db", "GH Archive → preprocess.py", [period.replace(" – ", " –<br/>"), "8 tables, PK-FK links",
                                                "anonymized ids, masked text"], "#e9ecef")
    box(g, "prep", "Point-in-time preparation",
        ["filter by row time &lt; t", "known-later columns only if &lt; t", "detect collection gaps (≥ 6 h)"],
        "#e9ecef")
    box(g, "table", "Training table (user, t, 6 labels)",
        ["weekly t: Sat 16:00 UTC (as test week)", "features at t - 4 h",
         "users: &#8805; 5 records in prior 90 days", f"labels with coverage &lt; {MIN_COV:.0%} dropped",
         "last 4 weeks = validation"], "#fff3cd")

    with g.subgraph(name="cluster_gbdt", graph_attr={"label": "A. Feature-based model", "style": "dashed",
                                                     "color": "#999999", "fontsize": "12"}) as a:
        box(a, "feat", "Point-in-time features (100)",
            ["own history: counts 1-90 days, recency", "relational: open PRs, peers in same repos",
             "cohort level, data coverage"], "#d1e7dd")
        box(a, "gbdt", "LightGBM x 6", ["one binary model per activity", "early stopping on validation"], "#d1e7dd")
        a.edge("feat", "gbdt")

    with g.subgraph(name="cluster_rdl", graph_attr={"label": "B. Relational Deep Learning", "style": "dashed",
                                                    "color": "#999999", "fontsize": "12"}) as b:
        box(b, "graph", "Relational entity graph",
            [f"{graph['nodes'] / 1e6:.2f}M nodes = rows", f"{graph['edge_types']} edge types (PK-FK, both ways)",
             "node time = row time"], "#cfe2ff")
        box(b, "sample", "Temporal neighbor sampling",
            ["only nodes with time &#8804; t", "most recent first, 3 hops (64/32/16)"], "#cfe2ff")
        box(b, "gnn", "Heterogeneous GNN",
            ["table encoder (PyTorch Frame)", "+ relative-time encoding (t - node time)",
             "3-layer HeteroGraphSAGE (sum)", "MLP head, 6 outputs, masked BCE", "3 seeds averaged"], "#cfe2ff")
        b.edge("graph", "sample")
        b.edge("sample", "gnn")

    box(g, "ens", "Ensemble", [f"p = {w:.1f} GBDT + {1 - w:.1f} RDL", "per-activity thresholds",
                               "tuned for F1 on validation"], "#f8d7da")
    box(g, "out", "output_{studentID}.csv", ["UserID + 6 binary columns", "checked with evaluate.py"], "#e2d9f3")

    # top row: data preparation (left to right); bottom row: ensemble and output
    with g.subgraph() as r:
        r.attr(rank="same")
        for n in ("db", "prep", "table"):
            r.node(n)
    with g.subgraph() as r:
        r.attr(rank="same")
        for n in ("ens", "out"):
            r.node(n)
    g.edge("db", "prep")
    g.edge("prep", "table")
    g.edge("table", "feat")
    g.edge("table", "graph", label="  seeds (user, t)", fontsize="9")
    g.edge("gbdt", "ens", label="  probabilities", fontsize="9")
    g.edge("gnn", "ens")
    g.edge("ens", "out")
    g.save(OUT / "q4_framework.gv")
    if shutil.which("dot"):
        for fmt in ["png", "pdf"]:
            g.render(OUT / "q4_framework", format=fmt, cleanup=True)
    print(f"saved {OUT / 'q4_framework'}.gv (+ png/pdf if Graphviz is installed)")


if __name__ == "__main__":
    main()
