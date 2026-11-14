"""Question 1: schema graph of the 8 tables (nodes = tables, edges = foreign key -> referenced primary key),
annotated with the row count and time range of each table in the data used by the model.

Run from the repository root:  python src/q1_schema.py
Writes outputs/figures/q1_schema_graph.gv and q1_stats.json, and .png/.pdf if Graphviz (dot) is installed.
"""
import json
import shutil
from pathlib import Path

import graphviz
import pandas as pd

from data import ROW_TIME, load_db

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# table -> (primary key, [foreign keys])
TABLES = {
    "Users": ("user_id", []),
    "Repositories": ("repo_id", []),
    "PullRequests": ("pr_id", ["repo_id", "author_id", "merged_by_id"]),
    "Issues": ("issue_id", ["repo_id", "author_id", "assignee_id", "pull_request_id"]),
    "IssueComments": ("comment_id", ["issue_id", "repo_id", "author_id"]),
    "PRReviews": ("review_id", ["pr_id", "repo_id", "reviewer_id"]),
    "PRReviewComments": ("comment_id", ["pr_id", "repo_id", "author_id", "pull_request_review_id", "in_reply_to_id"]),
    "Pushes": ("push_id", ["user_id", "repo_id"]),
}
FK_TARGET = {
    "repo_id": "Repositories", "author_id": "Users", "merged_by_id": "Users", "assignee_id": "Users",
    "reviewer_id": "Users", "user_id": "Users", "pull_request_id": "PullRequests", "issue_id": "Issues",
    "pr_id": "PullRequests", "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments",
}
KNOWN_LATER = {"merged_by_id"}  # FK filled in only after the row is created
HEADER = {"Users": "#4C72B0", "Repositories": "#55A868"}
EVENT_HEADER = "#8172B2"


def stats(db):
    out = {}
    for name, df in db.items():
        t = df[ROW_TIME[name]]
        out[name] = {"rows": int(len(df)), "time_col": ROW_TIME[name], "from": str(t.min()), "to": str(t.max())}
    return out


def label(name, st):
    pk, fks = TABLES[name]
    color = HEADER.get(name, EVENT_HEADER)
    rows = [f'<TR><TD BGCOLOR="{color}"><FONT COLOR="white"><B>{name}</B></FONT></TD></TR>',
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{st["rows"]:,} rows</FONT></TD></TR>',
            f'<TR><TD PORT="{pk}" ALIGN="LEFT" BGCOLOR="#FFF2CC"><B>PK</B>  {pk}</TD></TR>']
    for fk in fks:
        note = " *" if fk in KNOWN_LATER else ""
        rows.append(f'<TR><TD PORT="{fk}" ALIGN="LEFT" BGCOLOR="#DDEBF7"><B>FK</B>  {fk}{note}</TD></TR>')
    rows.append(f'<TR><TD ALIGN="LEFT"><I>time: {st["time_col"]}</I></TD></TR>')
    return '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">' + "".join(rows) + "</TABLE>>"


def build(st, period):
    n_fk = sum(len(f) for _, f in TABLES.values())
    g = graphviz.Digraph("schema")
    g.attr(rankdir="LR", splines="spline", nodesep="0.35", ranksep="1.1", dpi="200", labelloc="t",
           fontsize="16", fontname="Helvetica",
           label=f"Schema graph: 8 tables, {n_fk} foreign keys (FK → referenced PK).  * known only later\n"
                 f"Data: GH Archive {period}, built with the course preprocess.py")
    g.attr("node", shape="plaintext", fontname="Helvetica", fontsize="11")
    g.attr("edge", arrowsize="0.7")
    for group in (["PRReviewComments", "Pushes"], ["IssueComments", "PRReviews"],
                  ["Issues", "PullRequests"], ["Users", "Repositories"]):
        with g.subgraph() as s:
            s.attr(rank="same")
            for t in group:
                s.node(t, label(t, st[t]))
    for src, (_, fks) in TABLES.items():
        for fk in fks:
            dst = FK_TARGET[fk]
            pk, color = TABLES[dst][0], HEADER.get(dst, EVENT_HEADER)  # edge colored by the referenced table
            if src == dst:
                g.edge(f"{src}:{fk}:w", f"{dst}:{pk}:w", style="dashed", color=color)
            else:
                g.edge(f"{src}:{fk}:e", f"{dst}:{pk}:w", color=color)
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = stats(load_db())
    lo = min(pd.Timestamp(v["from"]) for v in st.values())
    hi = max(pd.Timestamp(v["to"]) for v in st.values())
    period = f"{lo:%Y-%m-%d %H:%M} – {hi:%Y-%m-%d %H:%M} UTC"
    (OUT / "q1_stats.json").write_text(json.dumps({"period": period, "tables": st}, indent=2))
    g = build(st, period)
    g.save(OUT / "q1_schema_graph.gv")
    if shutil.which("dot"):
        for fmt in ["png", "pdf"]:
            g.render(OUT / "q1_schema_graph", format=fmt, cleanup=True)
    print(f"period {period}; saved {OUT / 'q1_schema_graph'}.gv (+ png/pdf if Graphviz is installed)")


if __name__ == "__main__":
    main()
