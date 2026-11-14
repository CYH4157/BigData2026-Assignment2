"""Extra analyses for the report (all on the current data/db):

1. Persistence baselines with different look-back windows on the validation weeks (official scoring rule).
2. How late PRs / issues / reviews enter the data (row time first_seen_at vs created_at / submitted_at).
3. Label coverage and positive rate of every validation week.
4. Share of LightGBM gain per feature group (own history, relational, cohort level, data coverage).
5. Size of the relational entity graph.

Run from the repository root after train_gbdt.py:  python src/analysis.py --gbdt base
Writes outputs/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import ACTS, MIN_COV, activity_log, load_db
from train_gbdt import OUT as GBDT_OUT, YCOLS, build_dataset, cache_path, week_scores, weekly_starts

ROOT = Path(__file__).resolve().parents[1]
FK = {"PullRequests": ["repo_id", "author_id"], "Issues": ["repo_id", "author_id", "assignee_id", "pull_request_id"],
      "IssueComments": ["issue_id", "repo_id", "author_id"], "PRReviews": ["pr_id", "repo_id", "reviewer_id"],
      "PRReviewComments": ["pr_id", "repo_id", "author_id", "pull_request_review_id", "in_reply_to_id"],
      "Pushes": ["user_id", "repo_id"]}


def group_of(f):
    if f.startswith(("cov_", "lag_")) or f.endswith("_adj"):
        return "data coverage"
    if f.startswith("global_"):
        return "cohort level"
    if f.startswith(("n_repos", "org_share", "peer_", "own_", "reviews_recv", "issue_comments_recv")):
        return "relational"
    return "own history"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", default="base", help="GBDT experiment whose feature importance is summarized")
    ap.add_argument("--n-valid", type=int, default=4)
    args = ap.parse_args()
    db = load_db()
    log = activity_log(db)
    data_end = log.seen.max()
    starts = weekly_starts(data_end)
    df = build_dataset(db, log, starts, cache=cache_path(data_end))
    weeks = starts[-args.n_valid:]
    va = df[df.start.isin(weeks)].reset_index(drop=True)
    res = {"data_end": str(data_end), "valid_weeks": [str(w) for w in weeks], "min_cov": MIN_COV}

    # 1. persistence: 1 if the user did the activity in the last N days before the cutoff
    pers = {}
    for w in [1, 3, 7, 14, 28, 90]:
        sc = week_scores(lambda m: pd.DataFrame({a: (va.loc[m, f"cnt_{a}_{w}d"].to_numpy() > 0).astype("int64")
                                                 for a in ACTS}), va, weeks)
        pers[f"{w}d"] = float(np.nanmean([s["score"] for s in sc.values()]))
    res["persistence_by_window"] = pers

    # 2. delay between creation and entering the data
    delay = {}
    for t, c in [("PullRequests", "created_at"), ("Issues", "created_at"), ("PRReviews", "submitted_at")]:
        d = (db[t].first_seen_at - db[t][c]).dropna() / pd.Timedelta(hours=1)
        delay[t] = {"share_over_24h": float((d > 24).mean()), "share_over_30d": float((d > 24 * 30).mean()),
                    "median_h": float(d.median())}
    res["row_time_delay"] = delay

    # 3. validation weeks: coverage and positive rate
    res["valid_week_stats"] = {
        str(s): {"users": int((va.start == s).sum()),
                 "coverage": {a: round(float(va.loc[va.start == s, f"lcov_{a}"].iloc[0]), 3) for a in ACTS},
                 "positive_rate": {a: round(float(va.loc[va.start == s, f"y_{a}"].mean()), 3) for a in ACTS}}
        for s in weeks}
    # weekly positive rate over all weeks (data degradation)
    res["weekly_positive_rate"] = {str(s): {a: round(float(g[f"y_{a}"].mean()), 3) for a in ACTS}
                                   for s, g in df.groupby("start")}

    # 4. feature-group share of LightGBM gain, per activity
    share = {}
    for a in ACTS:
        imp = pd.read_csv(GBDT_OUT / args.gbdt / f"importance_{a}.csv", index_col=0).iloc[:, 0]
        g = imp.groupby(imp.index.map(group_of)).sum()
        share[a] = (g / g.sum()).round(3).to_dict()
        share[a]["top5"] = imp.head(5).index.tolist()
    res["gbdt_gain_share"] = share

    # 5. relational entity graph size
    res["graph"] = {"nodes": int(sum(len(d) for d in db.values())),
                    "pk_fk_links": int(sum(db[t][c].notna().sum() for t, cs in FK.items() for c in cs)),
                    "node_types": len(db), "edge_types": 2 * sum(len(c) for c in FK.values())}

    (ROOT / "outputs" / "analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "weekly_positive_rate"}, indent=1))


if __name__ == "__main__":
    main()
