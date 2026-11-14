"""Shared data utilities: loading the 8 tables, point-in-time filtering, labels and scoring."""
from pathlib import Path

import numpy as np
import pandas as pd

import os

ROOT = Path(__file__).resolve().parents[1]
# Tables written by the course preprocessing script: python assignment2/scripts/preprocess.py --raw gharchive --out data
DB_DIR = Path(os.environ.get("A2_DB", ROOT / "data" / "db"))
COURSE_DIR = ROOT / "assignment2"  # the course repository (scripts/, submission/)
EVAL_DIR = COURSE_DIR / "submission"  # test_input.csv
EVALUATE = COURSE_DIR / "scripts" / "evaluate.py"

TABLES = ["Users", "Repositories", "PullRequests", "Issues",
          "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]

# Row-time column of each table: a row exists in the data from this time on.
ROW_TIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
            "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
            "PRReviewComments": "created_at", "Pushes": "created_at"}

# Columns filled in only after the row was created; their value is usable only if earlier than t.
# Each entry: column -> time column that tells when the value became known.
KNOWN_LATER = {"PullRequests": {"merged_at": "merged_at", "merged_by_id": "merged_at", "closed_at": "closed_at"},
               "Issues": {"closed_at": "closed_at", "state_reason": "closed_at"},
               "PRReviews": {"dismissed_at": "dismissed_at"}}

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]

# How each target activity is read from the tables: (table, user column, activity time column, row filter).
# Verified against evaluation/examples/sample_truth.csv (window 2026-09-27 13:00 UTC + 7 days).
ACT_SOURCE = {
    "open_pr": ("PullRequests", "author_id", "created_at", None),
    "open_issue": ("Issues", "author_id", "created_at", lambda df: ~df.is_pull_request),
    "issue_comment": ("IssueComments", "author_id", "created_at", None),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at", None),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at", None),
    "push": ("Pushes", "user_id", "created_at", None),
}

WEEK = pd.Timedelta(days=7)

# GH Archive has hours with missing events. As in the official scoring ("if GH Archive is missing the
# data of an activity during the test week, that activity is not scored"), an activity counts in a
# week only if at least MIN_COV of the week's hours lie outside collection gaps (see label_coverage).
MIN_COV = 0.95


def load_db(db_dir=DB_DIR):
    return {n: pd.read_parquet(Path(db_dir) / f"{n}.parquet") for n in TABLES}


def snapshot(db, t):
    """The database as it was known at time t: rows with row time < t, known-later values masked."""
    t = pd.Timestamp(t)
    out = {}
    for n, df in db.items():
        df = df[df[ROW_TIME[n]] < t].copy()
        for col, when in KNOWN_LATER.get(n, {}).items():
            df.loc[~(df[when] < t), col] = None
        out[n] = df
    return out


def cohort_map(db):
    """Users.user_id (row index) -> github_user_id, cohort users only."""
    u = db["Users"]
    return u.loc[u.is_cohort].set_index("user_id").github_user_id


def activity_log(db):
    """One row per target activity by a cohort user: github_user_id, act, t, seen, repo_id.

    t is when the activity happened (used for labels); seen is the row time, when the record
    enters the data. A feature at time t0 may only use records with seen < t0.
    """
    coh = cohort_map(db)
    parts = []
    for act, (table, ucol, tcol, flt) in ACT_SOURCE.items():
        df = db[table]
        if flt is not None:
            df = df[flt(df)]
        df = df[list(dict.fromkeys([ucol, tcol, ROW_TIME[table], "repo_id"]))].dropna(subset=[ucol, tcol])
        gid = df[ucol].astype("int64").map(coh)
        parts.append(pd.DataFrame({"github_user_id": gid, "act": act, "t": df[tcol],
                                   "seen": df[ROW_TIME[table]], "repo_id": df.repo_id})
                     .dropna(subset=["github_user_id"]))
    log = pd.concat(parts, ignore_index=True)
    log["github_user_id"] = log.github_user_id.astype("int64")
    return log


def make_labels(log, start, users):
    """0/1 label for each user and activity in [start, start + 7 days)."""
    start = pd.Timestamp(start)
    w = log[(log.t >= start) & (log.t < start + WEEK)]
    lab = pd.crosstab(w.github_user_id, w.act).reindex(index=users, columns=ACTS, fill_value=0)
    return (lab > 0).astype("int64")


def missing_hours(times, start, end, min_gap=6):
    """Boolean per hour of [start, end): True if the hour lies in a run of >= min_gap hours with no record.

    The data has collection gaps (whole days without e.g. pushes); quiet hours are common for
    sparse activities, so only long empty runs count as missing.
    """
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    t = times[(times >= start) & (times < end)]
    empty = t.dt.floor("h").value_counts().reindex(hours, fill_value=0).to_numpy() == 0
    miss = np.zeros(len(hours), dtype=bool)
    i = 0
    while i < len(empty):
        if empty[i]:
            j = i
            while j < len(empty) and empty[j]:
                j += 1
            if j - i >= min_gap:
                miss[i:j] = True
            i = j
        else:
            i += 1
    return miss


def label_coverage(log, start):
    """Share of the label window [start, start + 7 days) without collection gaps, per activity."""
    start = pd.Timestamp(start)
    return {a: 1 - missing_hours(log.t[log.act == a], start, start + WEEK).mean() for a in ACTS}


def eligible_users(log, t, days=90, min_records=5):
    """Cohort users with >= min_records activity records in the `days` before t (test-set rule)."""
    t = pd.Timestamp(t)
    w = log[(log.t >= t - pd.Timedelta(days=days)) & (log.t < t) & (log.seen < t)]
    n = w.groupby("github_user_id").size()
    return n[n >= min_records].index.sort_values()


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def scored_acts(rows, min_cov=MIN_COV):
    """Activities scored in one week: rows are that week's samples, with lcov_<activity> columns."""
    return [a for a in ACTS if rows[f"lcov_{a}"].iloc[0] >= min_cov]


def score(pred, truth, acts=None):
    """Same metric as scripts/evaluate.py, over `acts` (default: all 6 activities).
    pred and truth: DataFrames indexed by user with activity columns."""
    acts = ACTS if acts is None else list(acts)
    if not acts:
        return {"macro_f1": np.nan, "micro_f1": np.nan, "score": np.nan, "per_act": {}, "acts": []}
    p = pred.loc[truth.index, acts].to_numpy()
    t = truth[acts].to_numpy()
    tp, fp, fn = ((p == 1) & (t == 1)).sum(0), ((p == 1) & (t == 0)).sum(0), ((p == 0) & (t == 1)).sum(0)
    per = np.array([_f1(*c) for c in zip(tp, fp, fn)])
    micro = _f1(tp.sum(), fp.sum(), fn.sum())
    return {"macro_f1": per.mean(), "micro_f1": micro, "score": (per.mean() + micro) / 2,
            "per_act": dict(zip(acts, per.round(4))), "acts": acts}
