"""The 8 tables as a RelBench Database for Relational Deep Learning.

Each table keeps its primary key, its foreign keys (graph edges) and its row-time column
(the node time used by temporal neighbor sampling). Columns known only later
(merged_at, merged_by_id, closed_at, state_reason, dismissed_at) are dropped, so no edge or
feature can reveal the future. Text is reduced to its length.
"""
import pandas as pd
from relbench.base import Database, Table
from torch_frame import stype

H = pd.Timedelta(hours=1)
C, N = stype.categorical, stype.numerical


def _ns(s):
    return s.astype("datetime64[ns]")


def _flag(s):
    """Nullable bool -> category string, keeping 'unknown' as its own value."""
    return s.map({True: "t", False: "f"}).fillna("na")


def make_database(db):
    U, R, PR, IS = db["Users"], db["Repositories"], db["PullRequests"], db["Issues"]
    IC, RV, RC, P = db["IssueComments"], db["PRReviews"], db["PRReviewComments"], db["Pushes"]

    tables = {
        "Users": (pd.DataFrame({
            "user_id": U.user_id, "is_bot": U.is_bot.astype(int), "is_cohort": U.is_cohort.astype(int),
            "first_seen_at": _ns(U.first_seen_at)}),
            {}, "user_id", "first_seen_at"),
        "Repositories": (pd.DataFrame({
            "repo_id": R.repo_id, "has_org": R.org_id.notna().astype(int),
            "first_seen_at": _ns(R.first_seen_at)}),
            {}, "repo_id", "first_seen_at"),
        "PullRequests": (pd.DataFrame({
            "pr_id": PR.pr_id, "repo_id": PR.repo_id, "author_id": PR.author_id,
            "is_cross_repo": _flag(PR.is_cross_repo), "draft": _flag(PR.draft),
            "opened_seen": PR.created_at.notna().astype(int),
            "opened_before_seen_h": ((PR.first_seen_at - PR.created_at) / H).clip(lower=0),
            "first_seen_at": _ns(PR.first_seen_at)}),
            {"repo_id": "Repositories", "author_id": "Users"}, "pr_id", "first_seen_at"),
        "Issues": (pd.DataFrame({
            "issue_id": IS.issue_id, "repo_id": IS.repo_id, "author_id": IS.author_id,
            "assignee_id": IS.assignee_id, "pull_request_id": IS.pull_request_id,
            "is_pull_request": IS.is_pull_request.astype(int), "num_labels": IS.num_labels.astype(float),
            "issue_type": IS.issue_type.fillna("na"), "locked": _flag(IS.locked),
            "created_via_app": IS.created_via_app.fillna("none"),
            "age_at_seen_d": ((IS.first_seen_at - IS.created_at) / (24 * H)).clip(lower=0),
            "first_seen_at": _ns(IS.first_seen_at)}),
            {"repo_id": "Repositories", "author_id": "Users", "assignee_id": "Users",
             "pull_request_id": "PullRequests"}, "issue_id", "first_seen_at"),
        "IssueComments": (pd.DataFrame({
            "comment_id": IC.comment_id, "issue_id": IC.issue_id, "repo_id": IC.repo_id, "author_id": IC.author_id,
            "is_pull_request": IC.is_pull_request.astype(int), "via_app": IC.via_app.fillna("none"),
            "body_len": IC.body.str.len().astype(float), "created_at": _ns(IC.created_at)}),
            {"issue_id": "Issues", "repo_id": "Repositories", "author_id": "Users"}, "comment_id", "created_at"),
        "PRReviews": (pd.DataFrame({
            "review_id": RV.review_id, "pr_id": RV.pr_id, "repo_id": RV.repo_id, "reviewer_id": RV.reviewer_id,
            "state": RV.state.fillna("na"), "body_len": RV.body.str.len().astype(float),
            "submitted_before_seen_h": ((RV.first_seen_at - RV.submitted_at) / H).clip(lower=0),
            "first_seen_at": _ns(RV.first_seen_at)}),
            {"pr_id": "PullRequests", "repo_id": "Repositories", "reviewer_id": "Users"}, "review_id", "first_seen_at"),
        "PRReviewComments": (pd.DataFrame({
            "comment_id": RC.comment_id, "pr_id": RC.pr_id, "repo_id": RC.repo_id, "author_id": RC.author_id,
            "pull_request_review_id": RC.pull_request_review_id, "in_reply_to_id": RC.in_reply_to_id,
            "subject_type": RC.subject_type.fillna("na"), "file_extension": RC.file_extension.fillna("none"),
            "is_outdated": RC.position.isna().astype(int), "body_len": RC.body.str.len().astype(float),
            "created_at": _ns(RC.created_at)}),
            {"pr_id": "PullRequests", "repo_id": "Repositories", "author_id": "Users",
             "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments"}, "comment_id", "created_at"),
        "Pushes": (pd.DataFrame({
            "push_id": P.push_id, "user_id": P.user_id, "repo_id": P.repo_id,
            "ref_type": P.ref_type.fillna("na"), "is_new_ref": P.is_new_ref.astype(int),
            "branch_len": P.branch.str.len().astype(float), "created_at": _ns(P.created_at)}),
            {"user_id": "Users", "repo_id": "Repositories"}, "push_id", "created_at"),
    }
    # missing numbers -> 0 plus a missing-flag column (NaN would otherwise reach the encoder)
    for name, (df, fk, pk, tc) in tables.items():
        for c in [c for c in df.columns if c not in (pk, tc, *fk) and df[c].dtype.kind == "f" and df[c].isna().any()]:
            df[f"{c}_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(0.0)
    rdb = Database({name: Table(df, fk, pk, tc) for name, (df, fk, pk, tc) in tables.items()})

    # stype of every feature column (keys and time columns are not features)
    col_to_stype = {}
    for name, (df, fk, pk, tc) in tables.items():
        cols = [c for c in df.columns if c not in (pk, tc, *fk)]
        col_to_stype[name] = {c: N if df[c].dtype.kind == "f" else C for c in cols}
    return rdb, col_to_stype
