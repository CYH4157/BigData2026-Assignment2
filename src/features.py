"""Point-in-time user features for the GBDT baseline.

Every feature for cutoff tf uses only records with row time (seen) < tf and known-later
columns only when their time is < tf.
"""
import numpy as np
import pandas as pd

from data import ACTS, cohort_map, missing_hours

DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28, 90]


class FeatureBuilder:
    def __init__(self, db, log):
        coh = cohort_map(db)
        self.n_cohort = len(coh)
        to_gid = lambda s: s.astype("float64").map(coh)

        pr = db["PullRequests"]
        self.pr = pd.DataFrame({"pr_id": pr.pr_id, "author": to_gid(pr.author_id), "created": pr.created_at,
                                "seen": pr.first_seen_at, "merged": pr.merged_at, "closed": pr.closed_at})
        self.pr = self.pr.dropna(subset=["author", "created"])

        rv = db["PRReviews"].merge(pr[["pr_id", "author_id"]], on="pr_id")
        self.rv_recv = pd.DataFrame({"author": to_gid(rv.author_id), "reviewer": rv.reviewer_id,
                                     "author_uid": rv.author_id, "t": rv.submitted_at, "seen": rv.first_seen_at})
        self.rv_recv = self.rv_recv[self.rv_recv.author.notna() & (self.rv_recv.reviewer != self.rv_recv.author_uid)]

        iss = db["Issues"]
        ic = db["IssueComments"].merge(iss[["issue_id", "author_id"]].rename(columns={"author_id": "issue_author"}),
                                       on="issue_id")
        self.ic_recv = pd.DataFrame({"author": to_gid(ic.issue_author), "t": ic.created_at})[
            (ic.author_id != ic.issue_author).to_numpy()].dropna(subset=["author"])

        org = db["Repositories"].set_index("repo_id").org_id.notna()
        self.log = log.assign(org=log.repo_id.map(org).fillna(False).astype("int8"))

    def build(self, tf, users):
        tf = pd.Timestamp(tf)
        h = self.log[(self.log.seen < tf) & (self.log.t < tf)]
        h90 = h[h.t >= tf - 90 * DAY]
        f = pd.DataFrame(index=pd.Index(users, name="github_user_id"))

        # activity counts per window
        age = (tf - h90.t) / DAY
        for w in WINDOWS:
            c = pd.crosstab(h90.github_user_id[age < w], h90.act[age < w])
            c = c.reindex(index=f.index, columns=ACTS, fill_value=0)
            for a in ACTS:
                f[f"cnt_{a}_{w}d"] = c[a].to_numpy()
        f["cnt_all_7d"] = f[[f"cnt_{a}_7d" for a in ACTS]].sum(1)
        f["cnt_all_28d"] = f[[f"cnt_{a}_28d" for a in ACTS]].sum(1)
        f["trend_7_28"] = f.cnt_all_7d / (f.cnt_all_28d / 4 + 1)

        # recency
        last = h.groupby(["github_user_id", "act"]).t.max().unstack().reindex(index=f.index, columns=ACTS)
        for a in ACTS:
            f[f"days_since_{a}"] = ((tf - last[a]) / DAY).to_numpy()
        f["days_since_any"] = f[[f"days_since_{a}" for a in ACTS]].min(1)
        first = h.groupby("github_user_id").t.min().reindex(f.index)
        f["days_since_first"] = ((tf - first) / DAY).to_numpy()

        # regularity: distinct active days
        day = h90.t.dt.floor("D")
        for w in [7, 28, 90]:
            m = age < w
            f[f"active_days_{w}d"] = (day[m].groupby(h90.github_user_id[m]).nunique()
                                      .reindex(f.index, fill_value=0).to_numpy())

        # repositories
        h28 = h90[age < 28]
        f["n_repos_7d"] = h90[age < 7].groupby("github_user_id").repo_id.nunique().reindex(f.index, fill_value=0)
        f["n_repos_28d"] = h28.groupby("github_user_id").repo_id.nunique().reindex(f.index, fill_value=0)
        f["org_share_28d"] = h28.groupby("github_user_id").org.mean().reindex(f.index)

        # peers: other cohort users' activity in the repos this user worked on in the last 28 days
        h7 = h90[age < 7]
        repo7 = h7.groupby("repo_id").size()
        own7 = h7.groupby(["github_user_id", "repo_id"]).size()
        ur = h28[["github_user_id", "repo_id"]].drop_duplicates()
        ur = ur.assign(tot=ur.repo_id.map(repo7).fillna(0).to_numpy(),
                       own=own7.reindex(pd.MultiIndex.from_frame(ur)).fillna(0).to_numpy())
        f["peer_act_7d"] = (ur.tot - ur.own).groupby(ur.github_user_id).sum().reindex(f.index, fill_value=0)

        # the user's own PRs: open, recently merged, reviews received
        pr = self.pr[(self.pr.seen < tf) & (self.pr.created < tf) & (self.pr.created >= tf - 90 * DAY)]
        is_open = ~(pr.closed < tf)
        f["own_open_prs"] = pr[is_open].groupby("author").size().reindex(f.index, fill_value=0)
        f["own_merged_prs_14d"] = (pr[(pr.merged < tf) & (pr.merged >= tf - 14 * DAY)]
                                   .groupby("author").size().reindex(f.index, fill_value=0))
        rv = self.rv_recv[(self.rv_recv.seen < tf) & (self.rv_recv.t < tf) & (self.rv_recv.t >= tf - 14 * DAY)]
        f["reviews_recv_14d"] = rv.groupby("author").size().reindex(f.index, fill_value=0)
        ic = self.ic_recv[(self.ic_recv.t < tf) & (self.ic_recv.t >= tf - 14 * DAY)]
        f["issue_comments_recv_14d"] = ic.groupby("author").size().reindex(f.index, fill_value=0)

        # global level of the whole cohort (captures seasonal drift)
        act7 = h7.groupby("act").github_user_id.nunique().reindex(ACTS, fill_value=0)
        for a in ACTS:
            f[f"global_{a}_7d"] = act7[a] / self.n_cohort

        # data coverage: collection gaps make recent counts and recency look lower than they are.
        # cov = share of hours with data, lag = hours since the last record of this activity (any user).
        for a in ACTS:
            times = h.t[h.act == a]
            miss = missing_hours(times, tf - 28 * DAY, tf)
            cov7, cov28 = 1 - miss[-24 * 7:].mean(), 1 - miss.mean()
            lag = (tf - times.max()) / pd.Timedelta(hours=1) if len(times) else np.nan
            f[f"cov_{a}_7d"], f[f"cov_{a}_28d"], f[f"lag_{a}_h"] = cov7, cov28, lag
            f[f"days_since_{a}_adj"] = f[f"days_since_{a}"] - lag / 24
            f[f"cnt_{a}_7d_adj"] = f[f"cnt_{a}_7d"] / max(cov7, 0.1)
            f[f"cnt_{a}_28d_adj"] = f[f"cnt_{a}_28d"] / max(cov28, 0.1)

        return f.astype("float32")
