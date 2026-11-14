"""Exploratory data analysis and label sanity check (against the course reference score).

Run from the repository root:  python src/eda.py
"""
from pathlib import Path

import pandas as pd

from data import ACTS, ROW_TIME, WEEK, activity_log, eligible_users, load_db, make_labels, score

# Reference week of the course README (2026/09/27 - 10/04, Taiwan time) and its "Copy last week" score.
SAMPLE_START = pd.Timestamp("2026-09-26 16:00")
REF_COPY_LAST_WEEK = {"macro_f1": 0.4301, "micro_f1": 0.5620, "score": 0.4961}
DATA_START = pd.Timestamp("2025-12-01")

pd.set_option("display.width", 200)


def main():
    db = load_db()

    print("== Tables ==")
    for n, df in db.items():
        c = ROW_TIME[n]
        print(f"{n:17s} rows={len(df):>9,}  {df[c].min()} -> {df[c].max()}")
    u = db["Users"]
    print(f"cohort users: {u.is_cohort.sum()}  bots: {u.is_bot.sum()}")

    log = activity_log(db)
    print("\nactivity records per type:", log.act.value_counts().to_dict())

    print("\n== Label check: reproduce the course reference score of Copy last week ==")
    print("course README:", REF_COPY_LAST_WEEK)
    users = eligible_users(log, SAMPLE_START)
    truth = make_labels(log, SAMPLE_START, users)
    print(f"users={len(users)}  positive rate:", truth[ACTS].mean().round(3).to_dict())

    print("\n== Persistence baseline (predict 1 if done in the previous N days) ==")
    for days in [7, 14, 21, 28]:
        hist = log[(log.t >= SAMPLE_START - pd.Timedelta(days=days)) & (log.t < SAMPLE_START)]
        pred = (pd.crosstab(hist.github_user_id, hist.act)
                .reindex(index=truth.index, columns=ACTS, fill_value=0) > 0).astype("int64")
        s = score(pred, truth)
        print(f"last {days:2d}d: macro={s['macro_f1']:.4f} micro={s['micro_f1']:.4f} score={s['score']:.4f}")

    print("\n== Weekly active cohort users per activity (weeks aligned to the sample week start) ==")
    w = log[log.t >= DATA_START].copy()
    w["week"] = SAMPLE_START + ((w.t - SAMPLE_START) // WEEK) * WEEK
    print(w.groupby(["week", "act"]).github_user_id.nunique().unstack(fill_value=0)[ACTS].to_string())


if __name__ == "__main__":
    main()
