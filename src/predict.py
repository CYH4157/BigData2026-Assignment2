"""Train the GBDT models on all available weeks and write the submission file.

Uses the number of boosting rounds, thresholds and settings chosen by train_gbdt.py (--tag).
Features are taken at 11-11 12:00 UTC (the final release); with an earlier release, at the
end of the available data (a dry run of the pipeline and the output format).

Run from the repository root:
    python src/predict.py --tag base --out outputs/output_<studentID>.csv
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data import ACTS, DB_DIR, EVAL_DIR, EVALUATE, activity_log, eligible_users, load_db
from features import FeatureBuilder
from train_gbdt import (GAP, OUT, TEST_START, build_dataset, cache_path, feature_cols, fit, predict,
                        save_prob, weekly_starts)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base", help="experiment whose settings are used (outputs/gbdt/<tag>)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "output.csv"))
    args = ap.parse_args()
    report = json.loads((OUT / args.tag / "valid_report.json").read_text())
    half_life, min_cov = report["args"]["half_life"], report["args"].get("min_cov", 0)
    best_iter, thr = report["best_iter"], report["threshold"]

    db = load_db()
    log = activity_log(db)
    data_end = log.seen.max()
    starts = weekly_starts(data_end)
    df = build_dataset(db, log, starts, cache=cache_path(data_end))
    feats = feature_cols(df)

    tf = TEST_START - GAP
    if data_end < tf - pd.Timedelta(hours=1):
        tf = data_end.ceil("h")
        print(f"WARNING: not the final release; features taken at {tf} (dry run)")
    print(f"training on {len(starts)} weeks ({len(df)} rows), features at {tf}")

    users = pd.read_csv(EVAL_DIR / "test_input.csv").UserID
    elig = set(eligible_users(log, tf))
    print(f"test users={len(users)}  not eligible at tf={len(set(users) - elig)}")
    X = FeatureBuilder(db, log).build(tf, users)

    prob = {}
    for a in ACTS:
        m = fit(df, feats, a, half_life, best_iter[a], min_cov)
        prob[a] = m.predict(X[feats])
    save_prob(pd.DataFrame({"UserID": users.to_numpy()}), prob, OUT / args.tag / "test_prob.parquet")
    pred = predict(prob, thr)
    pred.insert(0, "UserID", users.to_numpy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"saved {out}  positive rate: {pred[ACTS].mean().round(3).to_dict()}")

    subprocess.run([sys.executable, str(EVALUATE), "--pred", str(out),
                    "--input", str(EVAL_DIR / "test_input.csv")], check=True)


if __name__ == "__main__":
    main()
