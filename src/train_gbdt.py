"""LightGBM baseline: one binary classifier per activity on point-in-time user features.

Samples: one per (eligible user, weekly cutoff). The label window starts at the same weekday and
hour as the official test week (Sat 16:00 UTC); features use data up to 4 hours earlier, like the
final data (up to 11-21 12:00 UTC) before the test week (11-21 16:00 UTC).

Run from the repository root:
    python src/train_gbdt.py --tag base
    python src/train_gbdt.py --tag hl8_score --half-life 8 --tune score
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import (ACTS, MIN_COV, WEEK, activity_log, eligible_users, label_coverage, load_db, make_labels, score,
                  scored_acts)
from features import FeatureBuilder

TEST_START = pd.Timestamp("2026-11-21 16:00")  # 11-22 00:00 Taiwan
GAP = pd.Timedelta(hours=4)
FIRST_START = pd.Timestamp("2026-01-14 16:00")
SEED = 42
OUT = Path(__file__).resolve().parents[1] / "outputs" / "gbdt"
YCOLS = [f"y_{a}" for a in ACTS]

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, num_threads=32)


def weekly_starts(data_end):
    """Label-window starts aligned to the test week whose 7-day window is fully inside the data."""
    starts, s = [], TEST_START
    while s >= FIRST_START:
        if s + WEEK <= data_end:
            starts.append(s)
        s -= WEEK
    return sorted(starts)


def build_dataset(db, log, starts, cache=None):
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    fb = FeatureBuilder(db, log)
    rows = []
    for s in starts:
        tf = s - GAP
        users = eligible_users(log, tf)
        X = fb.build(tf, users)
        y = make_labels(log, s, users)
        cov = label_coverage(log, s)
        rows.append(pd.concat([X, y.add_prefix("y_")], axis=1).assign(start=s, **{f"lcov_{a}": c for a, c in cov.items()})
                    .reset_index())
        print(f"  {s}  users={len(users)}  pos={y.mean().round(3).to_dict()}  "
              f"label coverage={ {a: round(c, 2) for a, c in cov.items()} }", flush=True)
    df = pd.concat(rows, ignore_index=True)
    if cache is not None:
        df.to_parquet(cache)
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in ("github_user_id", "start") and not c.startswith(("y_", "lcov_"))]


def cache_path(data_end):
    return OUT / f"features_v3_{data_end:%Y%m%d}.parquet"


def covered(df):
    """Per activity, the rows whose label window has enough data to be scored (official rule)."""
    return {a: (df[f"lcov_{a}"] >= MIN_COV).to_numpy() for a in ACTS}


def week_scores(pred_fn, df, weeks):
    """Score per validation week over the activities with complete data in that week.
    pred_fn(mask) -> 0/1 DataFrame for the rows in mask."""
    out = {}
    for s in weeks:
        m = (df.start == s).to_numpy()
        t = df.loc[m, YCOLS].set_axis(ACTS, axis=1).reset_index(drop=True)
        out[str(s)] = score(pred_fn(m), t, scored_acts(df[m]))
    return out


def sample_weight(starts, half_life):
    """Exponential decay by age in weeks relative to the newest training week (1.0 = no decay)."""
    if not half_life:
        return None
    age = (starts.max() - starts) / WEEK
    return (0.5 ** (age / half_life)).to_numpy()


def fit(tr, feats, act, half_life, num_rounds, min_cov, valid=None):
    """Train on rows whose label window for this activity has coverage >= min_cov."""
    tr = tr[tr[f"lcov_{act}"] >= min_cov]
    dtr = lgb.Dataset(tr[feats], tr[f"y_{act}"], weight=sample_weight(tr.start, half_life))
    if valid is None:
        return lgb.train(PARAMS, dtr, num_rounds)
    valid = valid[valid[f"lcov_{act}"] >= min_cov]
    if valid.empty:  # no validation week has complete data for this activity: no early stopping
        return lgb.train(PARAMS, dtr, FALLBACK_ROUNDS)
    dva = lgb.Dataset(valid[feats], valid[f"y_{act}"])
    return lgb.train(PARAMS, dtr, num_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


FALLBACK_ROUNDS = 200  # boosting rounds when early stopping is impossible
GRID = np.round(np.arange(0.04, 0.80, 0.01), 2)


def f1_thresholds(prob, truth, keep=None):
    """Per activity, the threshold with the best F1 (on the rows in keep[a], if given)."""
    out = {}
    for a in ACTS:
        p, y = np.asarray(prob[a]), truth[a].to_numpy()
        if keep is not None and keep[a].any():
            p, y = p[keep[a]], y[keep[a]]
        f1 = [2 * ((p >= g) & (y == 1)).sum() / max((p >= g).sum() + (y == 1).sum(), 1) for g in GRID]
        out[a] = float(GRID[int(np.argmax(f1))])
    return out


def predict(prob, thr):
    return pd.DataFrame({a: (prob[a] >= thr[a]).astype("int64") for a in ACTS})


def score_thresholds(prob, truth, rounds=3):
    """Coordinate ascent on the final Score (macro + micro), starting from per-activity F1 thresholds."""
    thr = f1_thresholds(prob, truth)
    for _ in range(rounds):
        for a in ACTS:
            best = (score(predict(prob, thr), truth)["score"], thr[a])
            for g in GRID:
                s = score(predict(prob, {**thr, a: float(g)}), truth)["score"]
                if s > best[0]:
                    best = (s, float(g))
            thr[a] = best[1]
    return thr


def tune(prob, truth, how, keep=None):
    truth = truth.reset_index(drop=True)
    return score_thresholds(prob, truth) if how == "score" else f1_thresholds(prob, truth, keep)


def save_prob(keys, prob, path):
    """Predicted probabilities per sample (for ensembling): key columns + p_<activity>."""
    keys.reset_index(drop=True).assign(**{f"p_{a}": np.asarray(prob[a]) for a in ACTS}).to_parquet(path)


def persistence(df, days=14):
    return pd.DataFrame({a: (df[f"cnt_{a}_{days}d"].to_numpy() > 0).astype("int64") for a in ACTS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base", help="name of this experiment (output sub-folder)")
    ap.add_argument("--n-valid", type=int, default=4, help="number of most recent weeks held out for validation")
    ap.add_argument("--half-life", type=float, default=0, help="sample-weight half-life in weeks (0 = no decay)")
    ap.add_argument("--tune", choices=["f1", "score"], default="f1", help="threshold objective")
    ap.add_argument("--min-cov", type=float, default=0.0,
                    help="drop labels whose window has less data coverage than this (1.0 = no gaps); 0 keeps all")
    args = ap.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    db = load_db()
    log = activity_log(db)
    data_end = log.seen.max()
    starts = weekly_starts(data_end)
    print(f"data until {data_end}; {len(starts)} weekly samples {starts[0]} .. {starts[-1]}")

    df = build_dataset(db, log, starts, cache=cache_path(data_end))
    feats = feature_cols(df)
    valid_starts = starts[-args.n_valid:]
    tr, va = df[~df.start.isin(valid_starts)], df[df.start.isin(valid_starts)].reset_index(drop=True)
    print(f"train rows={len(tr)} valid rows={len(va)} features={len(feats)}")

    prob, best_iter = {}, {}
    for a in ACTS:
        m = fit(tr, feats, a, args.half_life, 3000, args.min_cov, valid=va)
        best_iter[a] = m.best_iteration or m.current_iteration()
        prob[a] = m.predict(va[feats], num_iteration=best_iter[a])
        pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False).to_csv(out / f"importance_{a}.csv")
    print("best iterations:", best_iter)
    save_prob(va[["github_user_id", "start"]], prob, out / "valid_prob.parquet")

    truth = va[YCOLS].set_axis(ACTS, axis=1)
    keep = covered(va)
    thr = tune(prob, truth, args.tune, keep)
    print(f"thresholds (tuned on validation labels with coverage >= {MIN_COV}):", thr)

    print("\n== validation (each week scored over its activities with complete data) ==")
    sm = week_scores(lambda m: predict({a: prob[a][m] for a in ACTS}, thr), va, valid_starts)
    sb = week_scores(lambda m: persistence(va[m]), va, valid_starts)
    weeks = {}
    for s in valid_starts:
        k = str(s)
        m = (va.start == s).to_numpy()
        cov = {a: round(float(va.loc[m, f"lcov_{a}"].iloc[0]), 2) for a in ACTS}
        weeks[k] = {"gbdt": sm[k], "persist14": sb[k], "label_coverage": cov}
        skipped = [a for a in ACTS if a not in sm[k]["acts"]]
        print(f"{s}  gbdt={sm[k]['score']:.4f} (macro {sm[k]['macro_f1']:.4f} micro {sm[k]['micro_f1']:.4f})   "
              f"persist14={sb[k]['score']:.4f}" + (f"  [not scored: {', '.join(skipped)}]" if skipped else ""))

    # honest check: thresholds tuned on the earlier weeks, scored on the last week
    early, last = (va.start < valid_starts[-1]).to_numpy(), (va.start == valid_starts[-1]).to_numpy()
    thr_h = tune({a: prob[a][early] for a in ACTS}, truth[early], args.tune,
                 {a: v[early] for a, v in keep.items()})
    hs = week_scores(lambda m: predict({a: prob[a][m] for a in ACTS}, thr_h), va, [valid_starts[-1]])
    honest = {"week": str(valid_starts[-1]), "gbdt": hs[str(valid_starts[-1])],
              "persist14": sb[str(valid_starts[-1])]}
    print(f"\nhonest check on {valid_starts[-1]}: gbdt={honest['gbdt']['score']:.4f}  "
          f"persist14={honest['persist14']['score']:.4f}")
    mean = float(np.nanmean([w["gbdt"]["score"] for w in weeks.values()]))
    mean_b = float(np.nanmean([w["persist14"]["score"] for w in weeks.values()]))
    print(f"mean over validation weeks: gbdt={mean:.4f}  persist14={mean_b:.4f}")

    with open(out / "valid_report.json", "w") as fh:
        json.dump({"args": vars(args), "data_end": str(data_end), "best_iter": best_iter, "threshold": thr,
                   "mean_valid_score": mean, "mean_valid_persist14": mean_b, "honest": honest, "weeks": weeks},
                  fh, indent=2, default=float)
    print(f"report saved to {out / 'valid_report.json'}")


if __name__ == "__main__":
    main()
