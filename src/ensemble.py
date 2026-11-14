"""Weighted average of GBDT and RDL probabilities, thresholds tuned on the validation weeks.

Validation probabilities come from models trained without the validation weeks; test
probabilities from the final models (GBDT: predict.py, RDL: train_rdl.py --final). RDL runs
with several seeds are averaged first.

Run from the repository root:
    python src/ensemble.py --gbdt base --rdl-valid l3_s42 l3_s43 l3_s44 --rdl-test l3_final_s42 l3_final_s43 l3_final_s44 \
        --out outputs/output_<studentID>.csv
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import ACTS, DB_DIR, EVAL_DIR, EVALUATE, score
from train_gbdt import OUT as GBDT_OUT, YCOLS, cache_path, covered, persistence, predict, tune, week_scores

ROOT = Path(__file__).resolve().parents[1]
RDL_OUT = ROOT / "outputs" / "rdl"
P = [f"p_{a}" for a in ACTS]
KEY = ["github_user_id", "start"]


def mean_prob(paths, key):
    """Average of several runs' probabilities; every run must cover exactly the same samples."""
    frames = [pd.read_parquet(p).set_index(key)[P].sort_index() for p in paths]
    for p, f in zip(paths[1:], frames[1:]):
        if not f.index.equals(frames[0].index):
            raise ValueError(f"{p} does not have the same samples as {paths[0]} (stale or failed run?)")
    return sum(frames) / len(frames)


def weekly(prob, va, weeks, thr):
    return week_scores(lambda m: predict({a: prob[a][m] for a in ACTS}, thr), va, weeks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", default="base", help="GBDT experiment tag (outputs/gbdt/<tag>)")
    ap.add_argument("--rdl-valid", nargs="+", required=True, help="RDL validation-mode tags (seeds are averaged)")
    ap.add_argument("--rdl-test", nargs="+", help="RDL final-mode tags for the test probabilities")
    ap.add_argument("--tune", choices=["f1", "score"], default="f1")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "output.csv"))
    args = ap.parse_args()

    report = json.loads((GBDT_OUT / args.gbdt / "valid_report.json").read_text())
    df = pd.read_parquet(cache_path(pd.Timestamp(report["data_end"])))
    g = mean_prob([GBDT_OUT / args.gbdt / "valid_prob.parquet"], KEY)
    r = mean_prob([RDL_OUT / t / "valid_prob.parquet" for t in args.rdl_valid], KEY)
    va = g.join(r, lsuffix="_g", rsuffix="_r", how="inner").reset_index().merge(df, on=KEY, how="left")
    if len(va) != len(g) or len(va) != len(r):
        raise ValueError(f"GBDT ({len(g)}) and RDL ({len(r)}) validation samples do not match ({len(va)} in common)")
    weeks = sorted(va.start.unique())
    print(f"validation samples={len(va)} weeks={[str(w) for w in weeks]}")
    truth = va[YCOLS].set_axis(ACTS, axis=1)

    def blend(w, mask=slice(None)):
        return {a: w * va[f"p_{a}_g"].to_numpy()[mask] + (1 - w) * va[f"p_{a}_r"].to_numpy()[mask] for a in ACTS}

    print("\n== weight of GBDT (0 = RDL only, 1 = GBDT only), thresholds tuned on all validation weeks ==")
    res = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        p = blend(w)
        thr = tune(p, truth, args.tune, covered(va))
        wk = weekly(p, va, weeks, thr)
        res[float(w)] = (np.nanmean([s["score"] for s in wk.values()]), thr, wk)
        per_week = " ".join(format(s["score"], ".4f") for s in wk.values())
        print(f"w={w:.1f}  mean={res[float(w)][0]:.4f}  [{per_week}]")
    best_w = max(res, key=lambda k: res[k][0])
    mean, thr, wk = res[best_w]
    base = np.nanmean([v["score"] for v in week_scores(lambda m: persistence(va[m]), va, weeks).values()])
    print(f"best w={best_w}: mean={mean:.4f} (persist14 {base:.4f})")

    # honest check: weight and thresholds chosen on the earlier weeks, scored on the last week
    early, last = (va.start < weeks[-1]).to_numpy(), (va.start == weeks[-1]).to_numpy()
    keep = covered(va)
    va_e = va[early].reset_index(drop=True)
    cand = {}
    for w in res:
        p = blend(w, early)
        th = tune(p, truth[early], args.tune, {a: v[early] for a, v in keep.items()})
        cand[w] = (np.nanmean([s["score"] for s in weekly(p, va_e, weeks[:-1], th).values()]), th)
    hw = max(cand, key=lambda k: cand[k][0])
    va_l = va[last].reset_index(drop=True)
    honest, honest_g, honest_r = (weekly(blend(w, last), va_l, [weeks[-1]], cand[w][1])[str(weeks[-1])]
                                  for w in (hw, 1.0, 0.0))
    print(f"honest check on {weeks[-1]}: ensemble (w={hw}) {honest['score']:.4f}  "
          f"GBDT only {honest_g['score']:.4f}  RDL only {honest_r['score']:.4f}")

    out_dir = ROOT / "outputs" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "valid_report.json").write_text(json.dumps(
        {"args": vars(args), "weight_gbdt": best_w, "threshold": thr, "mean_valid_score": mean,
         "mean_valid_persist14": base, "weeks": wk,
         "by_weight": {str(w): v[0] for w, v in res.items()},
         "honest": {"week": str(weeks[-1]), "weight_gbdt": hw, "ensemble": honest, "gbdt": honest_g, "rdl": honest_r}},
        indent=2, default=float))

    if not args.rdl_test:
        return
    gt = mean_prob([GBDT_OUT / args.gbdt / "test_prob.parquet"], "UserID")
    rt = mean_prob([RDL_OUT / t / "test_prob.parquet" for t in args.rdl_test], "UserID")
    users = pd.read_csv(EVAL_DIR / "test_input.csv").UserID
    gt, rt = gt.loc[users], rt.loc[users]
    p = {a: best_w * gt[f"p_{a}"].to_numpy() + (1 - best_w) * rt[f"p_{a}"].to_numpy() for a in ACTS}
    pred = predict(p, thr)
    pred.insert(0, "UserID", users.to_numpy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"saved {out}  positive rate: {pred[ACTS].mean().round(3).to_dict()}")
    subprocess.run([sys.executable, str(EVALUATE), "--pred", str(out),
                    "--input", str(EVAL_DIR / "test_input.csv")], check=True)


if __name__ == "__main__":
    main()
