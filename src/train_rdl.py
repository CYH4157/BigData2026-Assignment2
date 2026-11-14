"""Relational Deep Learning: a temporal heterogeneous GNN on the relational entity graph.

Nodes are the rows of the 8 tables, edges are primary-foreign key links, and each node carries
its row time. A training example is (user, cutoff time, 6 labels); its computation graph is
sampled only from nodes with time <= cutoff (temporal neighbor sampling), and every node gets
an encoding of its age relative to the cutoff. Same weekly samples, validation weeks and
threshold tuning as train_gbdt.py, so the scores are comparable.

Run from the repository root:
    python src/train_rdl.py --tag sage
    python src/train_rdl.py --tag sage_fuse --fuse     # also feed the GBDT user features to the head
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

from data import ACTS, DB_DIR, EVAL_DIR, EVALUATE, MIN_COV, activity_log, load_db, score
from features import FeatureBuilder
from rdl_data import make_database
from train_gbdt import (GAP, TEST_START, YCOLS, build_dataset, cache_path, covered, feature_cols, persistence,
                        predict, save_prob, tune, week_scores, weekly_starts)

import relbench.modeling.graph as _rg  # noqa: E402

# relbench 1.1.0 divides the array from Series.values in place, which is read-only under pandas 3
# (copy-on-write); use a copying version.
_rg.to_unix_time = lambda ser: ser.astype("datetime64[ns]").astype("int64").to_numpy().copy() // 10**9

# loader workers pass many small tensors; the default fd-based sharing exhausts file descriptors
torch.multiprocessing.set_sharing_strategy("file_system")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rdl"
ENTITY = "Users"


def unix(ts):
    return (pd.to_datetime(pd.Series(ts)).astype("datetime64[ns]").astype("int64") // 10**9).to_numpy()


class Attach:
    """Attach labels, label masks (and optional per-sample features) to the seed nodes of a sampled batch."""

    def __init__(self, y, mask, x=None):
        self.y, self.mask, self.x = y, mask, x

    def __call__(self, batch):
        i = batch[ENTITY].input_id
        batch[ENTITY].y = self.y[i]
        batch[ENTITY].mask = self.mask[i]
        if self.x is not None:
            batch[ENTITY].x_extra = self.x[i]
        return batch


class Model(torch.nn.Module):
    def __init__(self, data, col_stats, channels, layers, n_extra, time_enc=True):
        super().__init__()
        self.time_enc = time_enc
        self.encoder = HeteroEncoder(
            channels, {nt: data[nt].tf.col_names_dict for nt in data.node_types}, col_stats)
        self.temporal = HeteroTemporalEncoder([nt for nt in data.node_types if "time" in data[nt]], channels)
        self.gnn = HeteroGraphSAGE(data.node_types, data.edge_types, channels, aggr="sum", num_layers=layers)
        self.extra = (torch.nn.Sequential(torch.nn.Linear(n_extra, channels), torch.nn.ReLU())
                      if n_extra else None)
        self.head = MLP(channels * (2 if n_extra else 1), hidden_channels=channels, out_channels=len(ACTS),
                        num_layers=2, norm="batch_norm", dropout=0.2)

    def forward(self, batch):
        seed_time = batch[ENTITY].seed_time
        x = self.encoder(batch.tf_dict)
        if self.time_enc:
            for nt, rt in self.temporal(seed_time, batch.time_dict, batch.batch_dict).items():
                x[nt] = x[nt] + rt
        x = self.gnn(x, batch.edge_index_dict)
        h = x[ENTITY][: seed_time.size(0)]
        if self.extra is not None:
            h = torch.cat([h, self.extra(batch[ENTITY].x_extra)], dim=1)
        return self.head(h)


def extra_matrix(X):
    """GBDT features for the neural head: NaN (e.g. never did an activity) -> -1, then signed log1p."""
    X = X.astype("float64").fillna(-1.0).to_numpy()
    return np.sign(X) * np.log1p(np.abs(X))


def make_loader(data, uid, tsec, y, mask, x, args, shuffle):
    return NeighborLoader(
        data, num_neighbors=[max(args.neighbors // 2**i, 4) for i in range(args.layers)], time_attr="time",
        input_nodes=(ENTITY, torch.as_tensor(uid)), input_time=torch.as_tensor(tsec),
        temporal_strategy=args.strategy,
        transform=Attach(torch.as_tensor(y), torch.as_tensor(mask), None if x is None else torch.as_tensor(x)),
        batch_size=args.batch, shuffle=shuffle, num_workers=args.workers, persistent_workers=args.workers > 0)


@torch.no_grad()
def infer(model, loader, n, device):
    model.eval()
    out = np.zeros((n, len(ACTS)), dtype=np.float32)
    for batch in loader:
        batch = batch.to(device)
        out[batch[ENTITY].input_id.cpu().numpy()] = torch.sigmoid(model(batch)).cpu().numpy()
    return out


def evaluate(va, prob, valid_starts, how):
    """Thresholds tuned on the validation labels with complete data; per-week Score over the activities
    with complete data in that week, and their mean (as in train_gbdt.py)."""
    p = {a: prob[:, i] for i, a in enumerate(ACTS)}
    truth = va[YCOLS].set_axis(ACTS, axis=1)
    thr = tune(p, truth, how, covered(va))
    sm = week_scores(lambda m: predict({a: p[a][m] for a in ACTS}, thr), va, valid_starts)
    sb = week_scores(lambda m: persistence(va[m]), va, valid_starts)
    weeks = {k: {"rdl": sm[k], "persist14": sb[k]} for k in sm}
    return thr, weeks, float(np.nanmean([w["rdl"]["score"] for w in weeks.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sage")
    ap.add_argument("--n-valid", type=int, default=4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--neighbors", type=int, default=64, help="neighbors per edge type in hop 1 (halved at each further hop)")
    ap.add_argument("--strategy", choices=["last", "uniform"], default="last",
                    help="temporal sampling: most recent neighbors first, or uniform over the past")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fuse", action="store_true", help="also use the GBDT user features in the head")
    ap.add_argument("--tune", choices=["f1", "score"], default="f1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-time-enc", action="store_true",
                    help="ablation: no relative-time encoding of the nodes (temporal sampling is kept)")
    ap.add_argument("--min-cov", type=float, default=0.0,
                    help="ignore training labels whose window has less data coverage than this; 0 keeps all")
    ap.add_argument("--final", action="store_true", help="train on all weeks (no validation) for the final model")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop after this epoch, keeping the learning-rate schedule of --epochs (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = load_db()
    log = activity_log(db)
    data_end = log.seen.max()
    starts = weekly_starts(data_end)
    df = build_dataset(db, log, starts, cache=cache_path(data_end))
    feats = feature_cols(df)

    t0 = time.time()
    rdb, col_to_stype = make_database(db)
    data, col_stats = make_pkey_fkey_graph(rdb, col_to_stype, cache_dir=str(OUT / f"tensor_frames_v2_{data_end:%Y%m%d}"))
    print(f"graph built in {time.time() - t0:.0f}s: "
          + ", ".join(f"{nt}={data[nt].num_nodes}" for nt in data.node_types), flush=True)

    u = db["Users"]
    gid2uid = pd.Series(u.user_id.to_numpy(), index=u.github_user_id.to_numpy())
    uid = df.github_user_id.map(gid2uid).to_numpy()
    tsec = unix(df.start - GAP)
    y = df[YCOLS].to_numpy(np.float32)
    mask = np.stack([(df[f"lcov_{a}"] >= args.min_cov).to_numpy() for a in ACTS], axis=1).astype(np.float32)

    valid_starts = starts[-args.n_valid:]
    is_va = df.start.isin(valid_starts).to_numpy()
    tr_i, va_i = np.where(~is_va)[0], np.where(is_va)[0]
    if args.final:  # train on every week; no validation
        tr_i, va_i = np.arange(len(df)), np.array([], dtype=int)
    va = df.iloc[va_i].reset_index(drop=True)

    x = mu = sd = None
    if args.fuse:
        x = extra_matrix(df[feats])
        mu, sd = x[tr_i].mean(0), x[tr_i].std(0) + 1e-6
        x = ((x - mu) / sd).astype(np.float32)
    sub = lambda i: (uid[i], tsec[i], y[i], mask[i], None if x is None else x[i])
    train_loader = make_loader(data, *sub(tr_i), args, shuffle=True)
    valid_loader = None if args.final else make_loader(data, *sub(va_i), args, shuffle=False)

    model = Model(data, col_stats, args.channels, args.layers, len(feats) if args.fuse else 0,
                  time_enc=not args.no_time_enc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    stop = args.stop_at or args.epochs
    print(f"{'FINAL: ' if args.final else ''}train samples={len(tr_i)} valid samples={len(va_i)} "
          f"params={sum(p.numel() for p in model.parameters()):,} epochs={stop}/{args.epochs}")

    best = (-1.0, None)
    for ep in range(1, stop + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            m = batch[ENTITY].mask
            loss = (loss_fn(model(batch), batch[ENTITY].y) * m).sum() / m.sum().clamp(min=1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} step {n // args.batch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(batch[ENTITY].y), n + len(batch[ENTITY].y)
        sched.step()
        if args.final:
            print(f"epoch {ep:2d} loss={tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
            continue
        prob = infer(model, valid_loader, len(va_i), device)
        thr, weeks, mean = evaluate(va, prob, valid_starts, args.tune)
        per_week = " ".join(f"{w['rdl']['score']:.4f}" for w in weeks.values())
        print(f"epoch {ep:2d} loss={tot / n:.4f} valid mean score={mean:.4f} [{per_week}] "
              f"({time.time() - t0:.0f}s)", flush=True)
        if mean > best[0]:
            best = (mean, ep)
            torch.save(model.state_dict(), out / "model.pt")
            save_prob(va[["github_user_id", "start"]], {a: prob[:, i] for i, a in enumerate(ACTS)},
                      out / "valid_prob.parquet")
            report = {"args": vars(args), "data_end": str(data_end), "epoch": ep, "threshold": thr,
                      "mean_valid_score": mean, "weeks": weeks,
                      "mean_valid_persist14": float(np.mean([w["persist14"]["score"] for w in weeks.values()]))}
            (out / "valid_report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.final:
        torch.save(model.state_dict(), out / "model.pt")
        (out / "final_args.json").write_text(json.dumps({"args": vars(args), "data_end": str(data_end)}, indent=2))
    else:
        print(f"best epoch {best[1]}: mean valid score={best[0]:.4f}")
        model.load_state_dict(torch.load(out / "model.pt"))

    # test-user probabilities (the ensemble script turns them into 0/1 with validation thresholds)
    tf = TEST_START - GAP
    if data_end < tf - pd.Timedelta(hours=1):
        tf = data_end.ceil("h")
        print(f"WARNING: not the final release; test features taken at {tf} (dry run)")
    users = pd.read_csv(EVAL_DIR / "test_input.csv").UserID
    xt = None
    if args.fuse:
        xt = ((extra_matrix(FeatureBuilder(db, log).build(tf, users)[feats]) - mu) / sd).astype(np.float32)
    test_loader = make_loader(data, users.map(gid2uid).to_numpy(), np.full(len(users), unix([tf])[0]),
                              np.zeros((len(users), len(ACTS)), np.float32),
                              np.ones((len(users), len(ACTS)), np.float32), xt, args, shuffle=False)
    prob = infer(model, test_loader, len(users), device)
    save_prob(pd.DataFrame({"UserID": users.to_numpy()}), {a: prob[:, i] for i, a in enumerate(ACTS)},
              out / "test_prob.parquet")
    print(f"saved {out / 'test_prob.parquet'}  mean prob: {prob.mean(0).round(3).tolist()}")


if __name__ == "__main__":
    main()
