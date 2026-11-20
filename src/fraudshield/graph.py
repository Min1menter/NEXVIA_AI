"""Account-device-card-counterparty graph + GraphSAGE node classifier (PyTorch Geometric).
Graph is built from the TRAIN period only (no label leakage into val/test blend tuning)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score
from fraudshield.config import ART_DIR, DATA_DIR, load_params, mlflow_setup

KINDS = [("device", "device_id"), ("card", "card_id"), ("cp", "merchant_id")]


NULLS = {"", "unknown", "nan", "none", "-1", "na"}


def build_graph(df: pd.DataFrame, hub_limit: int = 200) -> dict:
    """Nodes: users, devices, cards, counterparties. Missing keys and 'hub' keys shared by > hub_limit users
    (e.g. a generic e-mail domain) are dropped: they connect everything and carry no ring signal."""
    users, user_idx = pd.factorize(df["user_id"])
    n_users = len(user_idx)
    offset, node_type = n_users, [np.zeros(n_users, int)]
    empty = np.empty(0, dtype=np.int64)
    src, dst, cluster_src, cluster_dst = [empty], [empty], [empty], [empty]
    for t, (kind, col) in enumerate(KINDS, start=1):
        e = pd.DataFrame({"u": users, "k": df[col].astype(str).to_numpy()})
        e = e[~e["k"].str.strip().str.lower().isin(NULLS)].drop_duplicates()
        e = e[e.groupby("k")["u"].transform("size") <= hub_limit]
        if e.empty:
            continue
        k_codes, k_idx = pd.factorize(e["k"])
        u, k = e["u"].to_numpy(), k_codes + offset
        src += [u, k]; dst += [k, u]
        if kind in ("device", "card"):
            cluster_src.append(u); cluster_dst.append(k)
        node_type.append(np.full(len(k_idx), t))
        offset += len(k_idx)
    node_type = np.concatenate(node_type)
    ei = np.vstack([np.concatenate(src), np.concatenate(dst)])
    deg = np.bincount(ei[0], minlength=offset)
    g = pd.DataFrame({"u": users, "la": np.log1p(df["amount"].to_numpy(float)),
                      "tr": df["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy(float)})
    agg = g.groupby("u").agg(n=("la", "size"), la=("la", "mean"), tr=("tr", "mean"))
    x = np.zeros((offset, 8), np.float32)
    x[np.arange(offset), node_type] = 1.0
    x[:, 4] = np.log1p(deg)
    x[agg.index.to_numpy(), 5] = np.log1p(agg["n"].to_numpy())
    x[agg.index.to_numpy(), 6] = agg["la"].to_numpy()
    x[agg.index.to_numpy(), 7] = agg["tr"].to_numpy()
    y = np.zeros(n_users, np.float32)
    y[np.unique(users[df["is_fraud"].to_numpy() == 1])] = 1.0
    # cluster = users connected through shared devices/cards ("ring" candidates)
    cs, cd = np.concatenate(cluster_src), np.concatenate(cluster_dst)
    adj = coo_matrix((np.ones(len(cs)), (cs, cd)), shape=(offset, offset))
    _, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp[:n_users])[comp[:n_users]]
    return dict(user_ids=np.asarray(user_idx), x=x, edge_index=ei, y=y, n_users=n_users, cluster_size=sizes, cluster_id=comp[:n_users])


def make_model(in_dim, cfg):
    import torch
    from torch_geometric.nn import SAGEConv

    class SAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dims = [in_dim] + [cfg["hidden"]] * cfg["layers"]
            self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
            self.out = torch.nn.Linear(cfg["hidden"], 1)
            self.drop = cfg["dropout"]

        def forward(self, x, ei):
            for c in self.convs:
                x = torch.relu(c(x, ei))
                x = torch.nn.functional.dropout(x, self.drop, self.training)
            return self.out(x).squeeze(-1)
    return SAGE()


def train_gnn(g: dict, cfg: dict, seed: int):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = g["n_users"]
    perm = rng.permutation(n)
    tr, va, te = perm[: int(.7 * n)], perm[int(.7 * n): int(.85 * n)], perm[int(.85 * n):]
    x, ei, y = torch.tensor(g["x"]), torch.tensor(g["edge_index"]), torch.tensor(g["y"])
    model = make_model(x.shape[1], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=5e-4)
    pos = float(y[tr].sum().clamp(min=1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(50.0, (len(tr) - pos) / pos)))
    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        loss = lossf(model(x, ei)[tr], y[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(x, ei))[:n].numpy()
        ap = average_precision_score(g["y"][va], s[va]) if g["y"][va].sum() else 0.0
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, ei))[:n].numpy()
    m = {"val_ap": float(best), "test_ap": float(average_precision_score(g["y"][te], scores[te])) if g["y"][te].sum() else None,
         "test_auc": float(roc_auc_score(g["y"][te], scores[te])) if 0 < g["y"][te].sum() < len(te) else None,
         "n_users": int(n), "n_fraud_users": int(g["y"].sum())}
    return model, scores, m


def main():
    import torch
    p = load_params()
    f = pd.read_parquet(DATA_DIR / "features.parquet", columns=["txn_id", "split"])
    raw = pd.read_parquet(DATA_DIR / "raw" / "transactions.parquet")
    raw = raw.merge(f, on="txn_id")
    hist = raw[raw.split == "train"]
    g = build_graph(hist, p["gnn"].get("hub_limit", 200))
    model, scores, m = train_gnn(g, p["gnn"], p["seed"])
    cutoff = pd.Timestamp(pd.read_json(DATA_DIR / "cutoff.json", typ="series")["cutoff"])
    pd.DataFrame({"user_id": g["user_ids"], "event_timestamp": cutoff, "graph_score": scores.astype(np.float32),
                  "cluster_size": g["cluster_size"].astype(np.int64)}).to_parquet(DATA_DIR / "graph_scores.parquet", index=False)
    pd.DataFrame({"user_id": g["user_ids"], "cluster_id": g["cluster_id"].astype(np.int64),
                  "cluster_size": g["cluster_size"].astype(np.int64)}).to_parquet(DATA_DIR / "rings.parquet", index=False)   # for the analyst "linked accounts" view
    ART_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": p["gnn"]}, ART_DIR / "gnn.pt")
    json.dump(m, open(ART_DIR / "metrics_gnn.json", "w"), indent=2)
    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="gnn-graphsage"):
        mlflow.log_params({f"gnn_{k}": v for k, v in p["gnn"].items()})
        mlflow.log_metrics({k: v for k, v in m.items() if v is not None})
    print(m)


if __name__ == "__main__":
    main()
