"""Ablation: how much of the stacker's gain comes from the GNN score vs. the device/card cluster size?
Run:  python -m fraudshield.ablation      (a subprocess avoids notebook import-path problems)"""
from __future__ import annotations
import json
import pandas as pd
from sklearn.metrics import average_precision_score as AP
import numpy as np
from fraudshield.blend import _business, fit_stack, paired_bootstrap_ap, pick_threshold, stack_features
from fraudshield.config import ART_DIR, DATA_DIR, load_params

# stack_features columns: 0 xgb_logit, 1 graph_logit, 2 has_graph, 3 xgb*graph, 4 log_cluster
VARIANTS = {"xgb only": [0], "xgb + cluster size": [0, 4], "xgb + GNN score": [0, 1, 2, 3],
            "xgb + GNN + cluster (no interaction)": [0, 1, 2, 4], "full stack": [0, 1, 2, 3, 4]}


def run_ablation(df: pd.DataFrame, p: dict) -> dict:
    cfg = p["blend"]
    va, te = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    Xv = stack_features(va.xgb_score, va.graph_score, va.cluster_size)
    Xt = stack_features(te.xgb_score, te.graph_score, te.cluster_size)
    out, scores = {}, {}
    for name, cols in VARIANTS.items():
        st, oof, C, oof_ap = fit_stack(va.is_fraud.values, va.user_id.values, Xv[:, cols], cfg["stack"]["Cs"], cfg["stack"]["folds"])
        s_te = scores[name] = st.predict(Xt[:, cols])
        thr = pick_threshold(va.is_fraud.values, oof, va.amount.values, cfg["review_cost"])     # threshold from validation only
        out[name] = dict(val_oof_ap=oof_ap, test_ap=float(AP(te.is_fraud, s_te)), **{k: v for k, v in _business(te, s_te, thr, cfg["review_cost"]).items()
                         if k in ("precision", "recall", "ring_fraud_recall", "net_savings")})
    rng = np.random.default_rng(p["seed"])
    nb = int(cfg.get("bootstrap", 200))
    y, g = te.is_fraud.values, te.user_id.values
    out["_bootstrap_ap_gain"] = {                      # paired cluster bootstrap over users on the TEST period
        "cluster_vs_xgb": paired_bootstrap_ap(y, scores["xgb only"], scores["xgb + cluster size"], g, nb, rng),
        "full_vs_cluster": paired_bootstrap_ap(y, scores["xgb + cluster size"], scores["full stack"], g, nb, rng),
        "gnn_vs_cluster": paired_bootstrap_ap(y, scores["xgb + cluster size"], scores["xgb + GNN score"], g, nb, rng)}
    return out


def main():
    p = load_params()
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = df.merge(pd.read_parquet(DATA_DIR / "preds_xgb.parquet")[["txn_id", "xgb_score"]], on="txn_id")
    df = df.merge(pd.read_parquet(DATA_DIR / "graph_scores.parquet")[["user_id", "graph_score", "cluster_size"]], on="user_id", how="left")
    res = run_ablation(df, p)
    json.dump(res, open(ART_DIR / "ablation.json", "w"), indent=2)
    boot = res.pop("_bootstrap_ap_gain")
    print(pd.DataFrame(res).T.round(3).to_string())
    for k, v in boot.items():
        print(f"PR-AUC gain {k:16s}: {v['mean_gain']:+.3f}  95% CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  P(better)={v['prob_better']:.2f}")
    res["_bootstrap_ap_gain"] = boot
    full, clus, gnn = res["full stack"]["test_ap"], res["xgb + cluster size"]["test_ap"], res["xgb + GNN score"]["test_ap"]
    print(f"\nTest PR-AUC: xgb only {res['xgb only']['test_ap']:.3f} | +cluster {clus:.3f} | +GNN {gnn:.3f} | full {full:.3f}")
    print("GNN adds on top of cluster size:", f"{full - clus:+.3f} PR-AUC")


if __name__ == "__main__":
    main()
