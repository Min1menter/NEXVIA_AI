"""Combine XGBoost + graph (GNN) scores.

Two combiners are fitted on the VALIDATION period and compared there:
  * weighted : alpha * xgb + (1 - alpha) * gnn            (the old blend)
  * stack    : logistic regression on [logit(xgb), logit(gnn), has_graph, xgb*gnn, log cluster size]
The stacker can learn "a risky transaction from a ring-linked account", which a weighted average cannot.
Selection uses out-of-fold validation scores (grouped by user) so nothing is judged on data it was fitted on.
The test period is only used for the final report. Coefficients live in meta.json, so serving needs no sklearn.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from fraudshield.config import ART_DIR, DATA_DIR, load_params, mlflow_setup
from fraudshield.features import FEATURE_COLUMNS
from fraudshield.train_xgb import evaluate

STACK_COLS = ["xgb_logit", "graph_logit", "has_graph", "xgb_x_graph", "log_cluster"]


def blend_scores(xgb, gnn, alpha: float) -> np.ndarray:
    """Weighted average. Users without a graph score (NaN) fall back to the XGBoost score."""
    xgb, gnn = np.asarray(xgb, float), np.asarray(gnn, float)
    return np.where(np.isnan(gnn), xgb, alpha * xgb + (1 - alpha) * gnn)


def tune_alpha(y, xgb, gnn, step=0.05) -> tuple[float, float]:
    best = (1.0, -1.0)
    for a in np.arange(1.0, -1e-9, -step):        # from 1.0 down; strict '>' => ties keep the XGBoost-only weight (e.g. no graph scores)
        ap = average_precision_score(y, blend_scores(xgb, gnn, a))
        if ap > best[1] + 1e-12:
            best = (float(round(a, 4)), float(ap))
    return best


def pick_threshold(y, score, amount, review_cost: float) -> dict:
    """Threshold maximising expected savings = fraud amount caught - review_cost * alerts."""
    y, score, amount = map(np.asarray, (y, score, amount))
    cands = np.unique(np.quantile(score, np.linspace(0.90, 0.9995, 200)))
    best = None
    for t in cands:
        flag = score >= t
        tp = flag & (y == 1)
        savings = amount[tp].sum() - review_cost * flag.sum()
        if best is None or savings > best["savings"]:
            best = dict(threshold=float(t), savings=float(savings), alerts=int(flag.sum()),
                        precision=float(tp.sum() / max(1, flag.sum())), recall=float(tp.sum() / max(1, (y == 1).sum())))
    return best


# ---------------------------------------------------------------- stacking
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(xgb, graph, cluster) -> np.ndarray:
    """The ONE place stacker inputs are built (training and serving). graph/cluster may contain NaN (= unknown user)."""
    xgb, graph, cluster = (np.atleast_1d(np.asarray(a, float)) for a in (xgb, graph, cluster))
    has = ~np.isnan(graph)
    g = np.where(has, graph, 0.0)
    return np.column_stack([_logit(xgb), np.where(has, _logit(g), 0.0), has.astype(float), _logit(xgb) * g,
                            np.log1p(np.nan_to_num(cluster, nan=0.0))])


class Stacker:
    def __init__(self, coef, intercept, mean, std):
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)

    @classmethod
    def fit(cls, X, y, C=1.0):
        mean, std = X.mean(0), X.std(0)
        std = np.where(std < 1e-9, 1.0, std)
        lr = LogisticRegression(C=C, max_iter=2000).fit((X - mean) / std, y)
        return cls(lr.coef_[0], lr.intercept_[0], mean, std)

    def predict(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(((X - self.mean) / self.std) @ self.coef + self.intercept)))

    def to_dict(self):
        return dict(coef=self.coef.tolist(), intercept=self.intercept, mean=self.mean.tolist(), std=self.std.tolist(), columns=STACK_COLS)

    @classmethod
    def from_dict(cls, d):
        return cls(d["coef"], d["intercept"], d["mean"], d["std"])


def combine(meta: dict, xgb, graph, cluster) -> np.ndarray:
    """Final score for arrays of transactions, using whichever combiner meta.json selected."""
    if meta.get("blend_method") == "stack":
        return Stacker.from_dict(meta["stacker"]).predict(stack_features(xgb, graph, cluster))
    return blend_scores(xgb, graph, meta["alpha"])


def fit_stack(y, groups, X, Cs, folds=5):
    """Pick C by grouped out-of-fold PR-AUC; return (final stacker fitted on all rows, OOF scores, best C, OOF AP)."""
    best = None
    k = max(2, min(folds, len(np.unique(groups))))
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(k).split(X, y, groups):
            oof[te] = Stacker.fit(X[tr], y[tr], C).predict(X[te]) if y[tr].sum() > 0 else 0.0
        ap = average_precision_score(y, oof)
        if best is None or ap > best[2]:
            best = (C, oof, ap)
    return Stacker.fit(X, y, best[0]), best[1], best[0], float(best[2])


def paired_bootstrap_ap(y, a, b, groups, n, rng) -> dict:
    """Cluster bootstrap over users: distribution of AP(b) - AP(a)."""
    codes = pd.factorize(groups)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    by_user = [order[bounds[i]:bounds[i + 1]] for i in range(codes.max() + 1)]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([by_user[i] for i in rng.integers(0, len(by_user), len(by_user))])
        if y[idx].sum() > 0:
            diffs.append(average_precision_score(y[idx], b[idx]) - average_precision_score(y[idx], a[idx]))
    d = np.array(diffs)
    return dict(mean_gain=float(d.mean()), ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)),
                prob_better=float((d > 0).mean()), n_boot=len(d))


def _business(te, score, thr, review_cost):
    flag = (score >= thr["threshold"])
    fr = te.is_fraud.values == 1
    ring, ato = fr & (te.ring_id.values >= 0), fr & (te.ring_id.values < 0)
    amount = te.amount.values
    return dict(threshold=thr["threshold"], precision=float((flag & fr).sum() / max(1, flag.sum())),
                recall=float((flag & fr).sum() / max(1, fr.sum())),
                ring_fraud_recall=float(flag[ring].mean()) if ring.any() else None,
                other_fraud_recall=float(flag[ato].mean()) if ato.any() else None,
                alerts_per_1k_txn=float(1000 * flag.mean()),
                net_savings=float(amount[flag & fr].sum() - review_cost * flag.sum()))


def run_blend(df: pd.DataFrame, p: dict) -> tuple[dict, dict]:
    """df needs: split, is_fraud, user_id, amount, ring_id, xgb_score, graph_score, cluster_size."""
    cfg = p["blend"]
    rng = np.random.default_rng(p["seed"])
    va, te = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    yv, yt = va.is_fraud.values, te.is_fraud.values
    alpha, w_ap = tune_alpha(yv, va.xgb_score, va.graph_score, cfg["alpha_step"])
    Xv = stack_features(va.xgb_score, va.graph_score, va.cluster_size)
    Xt = stack_features(te.xgb_score, te.graph_score, te.cluster_size)
    stacker, oof, C, oof_ap = fit_stack(yv, va.user_id.values, Xv, cfg["stack"]["Cs"], cfg["stack"]["folds"])
    xgb_ap_val = float(average_precision_score(yv, va.xgb_score))
    method = "stack" if oof_ap >= max(w_ap, xgb_ap_val) + cfg["stack"]["min_gain"] else "weighted"

    val_scores = {"xgb": va.xgb_score.values, "weighted": blend_scores(va.xgb_score, va.graph_score, alpha), "stack": oof}
    test_scores = {"xgb": te.xgb_score.values, "weighted": blend_scores(te.xgb_score, te.graph_score, alpha),
                   "stack": stacker.predict(Xt)}
    res = {"selection_on_validation": dict(xgb_ap=xgb_ap_val, weighted_ap=w_ap, stack_oof_ap=oof_ap, stack_C=C,
                                           min_gain=cfg["stack"]["min_gain"], chosen=method, alpha_xgb_weight=alpha),
           "stacker_coefficients": dict(zip(STACK_COLS, stacker.coef.round(3).tolist())), "test": {}}
    for name in ("xgb", "weighted", "stack"):
        thr = pick_threshold(yv, val_scores[name], va.amount, cfg["review_cost"])   # threshold fitted on validation only
        res["test"][name] = {**evaluate(yt, test_scores[name]), "at_threshold": _business(te, test_scores[name], thr, cfg["review_cost"])}
    chosen_thr = pick_threshold(yv, val_scores[method], va.amount, cfg["review_cost"])
    res["chosen_method"] = method
    res["test"]["chosen"] = {**evaluate(yt, test_scores[method]), "at_threshold": _business(te, test_scores[method], chosen_thr, cfg["review_cost"])}
    nb = int(cfg.get("bootstrap", 200))
    res["test"]["bootstrap_ap_gain_vs_xgb"] = {m: paired_bootstrap_ap(yt, test_scores["xgb"], test_scores[m], te.user_id.values, nb, rng)
                                               for m in ("weighted", "stack")}
    gt = te[te.graph_score.notna()]
    res["test"]["graph_coverage"] = float(te.graph_score.notna().mean())
    res["test"]["gnn_only_on_covered_txns"] = evaluate(gt.is_fraud, gt.graph_score) if gt.is_fraud.nunique() > 1 else None
    meta = dict(feature_columns=FEATURE_COLUMNS, alpha=alpha, blend_method=method, stacker=stacker.to_dict(),
                threshold=chosen_thr["threshold"], model_version=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                metrics=res, train_prevalence=float(df[df.split == "train"].is_fraud.mean()))
    return res, meta


def main():
    p = load_params()
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = df.merge(pd.read_parquet(DATA_DIR / "preds_xgb.parquet")[["txn_id", "xgb_score"]], on="txn_id")
    df = df.merge(pd.read_parquet(DATA_DIR / "graph_scores.parquet")[["user_id", "graph_score", "cluster_size"]], on="user_id", how="left")
    res, meta = run_blend(df, p)
    ART_DIR.mkdir(exist_ok=True)
    json.dump(meta, open(ART_DIR / "meta.json", "w"), indent=2)
    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="blend"):
        mlflow.log_params({"method": meta["blend_method"], "alpha": meta["alpha"], "threshold": meta["threshold"]})
        t = res["test"]
        mlflow.log_metrics({"test_pr_auc_xgb": t["xgb"]["pr_auc"], "test_pr_auc_weighted": t["weighted"]["pr_auc"],
                            "test_pr_auc_stack": t["stack"]["pr_auc"], "test_pr_auc_blend": t["chosen"]["pr_auc"],
                            "test_precision": t["chosen"]["at_threshold"]["precision"], "test_recall": t["chosen"]["at_threshold"]["recall"]})
        mlflow.log_artifact(str(ART_DIR / "meta.json"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
