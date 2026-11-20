"""XGBoost fraud model. Imbalance handled by class weights or SMOTE; evaluated with PR-AUC; tracked in MLflow."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from fraudshield.config import ART_DIR, DATA_DIR, load_params, mlflow_setup
from fraudshield.features import FEATURE_COLUMNS


def precision_at_recall(y, s, recall):
    p, r, _ = precision_recall_curve(y, s)
    ok = p[r >= recall]
    return float(ok.max()) if len(ok) else 0.0


def evaluate(y, s) -> dict:
    return dict(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)),
                prevalence=float(np.mean(y)), precision_at_recall_80=precision_at_recall(y, s, .8),
                precision_at_recall_50=precision_at_recall(y, s, .5))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    p = load_params()
    c = p["xgb"]
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    fb_path = DATA_DIR / "feedback_labeled.parquet"
    if fb_path.exists():                       # analyst feedback from the dashboard (retraining loop)
        fb = pd.read_parquet(fb_path)
        tr = pd.concat([tr, fb[FEATURE_COLUMNS + ["is_fraud"]]], ignore_index=True)
        print(f"added {len(fb)} analyst-labeled rows")
    Xtr, ytr = tr[FEATURE_COLUMNS], tr.is_fraud.values
    spw = 1.0
    if c["imbalance"] == "smote":
        from imblearn.over_sampling import SMOTE
        Xtr, ytr = SMOTE(sampling_strategy=0.1, random_state=p["seed"]).fit_resample(Xtr.fillna(-1), ytr)
    else:
        spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    clf = xgb.XGBClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"], learning_rate=c["learning_rate"],
                            subsample=c["subsample"], colsample_bytree=c["colsample_bytree"], scale_pos_weight=spw,
                            tree_method="hist", eval_metric="aucpr", early_stopping_rounds=c["early_stopping_rounds"],
                            random_state=p["seed"], n_jobs=-1)
    clf.fit(Xtr, ytr, eval_set=[(va[FEATURE_COLUMNS], va.is_fraud.values)], verbose=False)

    preds = pd.concat([d[["txn_id", "split"]].assign(xgb_score=clf.predict_proba(d[FEATURE_COLUMNS])[:, 1]) for d in (tr[tr.split.notna()], va, te)])
    m = {f"{n}_{k}": v for n, d in (("val", va), ("test", te))
         for k, v in evaluate(d.is_fraud.values, clf.predict_proba(d[FEATURE_COLUMNS])[:, 1]).items()}
    ART_DIR.mkdir(exist_ok=True)
    clf.get_booster().save_model(str(ART_DIR / "xgb.json"))
    preds.to_parquet(DATA_DIR / "preds_xgb.parquet", index=False)
    json.dump(m, open(ART_DIR / "metrics_xgb.json", "w"), indent=2)

    mlflow = mlflow_setup()
    with mlflow.start_run(run_name=f"xgb-{c['imbalance']}") as run:
        mlflow.log_params({**{f"xgb_{k}": v for k, v in c.items()}, "scale_pos_weight": spw, "n_features": len(FEATURE_COLUMNS)})
        mlflow.log_metrics(m)
        pr, rc, _ = precision_recall_curve(te.is_fraud.values, clf.predict_proba(te[FEATURE_COLUMNS])[:, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rc, pr)
        ax.axhline(m["test_prevalence"], ls="--", c="grey")
        ax.set(xlabel="recall", ylabel="precision", title=f"Test PR curve (AP={m['test_pr_auc']:.3f})")
        mlflow.log_figure(fig, "pr_curve_test.png")
        imp = pd.Series(clf.feature_importances_, FEATURE_COLUMNS).sort_values()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        imp.plot.barh(ax=ax2)
        mlflow.log_figure(fig2, "feature_importance.png")
        mlflow.log_artifact(str(ART_DIR / "xgb.json"))
        json.dump({"xgb_run_id": run.info.run_id}, open(ART_DIR / "run_ids.json", "w"))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
