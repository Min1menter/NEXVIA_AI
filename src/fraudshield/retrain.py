"""Retraining pipeline: export analyst feedback -> retrain XGBoost + GNN -> re-blend -> refresh Feast -> hot-reload API."""
from __future__ import annotations
import os
import requests
from fraudshield import blend, db as dbm, graph, train_xgb
from fraudshield.config import DATA_DIR, mlflow_setup


def retrain(reasons: list[str]) -> dict:
    sf = dbm.make_session_factory()
    with sf() as db:
        fb = dbm.export_feedback(db)
    if len(fb):
        fb.to_parquet(DATA_DIR / "feedback_labeled.parquet", index=False)
    train_xgb.main(); graph.main(); blend.main()
    if os.getenv("FEATURE_BACKEND") == "feast":
        from fraudshield.store import materialize
        materialize()
    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="retrain-trigger"):
        mlflow.set_tags({"trigger": "; ".join(reasons), "feedback_rows": str(len(fb))})
    result = {"feedback_rows": len(fb), "reload": None}
    url = os.getenv("API_URL")
    if url:
        try:
            result["reload"] = requests.post(f"{url}/admin/reload", headers={"X-API-Key": os.getenv("API_KEY", "")}, timeout=30).json()
        except Exception as e:  # noqa
            result["reload"] = f"failed: {e}"
    return result


if __name__ == "__main__":
    print(retrain(["manual"]))
