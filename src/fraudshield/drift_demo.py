"""Live demo of drift detection + automatic retraining trigger. Uses its OWN throw-away SQLite DB (drift_demo.db),
so your real alerts/scores are untouched.

  python -m fraudshield.drift_demo                    # inject drifted traffic, show the drift report + trigger decision
  python -m fraudshield.drift_demo --precision-drop   # instead simulate analysts rejecting most alerts (2nd trigger)
  python -m fraudshield.drift_demo --retrain          # also run the retraining pipeline (OVERWRITES artifacts/ - back up first)
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np
import pandas as pd


def make_drifted(ref: pd.DataFrame, rng: np.random.Generator, n: int = 5000) -> pd.DataFrame:
    """Resample reference rows and shift behaviour the way a real attack wave / new market would."""
    cur = ref.sample(n, replace=True, random_state=int(rng.integers(1 << 30))).reset_index(drop=True).copy()
    cur["log_amount"] += 1.5                                   # much larger amounts
    cur["amount_vs_avg"] *= 4
    cur["velocity_short"] = cur["velocity_short"] * 5 + 1      # bursts of activity
    cur["velocity_long"] = cur["velocity_long"] * 3 + 1
    cur["device_changed"] = (rng.random(n) < 0.45).astype(float)
    cur["geo_changed"] = (rng.random(n) < 0.35).astype(float)
    cur["device_geo_changed"] = cur["device_changed"] * cur["geo_changed"]
    cur["hour"] = rng.integers(0, 6, n).astype(float)          # night-time spike
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision-drop", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    os.environ["DATABASE_URL"] = "sqlite:///./drift_demo.db"         # must be set before importing fraudshield.db users
    from fraudshield import db as dbm
    from fraudshield.config import ART_DIR, DATA_DIR
    from fraudshield.monitoring import run_check

    if os.path.exists("drift_demo.db"):
        os.remove("drift_demo.db")
    rng = np.random.default_rng(7)
    ref = pd.read_parquet(DATA_DIR / "reference_features.parquet")
    sf = dbm.make_session_factory()
    feats = ref.sample(5000, replace=True, random_state=1) if a.precision_drop else make_drifted(ref, rng)
    with sf() as db:
        for i, row in enumerate(feats.to_dict("records")):
            db.add(dbm.Scored(txn_id=f"DEMO-{i}", user_id=f"DEMO-U{i % 300}", ts=1.8e9 + i, amount=100.0, score=0.9 if a.precision_drop and i < 60 else 0.01,
                              decision="alert" if a.precision_drop and i < 60 else "allow",
                              features={k: (None if v != v else float(v)) for k, v in row.items()}, model_version="demo"))
        if a.precision_drop:                                         # 60 alerts, analysts mark 50 of them false positives
            db.flush()
            for i in range(60):
                db.add(dbm.Alert(txn_id=f"DEMO-{i}", user_id=f"DEMO-U{i % 300}", ts=1.8e9 + i, amount=100.0, score=0.9, xgb_score=0.9, reasons=[],
                                 status="false_positive" if i < 50 else "confirmed_fraud", feedback_label=0 if i < 50 else 1))
        db.commit()
    if a.retrain:
        shutil.copytree(ART_DIR, str(ART_DIR) + "_backup", dirs_exist_ok=True)
        print(f"backed up artifacts to {ART_DIR}_backup")
    rep = run_check(retrain_if_needed=a.retrain)
    top = sorted(((k, v["psi"]) for k, v in rep.get("features", {}).items()), key=lambda x: -x[1])[:6]
    print(json.dumps({k: v for k, v in rep.items() if k != "features"}, indent=2, default=str))
    print("most drifted features (PSI):", [(k, round(p, 2)) for k, p in top])


if __name__ == "__main__":
    main()
