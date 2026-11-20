"""Kafka-free replay: pushes the test period through the real Scorer + DB and reports ONLINE metrics.
Also checks online-vs-offline feature parity against data/features.parquet."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from fraudshield import db as dbm
from fraudshield.config import ART_DIR, DATA_DIR
from fraudshield.features import FEATURE_COLUMNS
from fraudshield.scoring import build_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sc, sf = build_scorer(), dbm.make_session_factory()
    rp = pd.read_parquet(DATA_DIR / "replay.parquet").sort_values("timestamp")
    if a.limit:
        rp = rp.head(a.limit)
    off = pd.read_parquet(DATA_DIR / "features.parquet").set_index("txn_id")
    out, max_diff = [], 0.0
    with sf() as db:
        for r in rp.to_dict("records"):
            res = sc.score({**r, "timestamp": r["timestamp"].timestamp()})
            dbm.save_result(db, res)
            out.append((res["score"], res["decision"], r["is_fraud"], res["latency_ms"]))
            # drift between online features and the offline-computed ones for the same txn.
            # state at cutoff is identical by construction, so differences should be ~0 for users replayed in order
            f_off = off.loc[r["txn_id"], FEATURE_COLUMNS].astype(float).to_numpy()
            f_on = np.array([res["features"][c] for c in FEATURE_COLUMNS], float)
            ok = ~(np.isnan(f_off) | np.isnan(f_on))
            max_diff = max(max_diff, float(np.abs(f_off[ok] - f_on[ok]).max()) if ok.any() else 0.0)
    s = np.array([o[0] for o in out]); y = np.array([o[2] for o in out]); flag = np.array([o[1] == "alert" for o in out])
    rep = dict(n=len(y), fraud_rate=float(y.mean()), online_pr_auc=float(average_precision_score(y, s)),
               alert_precision=float((flag & (y == 1)).sum() / max(1, flag.sum())),
               fraud_recall=float((flag & (y == 1)).sum() / max(1, y.sum())), alerts=int(flag.sum()),
               p50_latency_ms=float(np.percentile([o[3] for o in out], 50)), p99_latency_ms=float(np.percentile([o[3] for o in out], 99)),
               max_abs_feature_diff_online_vs_offline=max_diff)
    backend = os.getenv("FEATURE_BACKEND", "memory")      # separate files so a Feast run never overwrites the in-memory run
    json.dump(rep, open(ART_DIR / f"metrics_online_replay_{backend}.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
