"""Proves training/serving parity: features recomputed from Feast point-in-time state must equal the offline training features."""
import json
import numpy as np
import pandas as pd
from feast import FeatureStore
from fraudshield.config import ART_DIR, DATA_DIR, FEATURE_REPO, load_params
from fraudshield.features import FEATURE_COLUMNS, STATE_FIELDS, txn_features

p = load_params()
fs = FeatureStore(repo_path=FEATURE_REPO)
raw = pd.read_parquet(DATA_DIR / "raw" / "transactions.parquet")
off = pd.read_parquet(DATA_DIR / "features.parquet")
sample = off[off.split.isin(["train", "val"])].sample(2000, random_state=1)
ent = sample[["txn_id", "user_id", "timestamp"]].rename(columns={"timestamp": "event_timestamp"})
hist = fs.get_historical_features(entity_df=ent, features=[f"user_state:{c}" for c in STATE_FIELDS]).to_df()
m = hist.merge(raw, on="txn_id", suffixes=("", "_raw")).set_index("txn_id")
diffs = []
for tid, r in m.iterrows():
    st = {c: r[c] for c in STATE_FIELDS}; st["n_txn"] = int(st["n_txn"])
    f = txn_features(st, dict(ts=r["timestamp"].timestamp(), amount=r["amount"], device_id=r["device_id"], country=r["country"],
                              type=r["type"], balance_before=r["balance_before"]), p["features"]["tau_short_s"], p["features"]["tau_long_s"])
    a = np.array([f[c] for c in FEATURE_COLUMNS], float); b = off.set_index("txn_id").loc[tid, FEATURE_COLUMNS].astype(float).to_numpy()
    ok = ~(np.isnan(a) | np.isnan(b)); diffs.append(np.abs(a[ok] - b[ok]).max() if ok.any() else 0.0)
res = dict(n=len(diffs), max_abs_diff=float(max(diffs)), identical=bool(max(diffs) < 1e-6))
json.dump(res, open(ART_DIR / "feast_parity.json", "w"), indent=2)
print(res)
assert res["identical"], "Feast point-in-time features differ from offline features!"
