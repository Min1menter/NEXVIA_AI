import math
import numpy as np
import pandas as pd
from fraudshield.features import (FEATURE_COLUMNS, build_offline, empty_state, row_to_txn, txn_features, update_state)
from fraudshield.store import MemoryStateStore
from fraudshield.synthetic import generate


def test_training_serving_parity():
    """Offline replay (training path) and per-transaction online path (get state -> features -> update) must match exactly."""
    df = generate(n_users=200, n_txn=3000, n_rings=2, seed=3).sort_values("timestamp").reset_index(drop=True)
    _, off, _ = build_offline(df, 3600.0, 86400.0, df.timestamp.iloc[-1])
    store, rows = MemoryStateStore(), []
    for r in df.itertuples(index=False):
        txn = row_to_txn(r)
        st = store.get_state(r.user_id)
        rows.append(txn_features(st, txn))
        store.put_state(r.user_id, update_state(st, txn))
    on = pd.DataFrame(rows)
    np.testing.assert_allclose(off[FEATURE_COLUMNS].to_numpy(float), on[FEATURE_COLUMNS].to_numpy(float), equal_nan=True)


def test_new_user_and_changes():
    t = dict(ts=1000.0, amount=100.0, device_id="d1", country="US", type="PAYMENT", balance_before=1000.0)
    f0 = txn_features(empty_state(), t)
    assert f0["is_new_user"] == 1 and f0["device_changed"] == 0 and f0["velocity_short"] == 0
    s1 = update_state(empty_state(), t)
    t2 = {**t, "ts": 1060.0, "device_id": "d2", "country": "NG", "amount": 900.0}
    f1 = txn_features(s1, t2)
    assert f1["device_geo_changed"] == 1 and f1["amount_vs_avg"] > 8
    assert math.isclose(f1["velocity_short"], math.exp(-60 / 3600), rel_tol=1e-9)


def test_welford_matches_numpy():
    s, xs = empty_state(), [10.0, 20.0, 35.0, 5.0, 50.0]
    for i, a in enumerate(xs):
        s = update_state(s, dict(ts=float(i), amount=a, device_id="d", country="US"))
    assert math.isclose(s["mean_amt"], np.mean(xs)) and math.isclose(s["m2_amt"] / (len(xs) - 1), np.var(xs, ddof=1))
