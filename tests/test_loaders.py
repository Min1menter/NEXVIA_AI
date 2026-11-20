"""Loaders are checked on tiny mock files that use the REAL column names of IEEE-CIS and PaySim."""
import numpy as np
import pandas as pd
from fraudshield import data as D
from fraudshield.features import build_offline, empty_state, txn_features, update_state
from fraudshield.graph import build_graph


def _check_schema(df):
    assert list(df.columns) == D.COLS
    assert df.timestamp.is_monotonic_increasing and df.timestamp.is_unique and str(df.timestamp.dt.tz) == "UTC"
    assert df[["device_id", "card_id", "country", "merchant_id"]].notna().all().all()
    assert df.is_fraud.isin([0, 1]).all()


def test_ieee_loader(tmp_path):
    n = 60
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"TransactionID": np.arange(n) + 2987000, "isFraud": (rng.random(n) < .1).astype(int),
                       "TransactionDT": np.sort(rng.integers(86400, 86400 * 90, n)), "TransactionAmt": rng.uniform(5, 500, n),
                       "ProductCD": rng.choice(list("WCRHS"), n), "card1": rng.integers(1000, 1010, n), "card2": np.where(rng.random(n) < .1, np.nan, 111.0),
                       "addr1": np.where(rng.random(n) < .1, np.nan, 300.0), "addr2": np.where(rng.random(n) < .1, np.nan, 87.0),
                       "P_emaildomain": rng.choice(["gmail.com", None], n), "R_emaildomain": rng.choice(["yahoo.com", None], n)})
    idn = pd.DataFrame({"TransactionID": tr.TransactionID[:15], "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G960F"], 15)})
    tr.to_csv(tmp_path / "t.csv", index=False); idn.to_csv(tmp_path / "i.csv", index=False)
    df = D.load_ieee({"transaction_csv": str(tmp_path / "t.csv"), "identity_csv": str(tmp_path / "i.csv")})
    _check_schema(df)
    assert len(df) == n and (df.device_id == "").sum() == n - 15            # missing identity -> "" (unknown)
    df2 = D.load_ieee({"transaction_csv": str(tmp_path / "t.csv"), "identity_csv": "does_not_exist.csv", "max_rows": 20})
    _check_schema(df2); assert len(df2) == 20


def test_paysim_loader(tmp_path):
    n = 80
    rng = np.random.default_rng(1)
    ps = pd.DataFrame({"step": np.sort(rng.integers(1, 50, n)), "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n),
                       "amount": rng.uniform(10, 9e5, n), "nameOrig": [f"C{i}" for i in rng.integers(0, 70, n)], "oldbalanceOrg": rng.uniform(0, 1e6, n),
                       "newbalanceOrig": 0.0, "nameDest": [f"M{i}" for i in rng.integers(0, 40, n)], "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                       "isFraud": (rng.random(n) < .05).astype(int), "isFlaggedFraud": 0})
    ps.to_csv(tmp_path / "p.csv", index=False)
    df = D.load_paysim({"csv": str(tmp_path / "p.csv"), "max_rows": 50}, seed=1)
    _check_schema(df); assert len(df) == 50 and df.balance_before.notna().all()
    _, feats, _ = build_offline(df, 3600.0, 86400.0, df.timestamp.iloc[-1])
    assert len(feats) == 50 and np.isfinite(feats["drain_ratio"]).all()


def test_unknown_device_or_country_is_not_a_change():
    t = dict(ts=1000.0, amount=10.0, device_id="iPhone", country="87", type="PAYMENT", balance_before=None)
    s = update_state(empty_state(), t)
    t2 = {**t, "ts": 1100.0, "device_id": "", "country": ""}
    f = txn_features(s, t2)
    assert f["device_changed"] == 0 and f["geo_changed"] == 0 and np.isnan(f["drain_ratio"])
    assert update_state(s, t2)["last_device"] == "iPhone"                  # unknown does not erase the last known device
    assert txn_features(s, {**t, "ts": 1100.0, "device_id": "Pixel"})["device_changed"] == 1


def test_graph_ignores_missing_keys_and_hubs():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"user_id": [f"U{i % 600}" for i in range(n)], "device_id": ["unknown"] * n,          # everyone "shares" unknown
                       "card_id": [f"C{i % 600}" for i in range(n)], "merchant_id": ["gmail.com"] * n,      # a hub shared by all
                       "amount": rng.uniform(1, 100, n), "type": "PAYMENT", "is_fraud": 0})
    g = build_graph(df, hub_limit=200)
    assert g["cluster_size"].max() == 1                                       # no giant cluster from missing/hub keys
    assert g["x"].shape[0] == 600 + 600                                       # only the user and card nodes exist
    assert len(g["cluster_id"]) == g["n_users"]
