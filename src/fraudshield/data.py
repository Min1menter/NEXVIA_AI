"""Dataset loading. All sources are mapped to one schema:
txn_id, timestamp(UTC), user_id, type, amount, device_id, card_id, country, merchant_id, balance_before, is_fraud, ring_id"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fraudshield import synthetic
from fraudshield.config import DATA_DIR, ROOT, load_params

COLS = ["txn_id", "timestamp", "user_id", "type", "amount", "device_id", "card_id", "country",
        "merchant_id", "balance_before", "is_fraud", "ring_id"]


def _finish(out: pd.DataFrame, max_rows=None) -> pd.DataFrame:
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if max_rows:                                   # time-contiguous subsample: keeps the true fraud rate and temporal order
        out = out.head(int(max_rows)).copy()
    out["timestamp"] = out["timestamp"].dt.floor("s") + pd.to_timedelta(np.arange(len(out)), unit="us")   # unique timestamps
    out[["device_id", "card_id", "country", "merchant_id"]] = out[["device_id", "card_id", "country", "merchant_id"]].fillna("").astype(str)
    return out[COLS]


def load_ieee(p: dict) -> pd.DataFrame:
    """IEEE-CIS Fraud Detection (Kaggle competition). No user id exists: proxy = card1 + addr1 + purchaser email domain.
    Device = DeviceInfo (only present for ~24% of rows; missing = unknown, not a change). Dates anchored at 2017-12-01 (assumption)."""
    tr = pd.read_csv(ROOT / p["transaction_csv"])
    idp = ROOT / p.get("identity_csv", "")
    if p.get("identity_csv") and idp.exists():
        tr = tr.merge(pd.read_csv(idp, usecols=["TransactionID", "DeviceInfo"]), on="TransactionID", how="left")
    else:
        tr["DeviceInfo"] = np.nan
    base = pd.Timestamp("2017-12-01", tz="UTC")
    addr1 = tr["addr1"].fillna(-1).astype(int).astype(str)
    out = pd.DataFrame({
        "txn_id": tr["TransactionID"].astype(str),
        "timestamp": base + pd.to_timedelta(tr["TransactionDT"], unit="s"),
        "user_id": tr["card1"].astype(str) + "_" + addr1 + "_" + tr["P_emaildomain"].fillna("na").astype(str),
        "type": tr["ProductCD"].map({"W": "PAYMENT", "C": "TRANSFER", "R": "CASH_OUT", "H": "CASH_IN", "S": "DEBIT"}).fillna("PAYMENT"),
        "amount": tr["TransactionAmt"].astype(float),
        "device_id": tr["DeviceInfo"],
        "card_id": tr["card1"].astype(str) + "_" + tr["card2"].fillna(-1).astype(int).astype(str),
        "country": tr["addr2"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        "merchant_id": tr["R_emaildomain"],
        "balance_before": np.nan, "is_fraud": tr["isFraud"].astype(int), "ring_id": -1})
    return _finish(out, p.get("max_rows"))


def load_paysim(p: dict, seed: int) -> pd.DataFrame:
    """PaySim (Kaggle 'ealaxi/paysim1'). It has no device/card/geo and almost no repeat originators, so device/card/country
    are derived deterministically from a hash of the account (synthetic, documented limitation). Balance-based features
    (drain_ratio) are the real signal. `max_rows` keeps the first N rows in time order."""
    df = pd.read_csv(ROOT / p["csv"])
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    if p.get("max_rows"):
        df = df.head(int(p["max_rows"]))
    rng = np.random.default_rng(seed)
    h = pd.util.hash_pandas_object(df["nameOrig"], index=False).to_numpy()
    base = pd.Timestamp("2026-01-01", tz="UTC")
    out = pd.DataFrame({
        "txn_id": "P" + pd.Series(np.arange(len(df))).astype(str).str.zfill(8),
        "timestamp": base + pd.to_timedelta(df["step"], unit="h") + pd.to_timedelta(rng.integers(0, 3600, len(df)), unit="s"),
        "user_id": df["nameOrig"], "type": df["type"], "amount": df["amount"].astype(float),
        "device_id": "D" + pd.Series(h % 4_000_000).astype(str), "card_id": "C" + pd.Series(h % 4_000_000).astype(str),
        "country": np.array(synthetic.COUNTRIES[:5])[(h % 5).astype(int)],
        "merchant_id": df["nameDest"], "balance_before": df["oldbalanceOrg"].astype(float),
        "is_fraud": df["isFraud"].astype(int), "ring_id": -1})
    return _finish(out)


def load_raw(params: dict | None = None) -> pd.DataFrame:
    p = params or load_params()
    src = p["data"]["source"]
    if src == "synthetic":
        return synthetic.generate(seed=p["seed"], **p["data"]["synthetic"])
    if src == "ieee":
        return load_ieee(p["data"]["ieee"])
    if src == "paysim":
        return load_paysim(p["data"]["paysim"], p["seed"])
    raise ValueError(src)


def main():
    p = load_params()
    df = load_raw(p)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "raw" / "transactions.parquet", index=False)
    print(f"{len(df):,} txns | fraud rate {df.is_fraud.mean():.3%} | users {df.user_id.nunique():,}")


if __name__ == "__main__":
    main()
