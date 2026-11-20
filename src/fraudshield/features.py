"""Feature engineering with ONE code path for training and serving.

Per-user state (8 scalars, stored in Feast) is updated incrementally:
  - Welford running mean/variance of amount          -> "amount vs user's average"
  - exponentially-decayed event counters (1h, 24h)   -> "transaction velocity"
  - last device / last country                       -> "device and location changes"
`txn_features(state_before, txn)` is the only place features are computed. Offline we replay history in time
order; online we read the state from Feast, call the same function, then push the updated state back.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from fraudshield.config import DATA_DIR, load_params

STATE_FIELDS = ["n_txn", "mean_amt", "m2_amt", "ew_cnt_short", "ew_cnt_long", "last_ts", "last_device", "last_country"]
TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
FEATURE_COLUMNS = (["log_amount", "amount_vs_avg", "amount_zscore", "drain_ratio", "velocity_short", "velocity_long",
                    "time_since_last", "is_new_user", "device_changed", "geo_changed", "device_geo_changed",
                    "hour", "dow", "amt_mean", "n_prior_txn"] + [f"type_{t}" for t in TYPES])
NEW_USER_GAP = 30 * 86400.0
UNKNOWN = {"", "unknown", "nan", "none", "-1"}      # missing values in real data (e.g. IEEE-CIS identity table)


def _known(v) -> bool:
    return str(v).strip().lower() not in UNKNOWN


def empty_state() -> dict:
    return dict(n_txn=0, mean_amt=0.0, m2_amt=0.0, ew_cnt_short=0.0, ew_cnt_long=0.0,
                last_ts=0.0, last_device="", last_country="")


def txn_features(state: dict, txn: dict, tau_s: float = 3600.0, tau_l: float = 86400.0) -> dict:
    """txn: ts (epoch s), amount, device_id, country, type, balance_before (may be NaN/None)."""
    n, ts, amt = int(state["n_txn"]), float(txn["ts"]), float(txn["amount"])
    dt = max(ts - state["last_ts"], 0.0) if n > 0 else NEW_USER_GAP
    std = math.sqrt(state["m2_amt"] / (n - 1)) if n > 1 else 0.0
    bal = txn.get("balance_before")
    bal = float("nan") if bal is None else float(bal)
    dev = int(n > 0 and _known(txn["device_id"]) and _known(state["last_device"]) and txn["device_id"] != state["last_device"])
    geo = int(n > 0 and _known(txn["country"]) and _known(state["last_country"]) and txn["country"] != state["last_country"])
    f = dict(
        log_amount=math.log1p(amt),
        amount_vs_avg=amt / (state["mean_amt"] + 1.0),
        amount_zscore=(amt - state["mean_amt"]) / (std + 1.0) if n >= 3 else 0.0,
        drain_ratio=amt / (bal + 1.0) if not math.isnan(bal) else float("nan"),
        velocity_short=state["ew_cnt_short"] * math.exp(-dt / tau_s) if n > 0 else 0.0,
        velocity_long=state["ew_cnt_long"] * math.exp(-dt / tau_l) if n > 0 else 0.0,
        time_since_last=dt, is_new_user=int(n == 0), device_changed=dev, geo_changed=geo,
        device_geo_changed=dev * geo, hour=float(int(ts // 3600) % 24), dow=float((int(ts // 86400) + 4) % 7),
        amt_mean=state["mean_amt"], n_prior_txn=float(n))
    for t in TYPES:
        f[f"type_{t}"] = int(txn["type"] == t)
    return f


def update_state(state: dict, txn: dict, tau_s: float = 3600.0, tau_l: float = 86400.0) -> dict:
    n, ts, amt = int(state["n_txn"]), float(txn["ts"]), float(txn["amount"])
    dt = max(ts - state["last_ts"], 0.0) if n > 0 else 0.0
    n1 = n + 1
    delta = amt - state["mean_amt"]
    mean = state["mean_amt"] + delta / n1
    return dict(n_txn=n1, mean_amt=mean, m2_amt=state["m2_amt"] + delta * (amt - mean),
                ew_cnt_short=state["ew_cnt_short"] * math.exp(-dt / tau_s) + 1.0 if n > 0 else 1.0,
                ew_cnt_long=state["ew_cnt_long"] * math.exp(-dt / tau_l) + 1.0 if n > 0 else 1.0,
                last_ts=max(ts, state["last_ts"]),
                last_device=str(txn["device_id"]) if _known(txn["device_id"]) else state["last_device"],
                last_country=str(txn["country"]) if _known(txn["country"]) else state["last_country"])


def row_to_txn(r) -> dict:
    return dict(ts=r.timestamp.timestamp(), amount=r.amount, device_id=r.device_id, country=r.country,
                type=r.type, balance_before=r.balance_before, card_id=r.card_id)


def build_offline(df: pd.DataFrame, tau_s: float, tau_l: float, cutoff: pd.Timestamp):
    """Replay all txns in time order. Returns (state-before snapshots, features, states at cutoff)."""
    states, snaps, feats, cut_states = {}, [], [], None
    for r in df.sort_values("timestamp").itertuples(index=False):
        if cut_states is None and r.timestamp > cutoff:
            cut_states = dict(states)
        st = states.get(r.user_id) or empty_state()
        txn = row_to_txn(r)
        snaps.append({"user_id": r.user_id, "event_timestamp": r.timestamp, **st})
        feats.append(txn_features(st, txn, tau_s, tau_l))
        states[r.user_id] = update_state(st, txn, tau_s, tau_l)
    return pd.DataFrame(snaps), pd.DataFrame(feats), (cut_states if cut_states is not None else states)


def main():
    p = load_params()
    df = pd.read_parquet(DATA_DIR / "raw" / "transactions.parquet").sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(n * p["data"]["train_frac"]), int(n * (p["data"]["train_frac"] + p["data"]["val_frac"]))
    df["split"] = np.where(np.arange(n) < i_tr, "train", np.where(np.arange(n) < i_va, "val", "test"))
    cutoff = df["timestamp"].iloc[i_va - 1]
    snaps, feats, cut_states = build_offline(df, p["features"]["tau_short_s"], p["features"]["tau_long_s"], cutoff)
    out = pd.concat([df[["txn_id", "timestamp", "user_id", "split", "is_fraud", "ring_id", "amount"]].reset_index(drop=True), feats], axis=1)
    out.to_parquet(DATA_DIR / "features.parquet", index=False)

    # Feast offline source: before-state rows up to cutoff + one "current state" row per user at cutoff
    final = pd.DataFrame([{"user_id": u, "event_timestamp": cutoff, **s} for u, s in cut_states.items()])
    final.to_parquet(DATA_DIR / "state_final.parquet", index=False)
    pd.concat([snaps[snaps["event_timestamp"] <= cutoff], final]).to_parquet(DATA_DIR / "state_snapshots.parquet", index=False)
    df[df["split"] == "test"].drop(columns="split").to_parquet(DATA_DIR / "replay.parquet", index=False)
    tr = out[out["split"] == "train"]
    tr.sample(min(20_000, len(tr)), random_state=p["seed"])[FEATURE_COLUMNS].to_parquet(DATA_DIR / "reference_features.parquet", index=False)
    pd.Series({"cutoff": str(cutoff)}).to_json(DATA_DIR / "cutoff.json")
    print(out["split"].value_counts().to_dict(), "| cutoff", cutoff)


if __name__ == "__main__":
    main()
