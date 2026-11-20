"""PaySim-style synthetic transactions with device/card/country plus planted fraud:
(a) account-takeover bursts (tabular signal) and (b) fraud rings sharing devices/cards/mules (graph signal)."""
from __future__ import annotations
import numpy as np
import pandas as pd

COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "CN", "MX"]
TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


def generate(n_users=5000, n_txn=200_000, fraud_rate=0.008, n_rings=25, days=60, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    home = rng.choice(COUNTRIES[:5], n_users, p=[.45, .15, .15, .10, .15])
    mu = rng.normal(3.8, 0.6, n_users)
    wealth = np.exp(rng.normal(8.0, 1.0, n_users))
    act = rng.pareto(2.0, n_users) + 1
    p = act / act.sum()
    dev_p = np.array([f"D{u}-0" for u in range(n_users)], dtype=object)
    dev_s = np.array([f"D{u}-1" for u in range(n_users)], dtype=object)
    card_p = np.array([f"C{u}-0" for u in range(n_users)], dtype=object)
    card_s = np.array([f"C{u}-1" for u in range(n_users)], dtype=object)
    uid = np.array([f"U{u:06d}" for u in range(n_users)], dtype=object)

    n_fraud = int(n_txn * fraud_rate)
    n_ato = n_fraud // 2
    n_bursts = max(1, n_ato // 4)
    ring_sizes = rng.integers(5, 13, n_rings)
    ring_accounts = int(ring_sizes.sum())
    ring_benign_per = 8
    n_norm = n_txn - n_fraud - ring_accounts * ring_benign_per
    frames = []

    # ---- normal traffic
    u = rng.choice(n_users, n_norm, p=p)
    typ = rng.choice(TYPES, n_norm, p=[.45, .15, .15, .15, .10])
    amt = np.exp(rng.normal(mu[u], 0.5)) * np.where(np.isin(typ, ["TRANSFER", "CASH_IN"]), 1.5, 1.0)
    r = rng.random(n_norm)
    newdev = np.array([f"DX{i}" for i in rng.integers(0, 10**7, n_norm)], dtype=object)
    device = np.where(r < .94, dev_p[u], np.where(r < .99, dev_s[u], newdev))
    ctry = np.where(rng.random(n_norm) < .97, home[u], rng.choice(COUNTRIES, n_norm))
    card = np.where(rng.random(n_norm) < .9, card_p[u], card_s[u])
    ts = rng.integers(0, days, n_norm) * 86400 + np.mod(rng.normal(14 * 3600, 4 * 3600, n_norm), 86400)
    frames.append(pd.DataFrame(dict(
        ts=ts, user_id=uid[u], type=typ, amount=amt, device_id=device, card_id=card, country=ctry,
        merchant_id=[f"M{i:04d}" for i in rng.integers(0, 3000, n_norm)],
        balance_before=wealth[u] * np.exp(rng.normal(0, .3, n_norm)), is_fraud=0, ring_id=-1)))

    # ---- account-takeover bursts: new device + foreign country + large amount
    rows = []
    for b, v in enumerate(rng.integers(0, n_users, n_bursts)):
        t0 = rng.uniform(0, days * 86400)
        foreign = rng.choice([c for c in COUNTRIES if c != home[v]])
        for k in range(4):
            rows.append(dict(ts=t0 + rng.uniform(0, 1800), user_id=uid[v], type=rng.choice(["TRANSFER", "CASH_OUT"]),
                             amount=float(np.exp(mu[v] + rng.normal(1.6, .4))), device_id=f"DATO-{b}",
                             card_id=card_p[v] if rng.random() < .5 else f"CX{b}", country=foreign,
                             merchant_id=f"M{rng.integers(0, 3000):04d}",
                             balance_before=float(wealth[v] * np.exp(rng.normal(0, .3))), is_fraud=1, ring_id=-1))
    frames.append(pd.DataFrame(rows))

    # ---- fraud rings: members look normal individually but share devices, cards and mule counterparties
    n_ring_fraud = max(0, n_fraud - 4 * n_bursts)
    per_acct = max(1, round(n_ring_fraud / ring_accounts))
    rows = []
    for rid, size in enumerate(ring_sizes):
        devs = [f"DR{rid}-{j}" for j in range(rng.integers(1, 3))]
        cards = [f"CR{rid}-{j}" for j in range(rng.integers(2, 4))]
        mules = [f"MULE-{rid}-{j}" for j in range(rng.integers(1, 3))]
        for a in range(size):
            acct, hc, m = f"R{rid:03d}-{a:02d}", rng.choice(COUNTRIES[:5]), rng.normal(4.2, .3)
            for fraud, n in ((0, ring_benign_per), (1, per_acct)):
                for _ in range(n):
                    rows.append(dict(ts=rng.uniform(0, days * 86400), user_id=acct,
                                     type=rng.choice(["TRANSFER", "CASH_OUT"] if fraud else TYPES),
                                     amount=float(np.exp(m + rng.normal(0, .4)) * (1.3 if fraud else 1)),
                                     device_id=rng.choice(devs), card_id=rng.choice(cards), country=hc,
                                     merchant_id=rng.choice(mules) if fraud else f"M{rng.integers(0, 3000):04d}",
                                     balance_before=float(np.exp(rng.normal(8, .5))), is_fraud=fraud, ring_id=rid))
    frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    # strictly increasing timestamps (1 microsecond apart) => unambiguous point-in-time joins
    df["timestamp"] = start + pd.to_timedelta(df["ts"].to_numpy(), unit="s") + pd.to_timedelta(np.arange(len(df)), unit="us")
    df["txn_id"] = [f"T{i:08d}" for i in range(len(df))]
    df["is_fraud"] = df["is_fraud"].astype(int)
    cols = ["txn_id", "timestamp", "user_id", "type", "amount", "device_id", "card_id", "country",
            "merchant_id", "balance_before", "is_fraud", "ring_id"]
    return df[cols]
