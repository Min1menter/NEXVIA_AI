"""Replays the held-out test-period transactions into Kafka (time-compressed). Usage:
python -m fraudshield.producer --speed 3600 --limit 20000    (speed=3600 => 1 hour of data per second)"""
from __future__ import annotations
import argparse, json, os, time
import pandas as pd
from fraudshield.config import DATA_DIR, load_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=3600.0, help="data seconds per wall-clock second; 0 = as fast as possible")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    a = ap.parse_args()
    from confluent_kafka import Producer
    topic = load_params()["kafka"]["topic_in"]
    df = pd.read_parquet(DATA_DIR / "replay.parquet").sort_values("timestamp")
    if a.limit:
        df = df.head(a.limit)
    prod = Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 20})
    prev = None
    for r in df.itertuples(index=False):
        ts = r.timestamp.timestamp()
        if a.speed > 0 and prev is not None:
            time.sleep(min(max(ts - prev, 0) / a.speed, 2.0))
        prev = ts
        msg = dict(txn_id=r.txn_id, user_id=r.user_id, timestamp=ts, type=r.type, amount=r.amount, device_id=r.device_id,
                   card_id=r.card_id, country=r.country, merchant_id=r.merchant_id,
                   balance_before=None if pd.isna(r.balance_before) else r.balance_before)
        prod.produce(topic, key=r.user_id, value=json.dumps(msg))   # key => same partition => per-user ordering
        prod.poll(0)
    prod.flush()
    print(f"produced {len(df):,} transactions to {topic}")


if __name__ == "__main__":
    main()
