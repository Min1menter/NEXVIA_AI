"""Kafka consumer: transactions -> score -> DB -> fraud-scores topic. Failures go to a dead-letter topic.
Messages are keyed by user_id so per-user ordering (needed for stateful features) is preserved."""
from __future__ import annotations
import json, logging, os, signal, threading
from fraudshield import db as dbm
from fraudshield.config import load_params

log = logging.getLogger("consumer")


def run_consumer(scorer, session_factory, stop: threading.Event, bootstrap: str | None = None, group: str = "fraud-scorers"):
    from confluent_kafka import Consumer, Producer
    k = load_params()["kafka"]
    bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": group, "enable.auto.commit": False,
                  "auto.offset.reset": "earliest"})
    prod = Producer({"bootstrap.servers": bootstrap})
    c.subscribe([k["topic_in"]])
    log.info("consuming %s from %s", k["topic_in"], bootstrap)
    try:
        while not stop.is_set():
            msg = c.poll(1.0)
            if msg is None or msg.error():
                if msg is not None:
                    log.warning("kafka error: %s", msg.error())
                continue
            try:
                raw = json.loads(msg.value())
                r = scorer.score(raw, commit=True)
                with session_factory() as db:
                    r["alert_id"] = dbm.save_result(db, r)
                out = {x: r[x] for x in ("txn_id", "user_id", "score", "decision", "xgb_score", "graph_score", "model_version", "latency_ms")}
                prod.produce(k["topic_out"], key=r["user_id"], value=json.dumps(out))
                prod.poll(0)
            except Exception as e:                      # poison message -> DLQ, keep the stream alive
                log.exception("failed to process message")
                prod.produce(k["topic_dlq"], key=msg.key(), value=msg.value(), headers={"error": str(e)[:200]})
            c.commit(msg, asynchronous=False)
    finally:
        prod.flush(5)
        c.close()


def main():
    logging.basicConfig(level=logging.INFO)
    from fraudshield.scoring import build_scorer
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    run_consumer(build_scorer(), dbm.make_session_factory(), stop)


if __name__ == "__main__":
    main()
