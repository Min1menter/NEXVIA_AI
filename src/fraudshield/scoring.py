"""Real-time scorer: fetch state -> features -> XGBoost -> blend with graph score -> rules -> decision -> update state."""
from __future__ import annotations
import time
import pandas as pd
from fraudshield.config import load_params
from fraudshield.features import txn_features, update_state


def normalize_txn(raw: dict) -> dict:
    ts = raw["timestamp"]
    ts = pd.Timestamp(ts, tz="UTC").timestamp() if not isinstance(ts, (int, float)) else float(ts)
    return dict(txn_id=raw["txn_id"], user_id=raw["user_id"], ts=ts, amount=float(raw["amount"]),
                type=raw.get("type", "PAYMENT"), device_id=raw.get("device_id", ""), card_id=raw.get("card_id", ""),
                country=raw.get("country", ""), merchant_id=raw.get("merchant_id", ""),
                balance_before=raw.get("balance_before"))


class Scorer:
    def __init__(self, store, bundle, params: dict | None = None):
        self.store, self.bundle, self.p = store, bundle, params or load_params()
        self.tau_s, self.tau_l = self.p["features"]["tau_short_s"], self.p["features"]["tau_long_s"]

    def score(self, raw: dict, commit: bool = True) -> dict:
        t0 = time.perf_counter()
        txn = normalize_txn(raw)
        state = self.store.get_state(txn["user_id"])
        feats = txn_features(state, txn, self.tau_s, self.tau_l)
        xgb_p, reasons = self.bundle.predict(feats)
        graph = self.store.get_graph(txn["user_id"])
        g_score, cluster = (graph if graph else (None, None))
        final = self.bundle.combine(xgb_p, g_score, cluster)
        thr = self.bundle.threshold
        rule = (feats["amount_vs_avg"] >= self.p["rules"]["ato_amount_ratio"] and feats["device_geo_changed"] == 1)
        if rule:
            final = max(final, thr)
            reasons = [dict(feature="rule:ato_pattern", value=feats["amount_vs_avg"], contribution=0.0)] + reasons
        if g_score is not None and g_score >= 0.5:
            reasons.append(dict(feature="graph:ring_suspect", value=float(cluster or 0), contribution=g_score))
        if commit:
            self.store.put_state(txn["user_id"], update_state(state, txn, self.tau_s, self.tau_l))
        return dict(txn_id=txn["txn_id"], user_id=txn["user_id"], ts=txn["ts"], amount=txn["amount"], score=final,
                    xgb_score=xgb_p, graph_score=g_score, cluster_size=cluster, threshold=thr,
                    decision="alert" if final >= thr else "allow", reasons=reasons, features=feats,
                    model_version=self.bundle.version, latency_ms=(time.perf_counter() - t0) * 1000)


def build_scorer() -> Scorer:
    from fraudshield.model_bundle import ModelBundle
    from fraudshield.store import build_store
    return Scorer(build_store(), ModelBundle())
