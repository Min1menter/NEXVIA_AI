"""State stores. FeastStateStore = production path; MemoryStateStore = no-infra dev/test path.
Both expose the same interface so the Scorer is identical."""
from __future__ import annotations
import sys
from datetime import datetime, timezone
import pandas as pd
from fraudshield.config import DATA_DIR, FEATURE_REPO, load_params
from fraudshield.features import STATE_FIELDS, empty_state


class MemoryStateStore:
    def __init__(self, states=None, graph=None):
        self.states, self.graph = dict(states or {}), dict(graph or {})

    @classmethod
    def from_disk(cls):
        st = pd.read_parquet(DATA_DIR / "state_final.parquet")
        states = {r["user_id"]: {k: r[k] for k in STATE_FIELDS} for r in st.to_dict("records")}
        graph = {}
        gp = DATA_DIR / "graph_scores.parquet"
        if gp.exists():
            g = pd.read_parquet(gp)
            graph = {r["user_id"]: (float(r["graph_score"]), int(r["cluster_size"])) for r in g.to_dict("records")}
        return cls(states, graph)

    def get_state(self, user_id: str) -> dict:
        return dict(self.states.get(user_id) or empty_state())

    def put_state(self, user_id: str, state: dict) -> None:
        self.states[user_id] = dict(state)

    def get_graph(self, user_id: str):
        return self.graph.get(user_id)  # (score, cluster_size) or None


class FeastStateStore:
    def __init__(self, repo_path: str | None = None):
        from feast import FeatureStore
        self.fs = FeatureStore(repo_path=repo_path or FEATURE_REPO)

    def get_state(self, user_id: str) -> dict:
        r = self.fs.get_online_features(features=[f"user_state:{f}" for f in STATE_FIELDS],
                                        entity_rows=[{"user_id": user_id}]).to_dict()
        if r["n_txn"][0] is None:
            return empty_state()
        s = {f: r[f][0] for f in STATE_FIELDS}
        s["n_txn"] = int(s["n_txn"])
        return s

    def put_state(self, user_id: str, state: dict) -> None:
        from feast.data_source import PushMode
        row = {"user_id": user_id, "event_timestamp": datetime.fromtimestamp(state["last_ts"], tz=timezone.utc), **state}
        self.fs.push("user_state_push", pd.DataFrame([row]), to=PushMode.ONLINE)

    def get_graph(self, user_id: str):
        r = self.fs.get_online_features(features=["graph_scores:graph_score", "graph_scores:cluster_size"],
                                        entity_rows=[{"user_id": user_id}]).to_dict()
        if r["graph_score"][0] is None:
            return None
        return float(r["graph_score"][0]), int(r["cluster_size"][0])


def build_store():
    import os
    if os.getenv("FEATURE_BACKEND", "memory") == "feast":
        return FeastStateStore()
    return MemoryStateStore.from_disk()


def materialize():
    """Load the latest feature values into the online store (after `feast apply`)."""
    from feast import FeatureStore
    fs = FeatureStore(repo_path=FEATURE_REPO)
    cutoff = pd.Timestamp(pd.read_json(DATA_DIR / "cutoff.json", typ="series")["cutoff"]).to_pydatetime()
    fs.materialize(start_date=datetime(2000, 1, 1, tzinfo=timezone.utc), end_date=cutoff.replace(tzinfo=timezone.utc) + pd.Timedelta(days=1))
    print("materialized to online store")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "materialize":
        materialize()
