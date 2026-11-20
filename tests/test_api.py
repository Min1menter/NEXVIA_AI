import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from fraudshield import db as dbm
from fraudshield.api import create_app
from fraudshield.scoring import Scorer
from fraudshield.store import MemoryStateStore


class StubBundle:
    alpha, threshold, version = 0.5, 0.5, "test"

    def predict(self, feats):
        # first-ever txn of a user has no history -> low risk in this stub (ratio vs. a zero average is meaningless)
        p = 0.0 if feats["is_new_user"] else min(1.0, feats["amount_vs_avg"] / 20)
        return p, [dict(feature="amount_vs_avg", value=feats["amount_vs_avg"], contribution=p)]

    def combine(self, xgb_p, g_score, cluster):
        return xgb_p if g_score is None else 0.5 * xgb_p + 0.5 * g_score

    def reload(self):
        pass


PARAMS = {"features": {"tau_short_s": 3600, "tau_long_s": 86400}, "rules": {"ato_amount_ratio": 8.0}}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    store = MemoryStateStore(graph={"U1": (0.9, 7)})
    app = create_app(Scorer(store, StubBundle(), PARAMS), dbm.make_session_factory(f"sqlite:///{tmp_path}/t.db"))
    return TestClient(app)


def txn(i, amount, device="d1", country="US"):
    return dict(txn_id=f"T{i}", user_id="U1", timestamp=1_700_000_000 + i * 60, amount=amount, type="TRANSFER",
                device_id=device, country=country, balance_before=10_000)


def test_score_alert_case_workflow(client):
    for i in range(5):
        assert client.post("/score", json=txn(i, 100)).json()["decision"] == "allow"
    r = client.post("/score", json=txn(10, 5000, "new-dev", "NG")).json()
    assert r["decision"] == "alert" and r["alert_id"] and r["graph_score"] == 0.9
    assert any(x["feature"] == "rule:ato_pattern" for x in r["reasons"])
    alerts = client.get("/alerts", params={"status": "new"}).json()
    assert len(alerts) == 1
    upd = client.patch(f"/alerts/{r['alert_id']}", json={"status": "confirmed_fraud", "assignee": "ann"}).json()
    assert upd["status"] == "confirmed_fraud" and upd["feedback_label"] == 1
    s = client.get("/stats").json()
    assert s["reviewed"] == 1 and s["alert_precision"] == 1.0


def test_idempotent_and_validation(client):
    client.post("/score", json=txn(1, 100))
    client.post("/score", json=txn(1, 100))                      # duplicate delivery must not double count
    assert client.get("/stats").json()["scored"] == 1
    assert client.post("/score", json={**txn(2, -5)}).status_code == 422
    assert client.patch("/alerts/999", json={"status": "new"}).status_code == 404


def test_api_key_enforced(client, monkeypatch):
    # key is read at app creation; build a second app with a key
    store = MemoryStateStore()
    app = create_app(Scorer(store, StubBundle(), PARAMS), dbm.make_session_factory("sqlite://"))
    monkeypatch.setenv("API_KEY", "k")
    app2 = create_app(Scorer(store, StubBundle(), PARAMS), dbm.make_session_factory("sqlite://"))
    c = TestClient(app2)
    assert c.patch("/alerts/1", json={"status": "new"}).status_code == 401


def test_ring_endpoint(client, tmp_path, monkeypatch):
    import pandas as pd
    pd.DataFrame({"user_id": ["A", "B", "C", "D"], "cluster_id": [1, 1, 1, 2], "cluster_size": [3, 3, 3, 1]}).to_parquet(tmp_path / "rings.parquet")
    monkeypatch.setattr("fraudshield.api.DATA_DIR", tmp_path)
    r = client.get("/rings/A").json()
    assert r["size"] == 3 and set(r["members"]) == {"A", "B", "C"}
    assert client.get("/rings/ZZZ").status_code == 404
