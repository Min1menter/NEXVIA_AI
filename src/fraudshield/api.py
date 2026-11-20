"""FastAPI scoring + case-management service.  `create_app(scorer, session_factory)` is injectable for tests."""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from fraudshield import db as dbm
import pandas as pd
from fraudshield.config import DATA_DIR, REPORTS_DIR, load_params

SCORED = Counter("fraud_scored_total", "Scored transactions", ["decision"])
LATENCY = Histogram("fraud_score_latency_seconds", "Scoring latency")


class TxnIn(BaseModel):
    txn_id: str
    user_id: str
    timestamp: Union[datetime, float]
    amount: float = Field(ge=0)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"] = "PAYMENT"
    device_id: str = ""
    card_id: str = ""
    country: str = ""
    merchant_id: str = ""
    balance_before: Optional[float] = None


class AlertUpdate(BaseModel):
    status: Optional[Literal["new", "investigating", "confirmed_fraud", "false_positive"]] = None
    assignee: Optional[str] = None
    feedback_label: Optional[Literal[0, 1]] = None
    note: Optional[str] = Field(None, max_length=2000)


def create_app(scorer=None, session_factory=None) -> FastAPI:
    app = FastAPI(title="FraudShield API", version="0.1.0")
    sf = session_factory or dbm.make_session_factory()
    holder = {"scorer": scorer}
    api_key = os.getenv("API_KEY")

    def get_db():
        with sf() as s:
            yield s

    def get_scorer():
        if holder["scorer"] is None:
            from fraudshield.scoring import build_scorer
            holder["scorer"] = build_scorer()
        return holder["scorer"]

    def auth(x_api_key: str | None = Header(None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.get("/health")
    def health():
        s = holder["scorer"]
        return {"status": "ok", "model_loaded": s is not None, "model_version": getattr(getattr(s, "bundle", None), "version", None)}

    @app.post("/score")
    def score(txn: TxnIn, commit: bool = True, db=Depends(get_db), sc=Depends(get_scorer)):
        raw = txn.model_dump()
        with LATENCY.time():
            r = sc.score(raw, commit=commit)
        if commit:
            r["alert_id"] = dbm.save_result(db, r)
        SCORED.labels(r["decision"]).inc()
        r.pop("features")
        return r

    @app.get("/alerts")
    def alerts(status: Optional[str] = None, min_score: float = 0.0, limit: int = 200, db=Depends(get_db)):
        return dbm.list_alerts(db, status, min_score, limit)

    @app.patch("/alerts/{alert_id}", dependencies=[Depends(auth)])
    def patch_alert(alert_id: int, body: AlertUpdate, db=Depends(get_db)):
        r = dbm.update_alert(db, alert_id, body.status, body.assignee, body.feedback_label, body.note)
        if r is None:
            raise HTTPException(404, "alert not found")
        return r

    @app.get("/rings/{user_id}")
    def ring(user_id: str, limit: int = 50):
        """Accounts linked to this user through shared devices/cards (a fraud-ring candidate)."""
        p = DATA_DIR / "rings.parquet"
        if not p.exists():
            raise HTTPException(404, "rings.parquet not found - run the graph step first")
        if holder.get("rings") is None or holder.get("rings_path") != str(p):
            holder["rings"], holder["rings_path"] = pd.read_parquet(p), str(p)
        df = holder["rings"]
        row = df[df.user_id == user_id]
        if row.empty:
            raise HTTPException(404, "user not in the graph")
        cid = int(row.cluster_id.iloc[0])
        members = df[df.cluster_id == cid]
        return {"user_id": user_id, "cluster_id": cid, "size": int(len(members)), "members": members.user_id.head(limit).tolist()}

    @app.get("/stats")
    def stats(db=Depends(get_db)):
        return dbm.stats(db)

    @app.get("/drift/latest")
    def drift_latest():
        p = REPORTS_DIR / "drift_latest.json"
        if not p.exists():
            raise HTTPException(404, "no drift report yet")
        import json
        return json.loads(p.read_text())

    @app.get("/drift/report.html")
    def drift_html():
        p = REPORTS_DIR / "drift_report.html"
        if not p.exists():
            raise HTTPException(404, "no report yet")
        return FileResponse(p, media_type="text/html")

    @app.post("/admin/reload", dependencies=[Depends(auth)])
    def reload(sc=Depends(get_scorer)):
        sc.bundle.reload()
        return {"model_version": sc.bundle.version}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def create_default() -> FastAPI:
    return create_app()
