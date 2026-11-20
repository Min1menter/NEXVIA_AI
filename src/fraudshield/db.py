"""Persistence for scored transactions, analyst alerts/cases and feedback labels (SQLite dev / Postgres prod)."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import JSON, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

STATUSES = ["new", "investigating", "confirmed_fraud", "false_positive"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Scored(Base):
    __tablename__ = "scored"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    txn_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(8))
    features: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(32))


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    txn_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    xgb_score: Mapped[float] = mapped_column(Float)
    graph_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasons: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="new", index=True)
    assignee: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feedback_label: Mapped[int | None] = mapped_column(Integer, nullable=True)   # 1 fraud, 0 legit
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=now)
    updated_at: Mapped[str] = mapped_column(String(40), default=now)


def make_session_factory(url: str | None = None):
    url = url or os.getenv("DATABASE_URL", "sqlite:///./fraudshield.db")
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    engine = create_engine(url, **kw)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def save_result(db, r: dict) -> int | None:
    """Idempotent by txn_id (Kafka is at-least-once). Returns alert id if an alert was created."""
    if db.scalar(select(Scored.id).where(Scored.txn_id == r["txn_id"])):
        return None
    db.add(Scored(txn_id=r["txn_id"], user_id=r["user_id"], ts=r["ts"], amount=r["amount"], score=r["score"],
                  decision=r["decision"], features={k: (None if v != v else v) for k, v in r["features"].items()},
                  model_version=r["model_version"]))
    alert_id = None
    if r["decision"] == "alert":
        a = Alert(txn_id=r["txn_id"], user_id=r["user_id"], ts=r["ts"], amount=r["amount"], score=r["score"],
                  xgb_score=r["xgb_score"], graph_score=r["graph_score"], cluster_size=r["cluster_size"], reasons=r["reasons"])
        db.add(a)
        db.flush()
        alert_id = a.id
    db.commit()
    return alert_id


def alert_dict(a: Alert) -> dict:
    return {c.name: getattr(a, c.name) for c in Alert.__table__.columns}


def list_alerts(db, status=None, min_score=0.0, limit=200):
    q = select(Alert).where(Alert.score >= min_score).order_by(Alert.score.desc()).limit(limit)
    if status:
        q = q.where(Alert.status == status)
    return [alert_dict(a) for a in db.scalars(q)]


def update_alert(db, alert_id: int, status=None, assignee=None, feedback_label=None, note=None):
    a = db.get(Alert, alert_id)
    if a is None:
        return None
    if status:
        a.status = status
        if status == "confirmed_fraud" and feedback_label is None:
            feedback_label = 1
        if status == "false_positive" and feedback_label is None:
            feedback_label = 0
    if assignee is not None:
        a.assignee = assignee
    if feedback_label is not None:
        a.feedback_label = int(feedback_label)
    if note is not None:
        a.note = note
    a.updated_at = now()
    db.commit()
    return alert_dict(a)


def stats(db) -> dict:
    total = db.scalar(select(func.count(Scored.id))) or 0
    alerts = db.scalar(select(func.count(Alert.id))) or 0
    by = dict(db.execute(select(Alert.status, func.count(Alert.id)).group_by(Alert.status)).all())
    conf, fp = by.get("confirmed_fraud", 0), by.get("false_positive", 0)
    return dict(scored=total, alerts=alerts, alert_rate=alerts / total if total else 0.0, by_status=by,
                reviewed=conf + fp, alert_precision=conf / (conf + fp) if conf + fp else None)


def recent_features(db, n: int) -> pd.DataFrame:
    rows = db.scalars(select(Scored).order_by(Scored.id.desc()).limit(n)).all()
    return pd.DataFrame([r.features for r in rows])


def export_feedback(db) -> pd.DataFrame:
    """Analyst-labeled alerts joined to the features that were scored -> extra training rows."""
    q = select(Alert.txn_id, Alert.feedback_label, Scored.features).join(Scored, Scored.txn_id == Alert.txn_id).where(Alert.feedback_label.isnot(None))
    rows = db.execute(q).all()
    df = pd.DataFrame([{**f, "is_fraud": int(l)} for _, l, f in rows])
    if len(df):
        from fraudshield.features import FEATURE_COLUMNS
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df
