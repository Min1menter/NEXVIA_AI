"""Tamper-evident audit log: every row stores sha256(prev_hash || canonical(record)).

Editing or deleting a row in the middle breaks the chain and `verify_chain` reports the
first bad id. (Deleting the *tail* is only detectable if you keep the latest head hash
somewhere else, e.g. export /api/audit/verify -> head_hash periodically.)
The DB additionally blocks UPDATE/DELETE/TRUNCATE with triggers (see db/init.sql).
"""
import hashlib
import json
from datetime import datetime, timezone

GENESIS = "0" * 64
FIELDS = ("ts", "actor", "action", "study_uid", "model_version", "ai_output", "reason")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def build_record(*, actor, action, study_uid=None, model_version=None, ai_output=None,
                 reason=None, ts=None) -> dict:
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "actor": actor,
        "action": action,
        "study_uid": study_uid,
        "model_version": model_version,
        "ai_output": None if ai_output is None else canonical(ai_output),
        "reason": reason,
    }


def compute_hash(prev_hash: str, record: dict) -> str:
    payload = prev_hash.encode() + canonical({k: record.get(k) for k in FIELDS}).encode()
    return hashlib.sha256(payload).hexdigest()


def append(conn, **kwargs) -> dict:
    """Append one entry inside the caller's transaction (serialized by an advisory lock)."""
    record = build_record(**kwargs)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(727001)")
        cur.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        prev = row[0] if row else GENESIS
        h = compute_hash(prev, record)
        cur.execute(
            "INSERT INTO audit_log (ts, actor, action, study_uid, model_version, ai_output, reason, prev_hash, hash) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (*[record[k] for k in FIELDS], prev, h),
        )
    return {**record, "prev_hash": prev, "hash": h}


def verify_chain(rows) -> dict:
    """rows: iterable of dicts ordered by id ASC with FIELDS + id, prev_hash, hash."""
    prev, n = GENESIS, 0
    for r in rows:
        n += 1
        if r["prev_hash"] != prev or compute_hash(prev, r) != r["hash"]:
            return {"ok": False, "checked": n, "first_bad_id": r["id"], "head_hash": None}
        prev = r["hash"]
    return {"ok": True, "checked": n, "first_bad_id": None, "head_hash": prev}
