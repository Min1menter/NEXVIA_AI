"""Radiologist-facing API: auth, worklist, scan/overlay delivery, decisions, audit log."""
import os
import time
from pathlib import Path
from typing import Literal, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from api.auth import (DUMMY_HASH, User, create_token, current_user, hash_password,
                      require_role, verify_password)
from common import audit, db

DEBUG = os.environ.get("API_DEBUG") == "1"
app = FastAPI(title="MedImaging-AI API (education only)",
              docs_url="/api/docs" if DEBUG else None, redoc_url=None,
              openapi_url="/api/openapi.json" if DEBUG else None)

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
ORTHANC_AUTH = (os.environ.get("ORTHANC_USER", ""), os.environ.get("ORTHANC_PASSWORD", ""))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results")).resolve()

# ------------------------------------------------------------ login lockout
_FAILS: dict = {}
MAX_FAILS, WINDOW = 5, 900


def _locked(username: str) -> bool:
    now = time.time()
    recent = [t for t in _FAILS.get(username, []) if now - t < WINDOW]
    _FAILS[username] = recent
    return len(recent) >= MAX_FAILS


# ----------------------------------------------------------------- startup
@app.on_event("startup")
def seed_users():
    """Create users from env on first start (never overwrites existing users)."""
    seeds = [(os.environ.get("RADIOLOGIST_USER"), os.environ.get("RADIOLOGIST_PASSWORD"), "radiologist"),
             (os.environ.get("ADMIN_USER"), os.environ.get("ADMIN_PASSWORD"), "admin")]
    with db.connection() as conn, conn.cursor() as cur:
        for name, pw, role in seeds:
            if name and pw:
                cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s) "
                            "ON CONFLICT (username) DO NOTHING", (name, hash_password(pw), role))


@app.get("/api/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------------------------- auth
@app.post("/api/auth/login")
def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    name = form.username[:64]
    if _locked(name):
        raise HTTPException(429, "Too many failed attempts. Try again later.")
    with db.connection() as conn:
        with db.cursor(conn) as cur:
            cur.execute("SELECT password_hash, role FROM users WHERE username=%s", (name,))
            row = cur.fetchone()
        ok = verify_password(form.password, row["password_hash"] if row else DUMMY_HASH) and row is not None
        audit.append(conn, actor=name, action="login_success" if ok else "login_failure",
                     reason=f"ip={request.client.host if request.client else '?'}")
    if not ok:
        _FAILS.setdefault(name, []).append(time.time())
        raise HTTPException(401, "Incorrect username or password")
    _FAILS.pop(name, None)
    return {"access_token": create_token(name, row["role"]), "token_type": "bearer",
            "username": name, "role": row["role"]}


# ----------------------------------------------------------------- studies
def _get_study(uid: str) -> dict:
    with db.connection() as conn, db.cursor(conn) as cur:
        cur.execute("SELECT * FROM studies WHERE study_uid=%s", (uid,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Study not found")
    return row


@app.get("/api/studies")
def list_studies(status: Optional[str] = None, user: User = Depends(current_user)):
    with db.connection() as conn, db.cursor(conn) as cur:
        cur.execute(
            "SELECT study_uid, patient_id, modality, status, top_label, top_prob, model_version, "
            "reviewed_by, created_at FROM studies WHERE (%s::text IS NULL OR status=%s) "
            "ORDER BY created_at DESC LIMIT 200", (status, status))
        return cur.fetchall()


@app.get("/api/studies/{uid}")
def study_detail(uid: str, user: User = Depends(current_user)):
    row = _get_study(uid)
    with db.connection() as conn:
        audit.append(conn, actor=user.username, action="view_study", study_uid=uid,
                     model_version=row["model_version"])
    row["overlays"] = sorted((row["overlays"] or {}).keys())
    row.pop("instance_id", None)
    return row


@app.get("/api/studies/{uid}/image")
def study_image(uid: str, user: User = Depends(current_user)):
    row = _get_study(uid)
    if not row["instance_id"]:
        raise HTTPException(404, "No image for this study")
    r = requests.get(f"{ORTHANC_URL}/instances/{row['instance_id']}/preview", auth=ORTHANC_AUTH, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, "PACS error")
    return Response(r.content, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/studies/{uid}/overlay/{label}")
def study_overlay(uid: str, label: str, user: User = Depends(current_user)):
    fname = (_get_study(uid)["overlays"] or {}).get(label)
    path = (RESULTS_DIR / fname).resolve() if fname else None
    if not path or path.parent != RESULTS_DIR or not path.exists():
        raise HTTPException(404, "Overlay not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


class Decision(BaseModel):
    action: Literal["approve", "reject"]
    reason: str = Field("", max_length=1000)


@app.post("/api/studies/{uid}/decision")
def decide(uid: str, body: Decision, user: User = Depends(require_role("radiologist"))):
    if body.action == "reject" and not body.reason.strip():
        raise HTTPException(422, "A reason is required when rejecting")
    with db.connection() as conn:
        with db.cursor(conn) as cur:
            cur.execute("SELECT * FROM studies WHERE study_uid=%s FOR UPDATE", (uid,))
            s = cur.fetchone()
            if not s:
                raise HTTPException(404, "Study not found")
            if s["status"] != "pending_review":
                raise HTTPException(409, f"Study already {s['status']}")
            status = "approved" if body.action == "approve" else "rejected"
            cur.execute("UPDATE studies SET status=%s, reviewed_by=%s, reviewed_at=now() WHERE study_uid=%s",
                        (status, user.username, uid))
        entry = audit.append(conn, actor=user.username, action=f"decision_{body.action}", study_uid=uid,
                             model_version=s["model_version"], ai_output=s["predictions"],
                             reason=body.reason.strip())
    return {"status": status, "audit_hash": entry["hash"]}


# ------------------------------------------------------------------- audit
@app.get("/api/audit")
def audit_list(limit: int = 100, user: User = Depends(current_user)):
    limit = max(1, min(limit, 500))
    with db.connection() as conn, db.cursor(conn) as cur:
        cur.execute("SELECT id, ts, actor, action, study_uid, model_version, reason, hash "
                    "FROM audit_log ORDER BY id DESC LIMIT %s", (limit,))
        return cur.fetchall()


@app.get("/api/audit/verify")
def audit_verify(user: User = Depends(current_user)):
    with db.connection() as conn, db.cursor(conn) as cur:
        cur.execute("SELECT * FROM audit_log ORDER BY id ASC")
        return audit.verify_chain(cur.fetchall())
