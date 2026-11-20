# Threat model (course-project scope)

**FOR EDUCATION ONLY.** Data is public and de-identified, but the system is built as if it were not.

## Assets
Scans and AI results, radiologist decisions and the audit log, credentials/JWT secret, model weights.

## Architecture and trust boundaries
Browser -> **viewer (nginx, :8080)** -> `/api/*` -> **API (FastAPI)** -> **Postgres** and **Orthanc** (PACS).
The **inference worker** polls Orthanc, writes results to Postgres and a shared results volume. Orthanc and the viewer publish ports on `127.0.0.1` only.

## Implemented controls
| Threat (STRIDE) | Control in this repo |
|---|---|
| Spoofing | bcrypt password hashes; dummy-hash verify for unknown users (timing); per-username lockout (5 failures / 15 min); JWT (HS256) with 8 h expiry; `JWT_SECRET` must be >= 32 chars or the API refuses to run; Orthanc requires credentials. |
| Tampering | Audit log is hash-chained (`sha256(prev_hash + canonical(record))`) **and** guarded by DB triggers that block UPDATE/DELETE/TRUNCATE; `GET /api/audit/verify` reports the first bad row. Decisions use `SELECT ... FOR UPDATE` so a study cannot be decided twice. |
| Repudiation | Every login, study view, and Approve/Reject is logged with actor, time, study UID, model version, the AI output shown, and the reviewer's reason. Rejecting requires a reason. |
| Information disclosure | Role-checked endpoints; `Cache-Control: no-store` on images; overlay filenames are resolved and must sit directly in the results directory (path-traversal guard); API docs disabled unless `API_DEBUG=1`; CSP, `nosniff`, `X-Frame-Options: DENY` in nginx; DB/Orthanc not exposed beyond localhost. |
| Denial of service | Lockout limits password guessing; worker errors are caught per study so one bad file does not stop the loop. (No request rate limiting beyond login.) |
| Elevation of privilege | Only `radiologist` can decide; containers drop all Linux capabilities, set `no-new-privileges`, and the API container is read-only. |

## Known gaps (be upfront about these)
- **No TLS in the default compose.** Put a TLS-terminating reverse proxy in front before exposing anything beyond localhost.
- **JWT in `sessionStorage`.** An XSS bug would expose it; httpOnly cookies + CSRF protection would be stronger. The CSP reduces, not removes, that risk.
- **Lockout state is in memory** (lost on restart, per-process). No MFA.
- **Audit tail truncation** is detectable only if you record the `head_hash` somewhere else periodically. A DB superuser can drop the triggers (the hash chain still detects edits to existing rows).
- **Orthanc DICOM port (4242)** is unauthenticated by design; it is bound to localhost only. Do not expose it.
- **Model files are pickles.** Only load checkpoints you trained or trust; `models/` is mounted read-only into the worker.
- **Untrusted DICOM input** (huge or malformed files) is not size-limited in the worker.
- Secrets live in `.env` (git-ignored); use a secrets manager for anything real.
