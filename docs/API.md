# API reference

> FOR EDUCATION ONLY - NOT FOR CLINICAL USE.

Base path `/api`. In the default setup it is reached through the nginx viewer container at `http://localhost:8080/api/...`
(the API container itself is not published). Interactive OpenAPI docs are disabled unless the API runs with `API_DEBUG=1`.

**Auth:** `POST /api/auth/login` returns a JWT; send it as `Authorization: Bearer <token>` on every other call except `/api/health`.
Roles: `radiologist` (can decide) and `admin` (read-only here). Passwords are bcrypt-hashed; after 5 failed logins for a username within
15 minutes further attempts get `429` (lockout state is in memory).

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /api/health` | none | liveness, returns `{"status":"ok"}` |
| `POST /api/auth/login` | none | form-encoded `username`, `password` -> `{access_token, token_type, username, role}`; `401` bad credentials, `429` locked |
| `GET /api/studies?status=` | any user | worklist, newest first, max 200; optional `status` = `pending_review` / `approved` / `rejected` / `error` |
| `GET /api/studies/{uid}` | any user | full record (predictions, overlay labels, review info); **writes a `view_study` audit entry** |
| `GET /api/studies/{uid}/image` | any user | the scan as PNG (rendered by Orthanc `/preview`), `Cache-Control: no-store`; `404` unknown study, `502` PACS error |
| `GET /api/studies/{uid}/overlay/{label}` | any user | AI overlay PNG for a label listed in the study's `overlays`; `404` otherwise |
| `POST /api/studies/{uid}/decision` | `radiologist` | JSON `{action: "approve"|"reject", reason}`; reason required for reject (`422`); `409` if not pending; returns `{status, audit_hash}` |
| `GET /api/audit?limit=` | any user | latest audit entries (default 100, max 500), newest first |
| `GET /api/audit/verify` | any user | recomputes the hash chain -> `{ok, checked, first_bad_id, head_hash}` |

`{uid}` is the DICOM StudyInstanceUID. Study `status` values: `pending_review`, `approved`, `rejected`, `error`.

## Study record (abridged)
```json
{
  "study_uid": "1.3.6.1.4.1....",
  "patient_id": "patient64541",
  "modality": "CR",
  "status": "pending_review",
  "model_version": "densenet121-<hash>-ep7",
  "top_label": "Edema",
  "top_prob": 0.83,
  "predictions": {"kind": "classification", "scores": {"Edema": 0.83, "Pleural Effusion": 0.41, "...": 0.0}},
  "overlays": ["Edema", "Lung Opacity", "Pleural Effusion"],
  "reviewed_by": null,
  "created_at": "2026-11-05T10:12:03Z"
}
```
Classification studies carry up to three overlays (the top findings). CT studies carry `predictions.kind = "segmentation"` with
`volume_ml`, `mean_confidence`, `slice_index`, `n_slices`, and one overlay labelled `Spleen mask`.

## Audit entry
`audit_log(id, ts, actor, action, study_uid, model_version, ai_output, reason, prev_hash, hash)`. Actions recorded: `login_success`, `login_failure`,
`view_study`, `decision_approve`, `decision_reject`, plus the inference service's results. Each `hash` is
`sha256(prev_hash + canonical JSON of the record)`, and database triggers reject UPDATE, DELETE and TRUNCATE (see `db/init.sql`).

## Examples
```bash
TOKEN=$(curl -s -d "username=radiologist&password=..." http://localhost:8080/api/auth/login | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/studies | head -c 400
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"action":"reject","reason":"overlay on image border"}' http://localhost:8080/api/studies/<uid>/decision
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/audit/verify
```

## Error codes
`401` missing/invalid token or credentials - `403` role not allowed - `404` not found - `409` study already decided - `422` validation (e.g. empty reject reason) -
`429` login lockout - `502` Orthanc unreachable.

## Internal interfaces (not exposed to the browser)
- **Orthanc REST** (`:8042`, basic auth): `/changes`, `/studies/{id}`, `/series/{id}`, `/instances/{id}/file`, `/instances/{id}/preview`, `POST /instances`.
- **Inference worker -> database/volume:** writes the `studies` row and overlay PNGs into the shared results volume that the API serves.
