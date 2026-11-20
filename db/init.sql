-- Runs once on first start of the Postgres container.
CREATE TABLE IF NOT EXISTS users (
  username      TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('radiologist','admin')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS studies (
  study_uid     TEXT PRIMARY KEY,
  orthanc_id    TEXT NOT NULL,
  instance_id   TEXT,                       -- Orthanc instance shown in the viewer
  patient_id    TEXT,
  modality      TEXT,
  status        TEXT NOT NULL DEFAULT 'pending_review'
                CHECK (status IN ('pending_review','approved','rejected','error')),
  model_version TEXT,
  predictions   JSONB,
  top_label     TEXT,
  top_prob      REAL,
  overlays      JSONB,                      -- {label: png filename in RESULTS_DIR}
  error         TEXT,
  reviewed_by   TEXT,
  reviewed_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS studies_status_idx ON studies (status, created_at DESC);

CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- Append-only, hash-chained audit log. ts/ai_output are TEXT so the hash is reproducible byte-for-byte.
CREATE TABLE IF NOT EXISTS audit_log (
  id            BIGSERIAL PRIMARY KEY,
  ts            TEXT NOT NULL,
  actor         TEXT NOT NULL,
  action        TEXT NOT NULL,
  study_uid     TEXT,
  model_version TEXT,
  ai_output     TEXT,
  reason        TEXT,
  prev_hash     TEXT NOT NULL,
  hash          TEXT NOT NULL UNIQUE
);

CREATE OR REPLACE FUNCTION audit_no_mutation() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_no_update ON audit_log;
CREATE TRIGGER audit_no_update BEFORE UPDATE OR DELETE ON audit_log
  FOR EACH ROW EXECUTE FUNCTION audit_no_mutation();
DROP TRIGGER IF EXISTS audit_no_truncate ON audit_log;
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log
  FOR EACH STATEMENT EXECUTE FUNCTION audit_no_mutation();
