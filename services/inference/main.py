"""Inference worker: poll Orthanc /changes -> run models -> store results (DB + overlays + SC DICOM).

Env: DATABASE_URL, ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD, MODEL_DIR, RESULTS_DIR, POLL_SECONDS
"""
import hashlib
import io
import logging
import os
import time
from pathlib import Path

import pydicom
from PIL import Image

from common import db
from inference.orthanc_client import Orthanc
from inference.pipeline import Pipeline, build_secondary_capture
from src.data.dicom_io import dicom_to_uint8, load_ct_volume

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "3"))


# ---------------------------------------------------------------- db helpers
def get_state(key, default="0"):
    with db.connection() as conn, db.cursor(conn) as cur:
        cur.execute("SELECT value FROM state WHERE key=%s", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def set_state(key, value):
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO state (key, value) VALUES (%s,%s) "
                    "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (key, str(value)))


def study_exists(uid) -> bool:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM studies WHERE study_uid=%s", (uid,))
        return cur.fetchone() is not None


def save_study(uid, orthanc_id, patient, modality, status, **kw):
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO studies (study_uid, orthanc_id, patient_id, modality, status, instance_id, "
            "model_version, predictions, top_label, top_prob, overlays, error) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (study_uid) DO NOTHING",
            (uid, orthanc_id, patient, modality, status, kw.get("instance_id"), kw.get("model_version"),
             db.Json(kw.get("predictions")), kw.get("top_label"), kw.get("top_prob"),
             db.Json(kw.get("overlays")), kw.get("error")),
        )


def save_png(uid: str, name: str, rgba) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{hashlib.sha1(uid.encode()).hexdigest()[:16]}_{name.lower().replace(' ', '_')}.png"
    Image.fromarray(rgba, "RGBA").save(RESULTS_DIR / fname)
    return fname


# ------------------------------------------------------------------ analysis
def analyse_xray(client, pipe, series, uid):
    candidates = series["Instances"][:4]
    best = candidates[0]
    for iid in candidates:  # prefer an AP/PA view
        if client.get(f"/instances/{iid}/simplified-tags").get("ViewPosition") in ("AP", "PA"):
            best = iid
            break
    ds = pydicom.dcmread(io.BytesIO(client.raw(f"/instances/{best}/file")))
    gray = dicom_to_uint8(ds)
    res = pipe.classify(gray)
    overlays = {label: save_png(uid, label, rgba) for label, rgba in res["overlays"].items()}
    try:  # write the result back to the PACS as a Secondary Capture
        client.upload(build_secondary_capture(ds, res["blended"], "AI Grad-CAM (EDUCATION ONLY)"))
    except Exception:
        log.exception("could not store Secondary Capture (non-fatal)")
    return dict(instance_id=best, model_version=pipe.cls_version, top_label=res["top_label"],
                top_prob=res["top_prob"], overlays=overlays,
                predictions={"kind": "classification", "scores": res["scores"]})


def analyse_ct(client, pipe, series, uid):
    ids = series["Instances"]
    if len(ids) < 10:
        raise ValueError(f"CT series too short ({len(ids)} slices)")
    dsets = [pydicom.dcmread(io.BytesIO(client.raw(f"/instances/{i}/file"))) for i in ids]
    vol, spacing, order = load_ct_volume(dsets)
    res = pipe.segment(vol, spacing)
    sorted_ids = [ids[i] for i in order]
    overlays = {"Spleen mask": save_png(uid, "spleen_mask", res["overlay"])}
    return dict(instance_id=sorted_ids[res["slice_index"]], model_version=pipe.seg_version,
                top_label="Spleen segmentation", top_prob=res["mean_confidence"], overlays=overlays,
                predictions={"kind": "segmentation", "structure": "spleen",
                             "volume_ml": res["volume_ml"], "mean_confidence": res["mean_confidence"],
                             "slice_index": res["slice_index"], "n_slices": len(ids),
                             "spacing_mm_zyx": list(spacing)})


def process_study(client, pipe, orthanc_id):
    info = client.get(f"/studies/{orthanc_id}")
    uid = info["MainDicomTags"]["StudyInstanceUID"]
    patient = info.get("PatientMainDicomTags", {}).get("PatientID")
    if study_exists(uid):  # also stops the Secondary Capture we upload from re-triggering us
        return
    series = client.get(f"/studies/{orthanc_id}/series", expand=1)
    by_mod = {}
    for s in series:
        by_mod.setdefault(s["MainDicomTags"].get("Modality"), []).append(s)
    modality = "CR" if ("CR" in by_mod or "DX" in by_mod) else ("CT" if "CT" in by_mod else None)
    try:
        if modality == "CR":
            s = (by_mod.get("CR") or by_mod.get("DX"))[0]
            result = analyse_xray(client, pipe, s, uid)
        elif modality == "CT":
            result = analyse_ct(client, pipe, max(by_mod["CT"], key=lambda s: len(s["Instances"])), uid)
        else:
            raise ValueError(f"unsupported modalities: {sorted(m for m in by_mod if m)}")
        save_study(uid, orthanc_id, patient, modality, "pending_review", **result)
        log.info("analysed %s (%s) -> %s %.2f", uid, modality, result["top_label"], result["top_prob"])
    except Exception as e:
        log.exception("analysis failed for %s", uid)
        save_study(uid, orthanc_id, patient, modality, "error", error=str(e)[:500])


def main():
    pipe = Pipeline(os.environ.get("MODEL_DIR", "/models"))
    client = Orthanc(os.environ.get("ORTHANC_URL", "http://orthanc:8042"),
                     os.environ["ORTHANC_USER"], os.environ["ORTHANC_PASSWORD"])
    since = int(get_state("orthanc_since", "0"))
    log.info("polling Orthanc from change #%d", since)
    while True:
        try:
            batch = client.changes(since)
            for ch in batch["Changes"]:
                if ch["ChangeType"] == "StableStudy":
                    process_study(client, pipe, ch["ID"])
            if batch["Last"] != since:
                since = batch["Last"]
                set_state("orthanc_since", since)
            if batch["Done"]:
                time.sleep(POLL_SECONDS)
        except Exception:
            log.exception("poll loop error; retrying")
            time.sleep(10)


if __name__ == "__main__":
    main()
