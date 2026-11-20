"""Wrap CheXpert/NIH PNG/JPG images into valid DICOM (CR) files.

Why: the public chest X-ray datasets ship as plain images, but the Orthanc
pipeline needs real DICOM. UIDs are deterministic (derived from patient/study/
view), so re-running the script produces identical files and Orthanc dedupes.

Usage (CheXpert-small):
    python -m src.data.dicom_wrap \
        --csv data/raw/CheXpert-v1.0-small/valid.csv \
        --root data/raw \
        --out data/dicom \
        --limit 200

--root is the folder that CONTAINS "CheXpert-v1.0-small", because the CSV
paths look like "CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg".

EDUCATIONAL USE ONLY. These files are not clinical images.
"""
import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1"  # Computed Radiography Image Storage
PATH_RE = re.compile(r"(patient\d+)/(study\d+)/(view\d+)_(frontal|lateral)")


def parse_path(rel_path: str):
    m = PATH_RE.search(rel_path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Unexpected CheXpert path: {rel_path}")
    return m.groups()  # patient, study, view, orientation


def build_dicom(img_path: Path, rel_path: str, sex=None, age=None, ap_pa=None) -> FileDataset:
    patient, study, view, orientation = parse_path(rel_path)

    img = Image.open(img_path).convert("L")
    pixels = np.asarray(img, dtype=np.uint8)
    rows, cols = pixels.shape

    # Deterministic UIDs
    study_uid = generate_uid(entropy_srcs=[patient, study])
    series_uid = generate_uid(entropy_srcs=[patient, study, orientation])
    sop_uid = generate_uid(entropy_srcs=[patient, study, view, orientation])

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CR_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(img_path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CR_SOP_CLASS
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid

    now = dt.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate, ds.ContentTime = ds.StudyDate, ds.StudyTime
    ds.AccessionNumber = f"{patient.replace('patient', 'P')}-{study.replace('study', 'S')}"[:16]  # SH VR max 16 chars
    ds.StudyID = study.replace("study", "")
    ds.SeriesNumber = "1"
    ds.InstanceNumber = view.replace("view", "")

    ds.Modality = "CR"
    ds.PatientName = patient
    ds.PatientID = patient
    ds.PatientSex = {"Male": "M", "Female": "F"}.get(str(sex), "O")
    if age is not None and not pd.isna(age):
        ds.PatientAge = f"{int(age):03d}Y"
    ds.ViewPosition = (str(ap_pa) if isinstance(ap_pa, str) and ap_pa in ("AP", "PA")
                       else ("PA" if orientation == "frontal" else "LL"))
    ds.StudyDescription = "CHEST X-RAY (CheXpert, educational)"
    ds.ImageComments = "Wrapped from public dataset. Educational use only, not clinical."

    # Pixel module (8-bit grayscale)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    return ds


def save(ds: FileDataset, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:  # pydicom >= 3
        ds.save_as(out_path, enforce_file_format=True)
    except TypeError:  # pydicom 2.x
        ds.save_as(out_path, write_like_original=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True, help="folder containing CheXpert-v1.0-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frontal-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.frontal_only:
        df = df[df["Frontal/Lateral"] == "Frontal"]
    if args.limit:
        df = df.head(args.limit)

    root, out = Path(args.root), Path(args.out)
    ok = 0
    for _, r in df.iterrows():
        img_path = root / r["Path"]
        if not img_path.exists():
            continue
        ds = build_dicom(img_path, r["Path"], r.get("Sex"), r.get("Age"), r.get("AP/PA"))
        patient, study, view, orient = parse_path(r["Path"])
        save(ds, out / patient / study / f"{view}_{orient}.dcm")
        ok += 1
    print(f"Wrote {ok} DICOM files to {out}")


if __name__ == "__main__":
    main()
