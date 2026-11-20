"""Convert an MSD NIfTI CT volume into a DICOM CT series (for Orthanc demos/tests).

    python -m src.data.nifti_to_dicom --nifti data/raw/msd/Task09_Spleen/imagesTr/spleen_2.nii.gz \
        --out data/dicom_ct/spleen_2 --patient spleen_2

Orientation: the volume is put in RAS-canonical order, then written as standard axial
LPS slices. src.data.dicom_io.dhw_to_ras_xyz is the exact inverse (covered by a test).
"""
import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"


def volume_to_slices(data_xyz: np.ndarray, spacing_xyz, patient: str):
    """data_xyz: RAS array (x,y,z) in HU. Yields one pydicom FileDataset per axial slice."""
    sx, sy, sz = spacing_xyz
    study_uid = generate_uid(entropy_srcs=[patient, "study"])
    series_uid = generate_uid(entropy_srcs=[patient, "series"])
    frame_uid = generate_uid(entropy_srcs=[patient, "frame"])
    for k in range(data_xyz.shape[2]):
        pix = data_xyz[::-1, ::-1, k].T  # (rows=y, cols=x), LPS
        pix = np.clip(np.round(pix), -32768, 32767).astype(np.int16)
        sop_uid = generate_uid(entropy_srcs=[patient, "sop", str(k)])
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_SOP_CLASS
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CT_SOP_CLASS, sop_uid
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PatientID, ds.PatientName = patient, patient
        ds.Modality = "CT"
        ds.StudyDescription = "CT (MSD, educational)"
        ds.SeriesDescription = "Axial"
        ds.StudyDate = ds.ContentDate = "20260101"
        ds.StudyTime = ds.ContentTime = "000000"
        ds.InstanceNumber = str(k + 1)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(k * sz)]
        ds.SliceLocation = float(k * sz)
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.SliceThickness = float(sz)
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = pix.shape
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = pix.tobytes()
        yield ds


def main():
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient", default="ct_demo")
    args = ap.parse_args()

    img = nib.as_closest_canonical(nib.load(args.nifti))
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(img.affine))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, ds in enumerate(volume_to_slices(data, spacing, args.patient)):
        try:
            ds.save_as(out / f"slice_{k:04d}.dcm", enforce_file_format=True)
        except TypeError:
            ds.save_as(out / f"slice_{k:04d}.dcm", write_like_original=False)
        n += 1
    print(f"Wrote {n} slices to {out}")


if __name__ == "__main__":
    main()
