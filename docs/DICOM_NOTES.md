# DICOM notes (what this project uses and why)

> FOR EDUCATION ONLY - NOT FOR CLINICAL USE.

DICOM is both a file format and a network protocol. A file is a list of tagged attributes (patient/study/series/image metadata)
plus the pixel data. Everything here is built on the attributes below.

## The identifier hierarchy
| Level | Key attribute | In this project |
|---|---|---|
| Patient | `PatientID` | CheXpert patient folder name, e.g. `patient64541` |
| Study | `StudyInstanceUID` | one visit/exam; the unit the worklist shows and the radiologist approves |
| Series | `SeriesInstanceUID` | one acquisition (a frontal film, or a CT volume) |
| Instance | `SOPInstanceUID` | one image (a CT series has one instance per slice) |

UIDs are globally unique strings. We generate them with `pydicom.uid.generate_uid(entropy_srcs=[...])` from the patient/study/view names,
so re-running a converter produces **identical UIDs** and Orthanc de-duplicates instead of storing copies.

## SOP classes (what kind of object it is)
- Chest X-ray: Computed Radiography Image Storage, `1.2.840.10008.5.1.4.1.1.1`, `Modality=CR`.
- CT: CT Image Storage, `1.2.840.10008.5.1.4.1.1.2`, `Modality=CT`.
- AI result written back to the PACS: Secondary Capture, `Modality=OT`, `ConversionType=WSD`, a new series inside the original study.
  The worker ignores studies it has already processed, so this extra series does not re-trigger analysis.

## Pixel data
- **Chest X-ray (from JPEG):** 8-bit, `MONOCHROME2`, `SamplesPerPixel=1`, `BitsAllocated=BitsStored=8`, `HighBit=7`, `PixelRepresentation=0`.
- **CT (from NIfTI):** signed 16-bit (`PixelRepresentation=1`), values in Hounsfield units via `RescaleSlope=1`, `RescaleIntercept=0`.
- `PhotometricInterpretation` says whether low values are dark (`MONOCHROME2`) or bright (`MONOCHROME1`). The reader inverts `MONOCHROME1`.
- Display value = `stored * RescaleSlope + RescaleIntercept`, then windowing (`WindowCenter`/`WindowWidth`) maps it to 0-255.
  Wrapped 8-bit images are passed through unchanged so inference sees the same intensities as training.

## CT geometry
A CT series is a stack of slices placed in patient space by:
- `ImagePositionPatient` - (x, y, z) in mm of the first pixel of this slice; z orders the slices.
- `ImageOrientationPatient` - direction cosines of the row and column axes.
- `PixelSpacing` (in-plane mm) and `SliceThickness` / the z spacing between slices.
DICOM uses the LPS patient coordinate system; the MSD/NIfTI training data used RAS. `src/data/dicom_io.py` contains the axis conversions
(`dhw_to_ras_xyz`, `ras_xyz_to_dhw`), and the segmentation volume in mL is voxel count x voxel volume from these spacings.

## Talking to Orthanc (REST)
- `GET /changes?since=N` lists events; the worker reacts to `StableStudy` (a study with no new instances for the stable-age window).
- `GET /studies/{id}`, `/series/{id}` walk the hierarchy; `GET /instances/{id}/file` returns the raw `.dcm`; `POST /instances` stores one.
- Orthanc uses its own internal IDs for these; the DICOM UIDs are in `MainDicomTags`.

## Limits of this setup
- Only uncompressed transfer syntax (Explicit VR Little Endian) is produced. Real-world compressed files (JPEG, JPEG 2000) need extra decoders.
- CheXpert-derived DICOMs carry only the metadata we write; they are not original scanner output.
- Source data is public and de-identified, but treat any DICOM as potentially identifying: nothing in the logs includes patient names.
