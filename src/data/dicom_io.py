"""DICOM reading helpers shared by the inference service, tests and converters."""
import io

import numpy as np
from pydicom.multival import MultiValue


def _first(v) -> float:
    return float(v[0]) if isinstance(v, (list, tuple, MultiValue)) else float(v)


def dicom_to_uint8(ds) -> np.ndarray:
    """Dataset -> 2D uint8 grayscale (modality LUT, windowing, MONOCHROME1 inversion).
    Compressed transfer syntaxes need extra decoders (pylibjpeg / gdcm)."""
    arr = ds.pixel_array
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        arr = arr[0]
    arr = arr.astype(np.float32)
    if arr.ndim == 3:  # RGB -> gray
        arr = arr.mean(axis=-1)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None and _first(ww) > 1:
        lo, hi = _first(wc) - _first(ww) / 2, _first(wc) + _first(ww) / 2
    elif int(getattr(ds, "BitsStored", 16)) == 8:
        # Already display-ready 8-bit (e.g. a CheXpert JPEG wrapped as DICOM): do not re-stretch,
        # otherwise inference sees different intensities than the model saw in training.
        lo, hi = 0.0, 255.0
    else:
        lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = 1.0 - arr
    return (arr * 255).astype(np.uint8)


def load_ct_volume(datasets):
    """One axial CT series -> (vol[D,H,W] in HU, spacing (sz, sy, sx) mm, order).
    `order` is the permutation that sorts `datasets` bottom-to-top (use it to map slices back)."""
    def zpos(i):
        d = datasets[i]
        return float(d.ImagePositionPatient[2]) if "ImagePositionPatient" in d else float(d.InstanceNumber)

    order = sorted(range(len(datasets)), key=zpos)
    sl = [datasets[i] for i in order]
    vol = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
        + float(getattr(s, "RescaleIntercept", 0)) for s in sl
    ])
    sy, sx = (float(v) for v in sl[0].PixelSpacing)
    zs = [zpos(i) for i in order]
    sz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(getattr(sl[0], "SliceThickness", 1.0))
    return vol, (sz, sy, sx), order


# Axis conventions: DICOM array (D,H,W) = (z, row, col), rows run anterior->posterior and
# columns run to patient-left (LPS). MONAI/MSD training used RAS (x,y,z) arrays.
def dhw_to_ras_xyz(vol_dhw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vol_dhw.transpose(2, 1, 0)[::-1, ::-1, :])


def ras_xyz_to_dhw(arr_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr_xyz[::-1, ::-1, :].transpose(2, 1, 0))


def dicom_to_bytes(ds) -> bytes:
    buf = io.BytesIO()
    try:  # pydicom >= 3
        ds.save_as(buf, enforce_file_format=True)
    except TypeError:  # pydicom 2.x
        ds.save_as(buf, write_like_original=False)
    return buf.getvalue()
