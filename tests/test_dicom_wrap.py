import numpy as np
import pytest
from PIL import Image

pydicom = pytest.importorskip("pydicom")
from src.data.dicom_io import dicom_to_uint8  # noqa: E402
from src.data.dicom_wrap import build_dicom, parse_path, save  # noqa: E402

REL = "CheXpert-v1.0-small/valid/patient00042/study1/view1_frontal.jpg"


@pytest.fixture
def img_path(tmp_path):
    p = tmp_path / "scan.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 48), dtype=np.uint8)).save(p)
    return p


def test_roundtrip_metadata_and_pixels(img_path, tmp_path):
    ds = build_dicom(img_path, REL, sex="Female", age=57, ap_pa="PA")
    out = tmp_path / "out.dcm"
    save(ds, out)
    back = pydicom.dcmread(out)
    assert back.Modality == "CR" and back.PatientID == "patient00042"
    assert back.PatientSex == "F" and back.PatientAge == "057Y" and back.ViewPosition == "PA"
    assert (back.Rows, back.Columns) == (64, 48)
    np.testing.assert_array_equal(back.pixel_array, np.asarray(Image.open(img_path).convert("L")))


def test_uids_are_deterministic_and_hierarchical(img_path):
    a = build_dicom(img_path, REL)
    b = build_dicom(img_path, REL)
    lateral = build_dicom(img_path, REL.replace("view1_frontal", "view2_lateral"))
    assert a.SOPInstanceUID == b.SOPInstanceUID and a.StudyInstanceUID == b.StudyInstanceUID
    assert a.StudyInstanceUID == lateral.StudyInstanceUID
    assert a.SOPInstanceUID != lateral.SOPInstanceUID


def test_bad_path_is_rejected():
    with pytest.raises(ValueError):
        parse_path("somewhere/else.jpg")


def test_8bit_wrapped_images_are_not_restretched(img_path):
    """Serving must feed the model the same intensities as training (no percentile stretch)."""
    ds = build_dicom(img_path, REL)
    np.testing.assert_array_equal(dicom_to_uint8(ds), ds.pixel_array)
