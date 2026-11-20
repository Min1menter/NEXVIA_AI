"""Model loading + inference (chest X-ray classifier with Grad-CAM, optional CT spleen segmenter)."""
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.data.datasets import get_transforms
from src.data.dicom_io import dhw_to_ras_xyz, dicom_to_bytes, ras_xyz_to_dhw
from src.explain.gradcam import GradCAM, blend_over_gray, heatmap_rgba
from src.labels import LABELS, NON_FINDINGS
from src.models.densenet_cls import load_checkpoint

log = logging.getLogger("pipeline")
SC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
TOP_K_OVERLAYS = 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


class Pipeline:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        mdir = Path(model_dir)
        self.cls_model = self.cam = self.seg_model = None
        self.cls_version = self.seg_version = None

        cls_path = mdir / "cls_best.pt"
        if cls_path.exists():
            self.cls_model, ckpt = load_checkpoint(cls_path, self.device)
            self.cam = GradCAM(self.cls_model)
            self.size = ckpt.get("args", {}).get("img_size", 224)
            self.tf = get_transforms(self.size, train=False)
            self.cls_version = f"densenet121-{_sha(cls_path)}"
            log.info("loaded classifier %s", self.cls_version)
        else:
            log.warning("no classifier at %s", cls_path)

        seg_path = mdir / "seg_unet.pt"
        if seg_path.exists():
            from src.models.unet3d_monai import build_unet

            ckpt = torch.load(seg_path, map_location=self.device)
            self.seg_model = build_unet().to(self.device)
            self.seg_model.load_state_dict(ckpt["model"])
            self.seg_model.eval()
            self.seg_roi = tuple(ckpt.get("roi", (96, 96, 96)))
            self.seg_version = f"unet3d-{_sha(seg_path)}"
            log.info("loaded segmenter %s", self.seg_version)
        else:
            log.info("no segmenter at %s (CT studies will be flagged unsupported)", seg_path)

    # ---------------------------------------------------------------- X-ray
    def classify(self, gray_u8: np.ndarray) -> dict:
        if self.cls_model is None:
            raise RuntimeError("classifier checkpoint (cls_best.pt) not found in MODEL_DIR")
        h, w = gray_u8.shape
        x = self.tf(Image.fromarray(gray_u8)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.cls_model(x))[0].cpu().numpy()
        scores = {l: float(p) for l, p in zip(LABELS, probs)}
        findings = sorted(((l, p) for l, p in scores.items() if l not in NON_FINDINGS), key=lambda t: -t[1])
        overlays = {}
        for label, _ in findings[:TOP_K_OVERLAYS]:
            cam = self.cam(x, LABELS.index(label))
            overlays[label] = heatmap_rgba(cam, (w, h))
        top_label, top_prob = findings[0]
        return {"scores": scores, "top_label": top_label, "top_prob": top_prob,
                "overlays": overlays, "blended": blend_over_gray(gray_u8, overlays[top_label])}

    # ------------------------------------------------------------------- CT
    def segment(self, vol_dhw: np.ndarray, spacing_zyx) -> dict:
        if self.seg_model is None:
            raise RuntimeError("segmentation checkpoint (seg_unet.pt) not found in MODEL_DIR")
        from monai.data import MetaTensor
        from monai.inferers import sliding_window_inference
        from monai.transforms import ScaleIntensityRange, Spacing

        from src.data.msd import HU_MAX, HU_MIN, SPACING

        sz, sy, sx = spacing_zyx
        xyz = dhw_to_ras_xyz(vol_dhw).astype(np.float32)
        affine = torch.diag(torch.tensor([sx, sy, sz, 1.0]))
        mt = MetaTensor(torch.from_numpy(xyz)[None], affine=affine)
        mt = Spacing(pixdim=SPACING, mode="bilinear")(mt)
        mt = ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True)(mt)
        inp = mt.as_tensor().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = sliding_window_inference(inp, self.seg_roi, 4, self.seg_model)
            prob = torch.softmax(logits, dim=1)[0, 1]
            prob = F.interpolate(prob[None, None], size=xyz.shape, mode="trilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
        mask_dhw = ras_xyz_to_dhw(prob > 0.5)
        prob_dhw = ras_xyz_to_dhw(prob)
        n_vox = int(mask_dhw.sum())
        per_slice = mask_dhw.reshape(mask_dhw.shape[0], -1).sum(axis=1)
        k = int(per_slice.argmax()) if n_vox else mask_dhw.shape[0] // 2
        sl = mask_dhw[k]
        rgba = np.zeros((*sl.shape, 4), dtype=np.uint8)
        rgba[sl] = (255, 40, 40, 120)
        return {
            "slice_index": k,
            "overlay": rgba,
            "volume_ml": float(n_vox * sx * sy * sz / 1000.0),
            "mean_confidence": float(prob_dhw[mask_dhw].mean()) if n_vox else 0.0,
        }


def build_secondary_capture(src_ds, rgb_u8: np.ndarray, description: str) -> bytes:
    """Secondary Capture DICOM in the same study so the result is visible in any PACS viewer."""
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SC_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = SC_SOP_CLASS, sop_uid
    for tag in ("PatientName", "PatientID", "PatientSex", "StudyInstanceUID", "StudyDate",
                "StudyTime", "AccessionNumber", "StudyID"):
        if tag in src_ds:
            setattr(ds, tag, src_ds.get(tag))
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber, ds.InstanceNumber = "999", "1"
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.SeriesDescription = description
    ds.ImageComments = "AI output. EDUCATION ONLY - NOT FOR CLINICAL USE."
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = rgb_u8.shape[:2]
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(rgb_u8).tobytes()
    return dicom_to_bytes(ds)
