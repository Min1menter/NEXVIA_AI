"""Grad-CAM for torchvision DenseNet121 (no extra dependency).

torchvision's DenseNet.forward is: features -> relu -> avgpool -> classifier.
We replay that manually so we can keep the gradient of the `features` output
(= norm5) without fighting the in-place ReLU.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: (1,3,H,W) normalized. Returns cam (H,W) float32 in 0..1 at the input resolution."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)  # guarantees a graph even if params are frozen
        with torch.enable_grad():
            feats = self.model.features(x)
            feats.retain_grad()
            pooled = F.adaptive_avg_pool2d(F.relu(feats), (1, 1)).flatten(1)
            logit = self.model.classifier(pooled)[0, class_idx]
            logit.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats.detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy().astype(np.float32)


def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def heatmap_rgba(cam: np.ndarray, size, max_alpha: float = 0.6, threshold: float = 0.25) -> np.ndarray:
    """cam (0..1) -> uint8 RGBA array of shape (H,W,4); `size` is (W,H). Low activations are transparent."""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(tuple(size), Image.BICUBIC)
    v = np.asarray(cam_img, dtype=np.float32) / 255.0
    rgb = (_jet(v) * 255).astype(np.uint8)
    alpha = np.where(v < threshold, 0.0, (v - threshold) / (1 - threshold) * max_alpha)
    return np.dstack([rgb, (alpha * 255).astype(np.uint8)])


def blend_over_gray(gray_u8: np.ndarray, rgba_u8: np.ndarray) -> np.ndarray:
    """Composite an RGBA overlay on a uint8 grayscale image -> uint8 RGB array (H,W,3)."""
    base = Image.fromarray(gray_u8).convert("RGBA")
    over = Image.fromarray(rgba_u8, "RGBA").resize(base.size)
    return np.asarray(Image.alpha_composite(base, over).convert("RGB"))
