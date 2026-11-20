import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from src.explain.gradcam import GradCAM, blend_over_gray, heatmap_rgba  # noqa: E402
from src.models.densenet_cls import build_model  # noqa: E402


def test_gradcam_shapes_and_range():
    model = build_model(14, pretrained=False).eval()
    cam = GradCAM(model)(torch.randn(1, 3, 224, 224), 3)
    assert cam.shape == (224, 224) and cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


def test_overlay_and_blend_shapes():
    cam = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    rgba = heatmap_rgba(cam, (100, 80))
    assert rgba.shape == (80, 100, 4) and rgba.dtype == np.uint8
    gray = np.zeros((80, 100), dtype=np.uint8)
    assert blend_over_gray(gray, rgba).shape == (80, 100, 3)
