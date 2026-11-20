import torch
import torch.nn as nn
from torchvision import models

from src.labels import LABELS


def build_model(n_classes: int = len(LABELS), pretrained: bool = True) -> nn.Module:
    """DenseNet121 with a fresh multi-label head (ImageNet transfer learning)."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    return m


def load_checkpoint(path, device):
    """Load a checkpoint written by src.train_cls -> (eval-mode model on `device`, ckpt dict)."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt.get("labels", LABELS)), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
