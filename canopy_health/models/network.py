"""Lightweight MobileNetV3 classifier definition."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from inference.fusion import CLASS_NAMES


def build_model(pretrained: bool = False) -> nn.Module:
    """Build MobileNetV3-Small with a three-class output layer."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model